// Copyright 2026 Candace Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Read one mined task directory into [`Task`]s. Every file past task.json is
//! optional: retry.json stands in for a task still waiting for a retry, and
//! fairness/<stage>.json, instruction.md and ../exam.jsonl appear only once
//! the fairness stages have run.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// Fairness stages in pipeline order; `gate` is the verdict over the others.
pub const STAGES: [&str; 6] = ["flake", "api", "describe", "probe", "specificity", "gate"];

/// Longest patch embedded per task, in bytes; the rest is cut with a note.
pub const MAX_PATCH: usize = 100_000;
/// Lines of each go test log embedded per task.
pub const LOG_TAIL_LINES: usize = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Evolve,
    Heldout,
    Excluded,
}

#[derive(Clone, Debug, Serialize)]
pub struct Task {
    pub sha12: String,
    pub sha: String,
    pub subject: String,
    pub body: String,
    pub module_root: String,
    pub packages: Vec<String>,
    pub src_files: Vec<String>,
    pub test_files: Vec<String>,
    pub src_churn: u64,
    pub valid: bool,
    pub reason: String,
    /// Only a retry.json exists: the last run was infra or a timeout.
    pub pending_retry: bool,
    pub parent_outcome: Option<String>,
    pub commit_outcome: Option<String>,
    /// fairness/<stage>.json by stage name, verbatim.
    pub fairness: BTreeMap<String, Value>,
    pub instruction: Option<String>,
    pub src_patch: String,
    pub tests_patch: String,
    pub parent_log_tail: String,
    pub commit_log_tail: String,
    pub area: String,
    /// Committer date, ISO 8601, when --repo knows the commit.
    pub date: Option<String>,
    /// Committer date as unix seconds (orders the split).
    pub ts: Option<i64>,
    pub split: Split,
    pub excluded_reason: Option<String>,
}

impl Task {
    /// The verdict of one fairness stage, when it ran and said.
    pub fn stage_pass(&self, stage: &str) -> Option<bool> {
        self.fairness.get(stage).and_then(|v| v.get("pass")).and_then(Value::as_bool)
    }

    /// Fairness has run for this task once its gate verdict exists.
    pub fn fairness_ran(&self) -> bool {
        self.fairness.contains_key("gate")
    }
}

fn str_field(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn list_field(v: &Value, k: &str) -> Vec<String> {
    v.get(k).and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn read_opt(p: &Path) -> Option<String> {
    std::fs::read(p).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn cap(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut cut = max;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        let total = s.len();
        s.truncate(cut);
        s.push_str(&format!("\n... [truncated: {cut} of {total} bytes shown]\n"));
    }
    s
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let from = lines.len().saturating_sub(n);
    let mut out = String::new();
    if from > 0 {
        out.push_str(&format!("... [{from} earlier lines]\n"));
    }
    out.push_str(&lines[from..].join("\n"));
    cap(out, 16_000)
}

/// The report's coarse "area" of a task: the first three path segments of
/// its primary package, repository-relative (module root joined with the
/// package), e.g. `svc/services/warden` for module `svc`, package
/// `./services/warden/election`.
pub fn area(module_root: &str, packages: &[String]) -> String {
    let pkg = packages.first().map(String::as_str).unwrap_or(".");
    let segs: Vec<&str> = module_root.split('/').chain(pkg.split('/'))
        .filter(|s| !s.is_empty() && *s != ".").take(3).collect();
    if segs.is_empty() { "(root)".to_string() } else { segs.join("/") }
}

/// Load one task directory, or `None` when it holds neither task.json nor
/// retry.json.
pub fn load_task(dir: &Path) -> Result<Option<Task>> {
    let (meta_path, pending_retry) = if dir.join("task.json").is_file() {
        (dir.join("task.json"), false)
    } else if dir.join("retry.json").is_file() {
        (dir.join("retry.json"), true)
    } else {
        return Ok(None);
    };
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)
        .with_context(|| format!("parsing {}", meta_path.display()))?;
    let sha = str_field(&meta, "sha");
    let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
    let sha12 = if sha.len() >= 12 { sha[..12].to_string() } else { dir_name };
    let mut fairness = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(dir.join("fairness")) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Some(text) = read_opt(&p) else { continue };
            let v: Value = serde_json::from_str(&text).unwrap_or_else(|err| {
                serde_json::json!({"pass": false, "reason": format!("unreadable {}: {err}", p.display())})
            });
            let stage = v.get("stage").and_then(Value::as_str).map(str::to_string)
                .or_else(|| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
                .unwrap_or_default();
            fairness.insert(stage, v);
        }
    }
    let module_root = str_field(&meta, "module_root");
    let packages = list_field(&meta, "packages");
    Ok(Some(Task {
        area: area(&module_root, &packages),
        sha12,
        sha,
        subject: str_field(&meta, "subject"),
        body: str_field(&meta, "body"),
        module_root,
        packages,
        src_files: list_field(&meta, "src_files"),
        test_files: list_field(&meta, "test_files"),
        src_churn: meta.get("src_churn").and_then(Value::as_u64).unwrap_or(0),
        valid: !pending_retry && meta.get("valid").and_then(Value::as_bool).unwrap_or(false),
        reason: str_field(&meta, "reason"),
        pending_retry,
        parent_outcome: meta.get("parent_outcome").and_then(Value::as_str).map(str::to_string),
        commit_outcome: meta.get("commit_outcome").and_then(Value::as_str).map(str::to_string),
        fairness,
        instruction: read_opt(&dir.join("instruction.md")),
        src_patch: cap(read_opt(&dir.join("src.patch")).unwrap_or_default(), MAX_PATCH),
        tests_patch: cap(read_opt(&dir.join("tests.patch")).unwrap_or_default(), MAX_PATCH),
        parent_log_tail: tail(&read_opt(&dir.join("parent.log")).unwrap_or_default(), LOG_TAIL_LINES),
        commit_log_tail: tail(&read_opt(&dir.join("commit.log")).unwrap_or_default(), LOG_TAIL_LINES),
        date: None,
        ts: None,
        split: Split::Excluded,
        excluded_reason: None,
    }))
}

/// Every task under `dir`, ordered by directory name.
pub fn load_tasks(dir: &Path) -> Result<Vec<Task>> {
    let mut dirs: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path())
        .filter(|p| p.is_dir()).collect();
    dirs.sort();
    let mut out = Vec::new();
    for d in dirs {
        if let Some(t) = load_task(&d)? {
            out.push(t);
        }
    }
    Ok(out)
}

/// Lines in `<tasks>/exam.jsonl`, when the fairness run has written it.
pub fn exam_lines(dir: &Path) -> Option<usize> {
    read_opt(&dir.join("exam.jsonl")).map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
}

/// Fill `date`/`ts` from the repository's committer dates; returns how many
/// tasks it could not date.
pub fn attach_dates(tasks: &mut [Task], repo: &Path) -> usize {
    let mut missing = 0;
    for t in tasks.iter_mut() {
        let out = Command::new("git").arg("-C").arg(repo)
            .args(["show", "-s", "--format=%ct %cI", &t.sha]).output();
        let line = match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            _ => String::new(),
        };
        match line.split_once(' ').and_then(|(ts, iso)| Some((ts.parse().ok()?, iso))) {
            Some((ts, iso)) => {
                t.ts = Some(ts);
                t.date = Some(iso.to_string());
            }
            None => missing += 1,
        }
    }
    missing
}
