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

//! Real examples for the report's "Start here" explainer. The explainer's
//! prose is generic and lives in report/report.html; this module only picks
//! tasks and log lines from the data at render time, so nothing specific to
//! one repository is ever committed.

use super::health::{Health, Status};
use super::load::{Split, Task};
use super::split::Summary;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

/// The log fragment Go prints both for a failed module download and for a
/// test importing a package the old tree does not have yet.
pub const GOPROXY_MARKER: &str = "module lookup disabled by GOPROXY=off";

/// Lines of hidden test shown in the worked example.
pub const EXCERPT_LINES: usize = 14;

#[derive(Debug, Serialize)]
pub struct WorkedExample {
    pub sha12: String,
    pub subject: String,
    pub split: Split,
    pub instruction: Option<String>,
    pub test_file: String,
    pub test_excerpt: String,
    pub parent_outcome: Option<String>,
    pub parent_failure: String,
    pub commit_ok: String,
}

#[derive(Debug, Serialize)]
pub struct TaskRef {
    pub sha12: String,
    pub subject: String,
    pub detail: String,
}

#[derive(Debug, Serialize)]
pub struct Stage {
    pub name: &'static str,
    /// "done" | "partial" | "next" | "not started"
    pub status: &'static str,
    pub detail: String,
}

#[derive(Debug, Serialize)]
pub struct Start {
    pub headline: Headline,
    pub worked: Option<WorkedExample>,
    pub already_passes: Option<TaskRef>,
    pub fails_on_commit: Option<TaskRef>,
    /// A real log line holding [`GOPROXY_MARKER`], and how many logs hold it.
    pub goproxy_line: Option<String>,
    pub goproxy_logs: usize,
    pub pending_retry: usize,
    pub leak: Option<TaskRef>,
    pub missing_name: Option<TaskRef>,
    pub flaky: Option<TaskRef>,
    pub health_warnings: Vec<String>,
    pub pipeline: Vec<Stage>,
}

/// `s` without ANSI colour escapes.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' && it.peek() == Some(&'[') {
            it.next();
            for d in it.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn short(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect::<String>() + "..."
    }
}

/// The most telling failure in a parent log: an assertion message, a
/// missing name, or the failing test, with a few lines of context.
pub fn failure_excerpt(log: &str) -> String {
    let log = strip_ansi(log);
    let lines: Vec<&str> = log.lines().collect();
    let pick = |pred: &dyn Fn(&str) -> bool, ctx: usize| -> Option<String> {
        let i = lines.iter().position(|l| pred(l))?;
        let mut out: Vec<&str> = Vec::new();
        for l in &lines[i..(i + ctx).min(lines.len())] {
            if l.trim_start().starts_with("----") {
                break;
            }
            out.push(l.trim_end());
        }
        Some(out.join("\n"))
    };
    pick(&|l| l.contains("[FAILED] Expected") || l.contains("Error:") && l.contains("_test.go"), 6)
        .or_else(|| pick(&|l| l.contains("undefined: "), 3))
        .or_else(|| pick(&|l| l.contains("--- FAIL"), 4))
        .or_else(|| pick(&|l| l.contains("[build failed]") || l.starts_with("FAIL"), 2))
        .unwrap_or_default()
}

/// The `ok  <package>` line of a passing run.
pub fn ok_line(log: &str) -> String {
    strip_ansi(log).lines().find(|l| l.starts_with("ok ")).unwrap_or("ok").trim_end().to_string()
}

/// A readable window of the hidden tests: the test the parent log names as
/// failing when it can be found, else the first test function or spec.
pub fn test_excerpt(tests_patch: &str, parent_log: &str) -> (String, String) {
    let log = strip_ansi(parent_log);
    let mut needles: Vec<String> = Vec::new();
    for l in log.lines() {
        if let Some(i) = l.find("[It] ") {
            needles.push(l[i + 5..].trim().to_string());
        }
        if let Some(i) = l.find("undefined: ") {
            needles.push(l[i + 11..].trim().to_string());
        }
        if let Some(i) = l.find("--- FAIL: ") {
            if let Some(name) = l[i + 10..].split_whitespace().next() {
                needles.push(format!("func {name}("));
            }
        }
    }
    needles.extend(["func Test".to_string(), "It(\"".to_string()]);
    // (file, added line) in patch order.
    let mut file = String::new();
    let mut added: Vec<(String, String)> = Vec::new();
    for l in tests_patch.lines() {
        if let Some(f) = l.strip_prefix("+++ b/") {
            file = f.to_string();
        } else if let Some(a) = l.strip_prefix('+') {
            added.push((file.clone(), a.to_string()));
        }
    }
    let start = needles.iter().filter(|n| !n.is_empty())
        .find_map(|n| added.iter().position(|(_, a)| a.contains(n.as_str())))
        .unwrap_or(0);
    let Some((f, _)) = added.get(start) else { return (String::new(), String::new()) };
    let body: Vec<&str> = added[start..].iter().take_while(|(g, _)| g == f)
        .take(EXCERPT_LINES).map(|(_, a)| a.as_str()).collect();
    (f.clone(), body.join("\n"))
}

fn tref(t: &Task, detail: String) -> TaskRef {
    TaskRef { sha12: t.sha12.clone(), subject: t.subject.clone(), detail }
}

fn added_test_lines(t: &Task) -> usize {
    t.tests_patch.lines().filter(|l| l.starts_with('+') && !l.starts_with("+++")).count()
}

fn stage_reason(t: &Task, s: &str) -> String {
    t.fairness.get(s).and_then(|v| v.get("reason")).and_then(Value::as_str).unwrap_or("failed").to_string()
}

/// Pick the worked example: an evolve task (else held-out) whose tests fail
/// with an assertion, small enough to read; ones with an instruction first.
pub fn pick_worked(tasks: &[Task]) -> Option<&Task> {
    let readable = |t: &&Task| (5..=60).contains(&t.src_churn) && (8..=80).contains(&added_test_lines(t));
    let mut ready: Vec<&Task> = tasks.iter().filter(|t| t.split != Split::Excluded).collect();
    ready.sort_by_key(|t| (
        t.instruction.is_none(),
        !readable(t),
        t.parent_outcome.as_deref() != Some("TestFail"),
        t.split != Split::Evolve,
        t.src_churn,
        t.sha.clone(),
    ));
    ready.into_iter().next()
}

fn read_log(dir: &Path, t: &Task, name: &str) -> String {
    std::fs::read(dir.join(&t.sha12).join(name))
        .map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

pub fn start(tasks: &[Task], dir: &Path, summary: &Summary, health: &Health) -> Start {
    let worked = pick_worked(tasks).map(|t| {
        let parent = read_log(dir, t, "parent.log");
        let (test_file, test_excerpt) = test_excerpt(&t.tests_patch, &parent);
        WorkedExample {
            sha12: t.sha12.clone(), subject: t.subject.clone(), split: t.split,
            instruction: t.instruction.clone(), test_file, test_excerpt,
            parent_outcome: t.parent_outcome.clone(),
            parent_failure: failure_excerpt(&parent),
            commit_ok: ok_line(&read_log(dir, t, "commit.log")),
        }
    });
    let smallest = |pred: &dyn Fn(&Task) -> bool| tasks.iter().filter(|t| pred(t))
        .min_by_key(|t| (t.src_churn, t.sha.clone()));
    let already_passes = smallest(&|t| !t.valid && t.reason.contains("already pass"))
        .map(|t| tref(t, format!("{} \u{2192} {}", t.parent_outcome.as_deref().unwrap_or("?"),
                                  t.commit_outcome.as_deref().unwrap_or("?"))));
    let fails_on_commit = smallest(&|t| !t.valid && t.reason.contains("fail on the commit"))
        .map(|t| tref(t, short(&failure_excerpt(&read_log(dir, t, "commit.log")), 300)));
    let mut goproxy_line = None;
    let mut goproxy_logs = 0;
    for t in tasks {
        for name in ["parent.log", "commit.log"] {
            let log = read_log(dir, t, name);
            if let Some(l) = log.lines().find(|l| l.contains(GOPROXY_MARKER)) {
                goproxy_logs += 1;
                goproxy_line.get_or_insert_with(|| short(&strip_ansi(l), 260));
            }
        }
    }
    let failing = |stage: &str| smallest(&|t| t.stage_pass(stage) == Some(false))
        .map(|t| tref(t, stage_reason(t, stage)));
    let missing_name = tasks.iter().filter_map(|t| {
        let req = t.fairness.get("api")?.as_object()?.values()
            .find_map(|v| v.as_array().filter(|a| !a.is_empty()))?;
        let r = &req[0];
        let sig = r.get("signature").and_then(Value::as_str).filter(|s| !s.is_empty())
            .or_else(|| r.get("symbol").and_then(Value::as_str))?;
        Some(tref(t, format!("required by the hidden tests: {sig}")))
    }).next().or_else(|| tasks.iter().filter(|t| t.valid).find_map(|t| {
        let log = strip_ansi(&read_log(dir, t, "parent.log"));
        let l = log.lines().find(|l| l.contains("undefined: "))?;
        Some(tref(t, short(l, 200)))
    }));
    let health_warnings = health.checks.iter().filter(|c| c.status == Status::Warn)
        .map(|c| format!("{}: {}", c.name, c.meaning)).collect();

    let ran = summary.fairness_ran;
    let mut pipeline = vec![
        Stage { name: "mine", status: if summary.candidates > 0 { "done" } else { "not started" },
                detail: format!("{} candidate commits", summary.candidates) },
        Stage { name: "validate", status: if summary.pending_retry == 0 { "done" } else { "partial" },
                detail: format!("{} valid questions, {} commits waiting for a retry", summary.valid, summary.pending_retry) },
        Stage { name: "fairness",
                status: if ran == 0 { "not started" } else if ran < summary.valid { "partial" } else { "done" },
                detail: format!("{ran} of {} valid tasks checked", summary.valid) },
        Stage { name: "split", status: "done",
                detail: format!("practice set {} tasks, final exam {} tasks", summary.evolve, summary.heldout) },
        Stage { name: "evolve with RRSI", status: "not started",
                detail: "runs are not visible to this report".into() },
        Stage { name: "final held-out exam", status: "not started", detail: "after evolution".into() },
    ];
    if let Some(s) = pipeline.iter_mut().find(|s| s.status != "done") {
        if s.status == "not started" {
            s.status = "next";
        }
    }
    Start {
        headline: headline(dir, tasks, summary, health),
        worked,
        already_passes,
        fails_on_commit,
        goproxy_line,
        goproxy_logs,
        pending_retry: summary.pending_retry,
        leak: failing("describe"),
        missing_name,
        flaky: failing("flake"),
        health_warnings,
        pipeline,
    }
}

/// One "things you can now try" item of the banner.
#[derive(Debug, Serialize)]
pub struct Try {
    pub lead: String,
    pub text: String,
    /// A task-browser filter the item links to, if any.
    pub filter: Option<String>,
    /// Not possible yet (shown marked as such).
    pub future: bool,
}

/// The banner at the very top: what the prototype has shown so far.
#[derive(Debug, Serialize)]
pub struct Headline {
    /// "exam" until RRSI run results exist, then "run".
    pub kind: &'static str,
    pub before: String,
    pub now: String,
    pub tries: Vec<Try>,
    pub not_shown: String,
    /// Key numbers, each with its unit and denominator.
    pub bullets: Vec<String>,
}

/// HOOK for RRSI run results. RRSI has not been run on a mined exam yet;
/// when it has (its frontier.json / held-out scores), parse them here and
/// return a run headline so the banner states the run result instead of
/// the exam-building result. Until then this always returns `None`.
pub fn rrsi_run_headline(_tasks_dir: &Path) -> Option<Headline> {
    None
}

/// How long the mining run took, from when the task files were written:
/// the first task's validation started `seconds` before its file was
/// written. `None` when it cannot be told or spans more than a day (then
/// the files come from several runs).
pub fn mining_duration(tasks: &[Task]) -> Option<f64> {
    let first = tasks.iter().filter_map(|t| Some((t.written_at?, t.seconds))).min_by_key(|x| x.0)?;
    let last = tasks.iter().filter_map(|t| t.written_at).max()?;
    let secs = (last - first.0) as f64 + first.1;
    (tasks.len() > 1 && secs > 0.0 && secs <= 86_400.0).then_some(secs)
}

/// "about 11 minutes", "about 2.5 hours".
pub fn duration_words(secs: f64) -> String {
    let m = (secs / 60.0).round();
    if m < 90.0 {
        format!("about {} minute{}", m.max(1.0), if m == 1.0 { "" } else { "s" })
    } else {
        format!("about {:.1} hours", secs / 3600.0)
    }
}

/// The exam-building headline, from the data.
pub fn headline(tasks_dir: &Path, tasks: &[Task], s: &Summary, h: &Health) -> Headline {
    if let Some(run) = rrsi_run_headline(tasks_dir) {
        return run;
    }
    let dropped = |needle: &str| s.rejections.iter().filter(|r| r.1 == "mine" && r.0.contains(needle))
        .map(|r| r.2).sum::<usize>();
    let already = dropped("already pass");
    let duration = mining_duration(tasks).map(|d| format!(" in {} on one machine", duration_words(d)))
        .unwrap_or_default();
    let before = "to learn whether a change to an AI coding agent actually helps on your own codebase, you \
                  either trusted public benchmarks built from other people's code, or wrote test tasks by hand \
                  and checked each one yourself.".to_string();
    let now = format!("one command turns your repository's git history into a verified exam \u{2014} {} recent \
                       commits{duration}, {} of them proven questions (the tests fail before the real fix and pass \
                       after it, offline) \u{2014} and it warns you when the final exam is not a fair test.",
                      s.candidates, s.valid);
    let instructions = if s.fairness_ran >= s.valid && s.valid > 0 {
        format!("written for all {}", s.valid)
    } else {
        format!("written for {} of {} so far", s.fairness_ran, s.valid)
    };
    let n_checks = h.checks.len();
    let tries = vec![
        Try { lead: "Hand an agent a real past bug from your own repo, as a proven-fair question.".into(),
              text: format!("Each of the {} tasks gives the repo at the commit before the fix and hidden tests that \
                             are known to fail without it and pass with it; the fairness stages add a written \
                             instruction that never shows the fix ({instructions}).", s.valid),
              filter: None, future: false },
        Try { lead: "Find tests that do not guard the change they shipped with.".into(),
              text: format!("{already} of {} commits added or changed tests that already passed before the change. \
                             For a pure refactor that is expected; for a bug fix it means the new test would not \
                             catch the bug coming back. The miner lists them.", s.candidates),
              filter: Some("tests already pass on parent".into()), future: false },
        Try { lead: "See which parts of your codebase an exam covers,".into(),
              text: "per subsystem, per code layout and per fix size, before trusting any score on it.".into(),
              filter: None, future: false },
        Try { lead: "Know in advance whether a final-exam result would mean anything.".into(),
              text: format!("The split-health checks flag a final exam that is harder, narrower or from a different \
                             part of the code than practice \u{2014} here, {} of {n_checks} checks warned.", h.warnings),
              filter: None, future: false },
        Try { lead: "Grow the exam automatically.".into(),
              text: "Re-running the miner only processes new commits, so every merged fix with a test becomes a \
                     new candidate question.".into(),
              filter: None, future: false },
        Try { lead: "Do it on any Go repository".into(),
              text: "(--repo PATH), with no API key: the search roles and the agent can run on a logged-in \
                     Copilot or Codex CLI, or a local model.".into(),
              filter: None, future: false },
        Try { lead: "One step away (grader and harness not wired yet):".into(),
              text: "compare two agent setups \u{2014} two models, two prompts, two tool sets \u{2014} by pass rate on \
                     your own code instead of on public benchmarks, and let RRSI evolve the harness against the \
                     practice set while the final exam checks for memorizing.".into(),
              filter: None, future: true },
    ];
    let mut bullets = vec![
        format!("{} candidate commits \u{2192} {} valid questions ({} of {}); dropped: {already} commits whose tests \
                 already passed before the fix, {} whose tests still failed with it{}",
                s.candidates, s.valid, pct_of(s.valid, s.candidates), s.candidates, dropped("fail on the commit"),
                if s.pending_retry > 0 { format!(", {} commits waiting for a retry", s.pending_retry) } else { String::new() }),
        format!("Exam split: practice set (evolve) {} tasks, final exam (held-out) the {} newest tasks \
                 (one task = one commit)", s.evolve, s.heldout),
    ];
    let warned: Vec<&str> = h.checks.iter().filter(|c| c.status == Status::Warn).map(|c| c.name.as_str()).collect();
    bullets.push(if warned.is_empty() {
        format!("Final-exam health: 0 of {n_checks} checks warn")
    } else {
        format!("Final-exam health: {} of {n_checks} checks warn ({})", warned.len(), warned.join("; "))
    });
    bullets.push(if s.fairness_ran == 0 {
        format!("Fairness stages: 0 of {} valid tasks checked (not run yet)", s.valid)
    } else {
        format!("Fairness stages: {} of {} valid tasks checked; {} passed every stage", s.fairness_ran, s.valid, s.exam_ready)
    });
    Headline {
        kind: "exam",
        before,
        now,
        tries,
        not_shown: "Not shown yet: whether RRSI then improves an agent on that exam.".into(),
        bullets,
    }
}

fn pct_of(k: usize, n: usize) -> String {
    if n == 0 { "0%".into() } else { format!("{:.0}%", 100.0 * k as f64 / n as f64) }
}
