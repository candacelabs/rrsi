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

//! Fairness stages: whether a mined task is a fair exam question.
//!
//! Each stage reads a task directory written by `mine` and records
//! `<task>/fairness/<stage>.json` = {"stage", "pass", "reason", ...details}.
//!
//! ```text
//! flake        re-run the commit tree's tests N times; every run must pass
//! api          the API the hidden tests need that the parent lacks (informational)
//! describe     write instruction.md with an LLM, then check it leaks nothing
//! probe        a second model judges instruction.md alone: sufficient, unambiguous?
//! specificity  do the tests pin strings, private names or call counts it omits?
//! gate         exam_ready = valid && flake && describe && probe && specificity
//! ```
//!
//! A verdict is reused (resume) while it is newer than every input it was
//! derived from, so re-running an earlier stage makes the later ones stale.
//! Infra failures and LLM errors write no verdict: they say nothing about the
//! task and are retried on the next run.

use crate::llm::Copilot;
use crate::scan::{self, Required};
use crate::{Docker, Outcome, TaskRecord};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

pub const STAGES: [&str; 6] = ["flake", "api", "describe", "probe", "specificity", "gate"];
/// The stages whose failure keeps a task out of the exam (api only informs).
pub const BLOCKING: [&str; 4] = ["flake", "describe", "probe", "specificity"];
pub const INSTRUCTION: &str = "instruction.md";

pub struct Task {
    pub dir: PathBuf,
    pub rec: TaskRecord,
}

impl Task {
    pub fn short(&self) -> &str {
        &self.rec.cand.sha[..12.min(self.rec.cand.sha.len())]
    }

    /// tests.patch restricted to the packages `go test` runs: test files of
    /// other packages are never graded, so no stage may rely on them.
    pub fn graded_tests(&self) -> Result<String> {
        Ok(scan::patch_in_dirs(&self.read("tests.patch")?, &self.package_dirs()))
    }

    pub fn read(&self, name: &str) -> Result<String> {
        std::fs::read_to_string(self.dir.join(name)).with_context(|| format!("{}/{name}", self.short()))
    }

    /// Repository-relative directories of the task's packages.
    pub fn package_dirs(&self) -> Vec<String> {
        let root = &self.rec.cand.module_root;
        self.rec.cand.packages.iter().map(|p| {
            let rel = p.trim_start_matches("./").trim_end_matches('/');
            match (root.is_empty(), rel.is_empty()) {
                (true, _) => rel.to_string(),
                (false, true) => root.clone(),
                (false, false) => format!("{root}/{rel}"),
            }
        }).collect()
    }
}

/// Every task directory (one holding task.json) under `tasks`, by name.
pub fn load_tasks(tasks: &Path, only: Option<&str>) -> Result<Vec<Task>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(tasks).with_context(|| format!("reading {}", tasks.display()))? {
        let dir = e?.path();
        let tj = dir.join("task.json");
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        if !tj.is_file() || only.is_some_and(|o| !name.starts_with(o)) {
            continue;
        }
        let rec: TaskRecord = serde_json::from_str(&std::fs::read_to_string(&tj)?)
            .with_context(|| format!("parsing {}", tj.display()))?;
        out.push(Task { dir, rec });
    }
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    Ok(out)
}

pub fn verdict_path(dir: &Path, stage: &str) -> PathBuf {
    dir.join("fairness").join(format!("{stage}.json"))
}

pub fn read_verdict(dir: &Path, stage: &str) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(verdict_path(dir, stage)).ok()?).ok()
}

pub fn write_verdict(dir: &Path, stage: &str, pass: bool, reason: &str, details: Value) -> Result<()> {
    let mut v = json!({"stage": stage, "pass": pass, "reason": reason});
    if let (Some(m), Value::Object(d)) = (v.as_object_mut(), details) {
        m.extend(d);
    }
    std::fs::create_dir_all(dir.join("fairness"))?;
    std::fs::write(verdict_path(dir, stage), serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

/// The files each stage's verdict is derived from (relative to the task).
pub fn inputs(stage: &str) -> Vec<String> {
    let v = |s: &str| format!("fairness/{s}.json");
    match stage {
        "describe" => vec![v("api")],
        "probe" => vec![v("describe"), INSTRUCTION.into()],
        "specificity" => vec![v("api"), v("describe"), INSTRUCTION.into()],
        "gate" => BLOCKING.iter().chain(["api"].iter()).map(|s| v(s)).collect(),
        _ => vec![],
    }.into_iter().chain(["task.json".to_string()]).collect()
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Whether `stage`'s verdict exists and is at least as new as every input
/// that exists.
pub fn is_fresh(dir: &Path, stage: &str) -> bool {
    let Some(t) = mtime(&verdict_path(dir, stage)) else { return false };
    inputs(stage).iter().all(|i| mtime(&dir.join(i)).is_none_or(|m| m <= t))
}

/// What one stage concluded for one task.
pub enum Step {
    /// A verdict about the task: written to fairness/<stage>.json.
    Verdict { pass: bool, reason: String, details: Value },
    /// No verdict (a prerequisite is missing, or infrastructure failed):
    /// nothing is written and the next run tries again.
    Retry(String),
}

pub fn verdict(pass: bool, reason: impl Into<String>, details: Value) -> Step {
    Step::Verdict { pass, reason: reason.into(), details }
}

/// Run `f` over the valid tasks (every task for `gate`) with `jobs`
/// workers, skipping fresh verdicts unless `force`. One line per task.
pub fn run_stage<F>(stage: &str, tasks: &[Task], jobs: usize, force: bool, f: F) -> Result<()>
where F: Fn(&Task) -> Result<Step> + Sync {
    if let Some(root) = tasks.first().and_then(|t| t.dir.parent()) {
        if let Some(tree) = crate::enclosing_work_tree(root) {
            bail!("refusing to write fairness verdicts under {} inside the git work tree {}",
                  root.display(), tree.display());
        }
    }
    let todo: Vec<&Task> = tasks.iter().filter(|t| stage == "gate" || t.rec.valid).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| while let Some(t) = todo.get(next.fetch_add(1, Ordering::SeqCst)) {
                if !force && is_fresh(&t.dir, stage) {
                    let v = read_verdict(&t.dir, stage).unwrap_or_default();
                    println!("[{stage}] {} {} {} (kept)", t.short(), pass_word(v["pass"].as_bool() == Some(true)),
                             v["reason"].as_str().unwrap_or(""));
                    continue;
                }
                match f(t).and_then(|step| match step {
                    Step::Verdict { pass, reason, details } => {
                        write_verdict(&t.dir, stage, pass, &reason, details)?;
                        Ok(format!("{} {reason}", pass_word(pass)))
                    }
                    Step::Retry(why) => Ok(format!("RETRY {why}")),
                }) {
                    Ok(line) => println!("[{stage}] {} {line}", t.short()),
                    Err(e) => println!("[{stage}] {} ERROR {}", t.short(), format!("{e:#}").replace('\n', " ")),
                }
            });
        }
    });
    Ok(())
}

fn pass_word(p: bool) -> &'static str {
    if p { "PASS" } else { "FAIL" }
}

// ---- flake ----

/// Re-run the commit tree's tests `runs` times; every run must pass.
pub fn flake(repo: &Path, t: &Task, docker: &Docker, runs: usize) -> Result<Step> {
    let work = tempfile::Builder::new().prefix("rrsi-flake-").tempdir()?;
    let tree = work.path().join("commit");
    crate::export_tree(repo, &t.rec.cand.sha, &tree)?;
    let (ok, log) = docker.download(&tree, &t.rec.cand.module_root)?;
    if !ok {
        let tail: String = log.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
        return Ok(Step::Retry(format!("infra: go mod download failed: {}", tail.trim())));
    }
    let mut outcomes = Vec::new();
    for i in 0..runs {
        let (out, log) = docker.go_test(&tree, &t.rec.cand.module_root, &t.rec.cand.packages)?;
        if out == Outcome::Infra {
            return Ok(Step::Retry(format!("infra on run {}", i + 1)));
        }
        if out != Outcome::Pass {
            std::fs::create_dir_all(t.dir.join("fairness"))?;
            std::fs::write(t.dir.join("fairness").join(format!("flake-{}.log", i + 1)), log)?;
        }
        outcomes.push(out);
    }
    let passed = outcomes.iter().filter(|o| **o == Outcome::Pass).count();
    let reason = if passed == runs { format!("{passed}/{runs} runs pass") }
                 else { format!("{passed}/{runs} runs pass: {:?}", outcomes) };
    Ok(verdict(passed == runs, reason, json!({"runs": runs, "outcomes": outcomes})))
}

// ---- api ----

pub fn required_of(t: &Task) -> Result<Vec<Required>> {
    let parent_log = t.read("parent.log").unwrap_or_default();
    // The module path (go.mod's `module`) is recovered from the log's
    // FAIL/ok line, so the stage needs no repository.
    let module = module_from_log(&parent_log, t)
        .or_else(|| module_from_log(&t.read("commit.log").unwrap_or_default(), t));
    Ok(scan::required_api(&parent_log, module.as_deref(), &t.read("src.patch")?,
                          &t.graded_tests()?, &t.package_dirs()))
}

/// The module path, read from a `FAIL\t<import path>` or `ok\t<import path>`
/// line of a log for one of the task's packages: the import path minus the
/// package's module-relative directory.
pub fn module_from_log(log: &str, t: &Task) -> Option<String> {
    for line in log.lines() {
        let Some(path) = line.strip_prefix("FAIL\t").or_else(|| line.strip_prefix("ok  \t")) else {
            continue;
        };
        let path = path.split(['\t', ' ']).next().unwrap_or("");
        for p in &t.rec.cand.packages {
            let rel = p.trim_start_matches("./").trim_end_matches('/');
            if rel.is_empty() {
                return Some(path.to_string());
            }
            if let Some(m) = path.strip_suffix(&format!("/{rel}")) {
                return Some(m.to_string());
            }
        }
    }
    None
}

pub fn api(t: &Task) -> Result<Step> {
    let req = required_of(t)?;
    let reason = format!("{} required symbols", req.len());
    Ok(verdict(true, reason, json!({"required": req})))
}

fn required_from_verdict(t: &Task) -> Option<Vec<Required>> {
    serde_json::from_value(read_verdict(&t.dir, "api")?.get("required")?.clone()).ok()
}

// ---- describe ----

pub const MAX_PROMPT_PATCH: usize = 80_000;

fn capped(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) { end -= 1; }
    format!("{}\n[... truncated ...]\n", &s[..end])
}

pub fn describe_prompt(t: &Task, tests_patch: &str, existing: &str, required: &[Required],
                       feedback: &str) -> String {
    let c = &t.rec.cand;
    let api: String = required.iter().map(|r| if r.signature.is_empty() {
        format!("- `{}` ({}; signature not known: describe what it must be from its use)\n", r.symbol, r.kind)
    } else {
        format!("- `{}` ({}): `{}`\n", r.symbol, r.kind, r.signature)
    }).collect();
    format!("\
You are writing the problem statement for a coding exercise in a Go repository.
A developer will receive ONLY your statement and the repository as it is now.
Write it as the GitHub issue a maintainer would file asking for this change.
You have no tools and need none: everything you may use is below.

Rules:
1. Describe the required BEHAVIOUR: what must happen, for which inputs and
   situations, including errors and edge cases the checks below rely on.
   Where exact values are observable (error messages, limits, orderings,
   counts, formats), state them exactly. When the checks pin how many times
   a collaborator's method is called, name that method and the count.
2. Require ONLY what the checks demonstrably exercise. Never infer a
   requirement from a check's name or description alone. For every situation
   the checks do not exercise, say that the current behaviour (see the
   existing code) must stay as it is: existing callers and checks of the
   package must keep passing. Leave internal structure (helper functions,
   unexported types or interfaces the checks never name) to the implementer
   and say so; do not ask for them.
   For every new error or early return, say what the other return values
   are and whether existing validation still runs first.
3. State EVERY symbol in the Required API list, each with its exact
   signature or declaration in inline code, and say what it must do.
4. NEVER include implementation code: no function bodies, no code blocks
   showing how to implement it, no step-by-step algorithm. Signatures,
   declarations and small usage examples are fine.
5. NEVER mention tests, hidden tests, test files, checks, grading, commits,
   patches or diffs. Do not say the change already exists anywhere.
6. Name the affected package directories. Output ONLY the markdown issue,
   starting with a `#` title line. No preamble, no remarks about yourself.

Package directories (module root `{root}`): {pkgs}

Commit subject (context only; do not quote it as a commit): {subject}
Commit body (context only):
{body}

Required API:
{api}
Existing code of the package(s) before the change (for context: what exists
and must keep working):
```go
{existing}
```

The checks that will judge the work (use them to learn the behaviour; do
not mention or quote them):
```diff
{tests}
```
{feedback}",
        root = c.module_root, pkgs = t.package_dirs().join(", "), subject = c.subject,
        body = if c.body.trim().is_empty() { "(none)" } else { c.body.trim() },
        api = if api.is_empty() { "(none)\n".into() } else { api },
        existing = if existing.trim().is_empty() { "(not available)" } else { existing },
        tests = capped(tests_patch, MAX_PROMPT_PATCH), feedback = feedback)
}

/// The issue in a writer's reply: a whole reply wrapped in one ```markdown
/// fence is unwrapped, and any preamble before the first `# ` title line is
/// dropped.
pub fn unfence(reply: &str) -> String {
    let mut t = reply.trim();
    if t.starts_with("```") && t.ends_with("```") && t.matches("```").count() == 2 {
        t = t[3..t.len() - 3].split_once('\n').map(|(_, b)| b).unwrap_or("").trim();
    }
    if !t.starts_with("# ") {
        if let Some(i) = t.find("\n# ") {
            t = t[i + 1..].trim();
        }
    }
    if t.ends_with("```") && t.matches("```").count() % 2 == 1 {
        t = t[..t.len() - 3].trim();
    }
    t.to_string()
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct DescribeCheck {
    pub missing_symbols: Vec<String>,
    pub forbidden: Vec<String>,
    pub leak: scan::LeakReport,
}

impl DescribeCheck {
    pub fn ok(&self) -> bool {
        self.missing_symbols.is_empty() && self.forbidden.is_empty() && !self.leak.leaks()
    }

    pub fn reason(&self) -> String {
        let mut r = Vec::new();
        if self.leak.leaks() {
            r.push(format!("leaks the fix ({} verbatim lines, {} shared 8-token runs)",
                           self.leak.verbatim.len(), self.leak.shingle_overlap));
        }
        if !self.missing_symbols.is_empty() {
            r.push(format!("omits {}", self.missing_symbols.join(", ")));
        }
        if !self.forbidden.is_empty() {
            r.push(format!("mentions {}", self.forbidden.join(", ")));
        }
        if r.is_empty() { "written, leak-free, names every required symbol".into() } else { r.join("; ") }
    }

    /// What to tell the writer on a retry. Never quotes the fix.
    pub fn feedback(&self) -> String {
        let mut f = String::from("\nYour previous draft was rejected:\n");
        if self.leak.leaks() {
            f += "- it contained implementation code. Describe behaviour only; no bodies or algorithms.\n";
        }
        for s in &self.missing_symbols {
            f += &format!("- it never named `{s}`; state it with its exact signature.\n");
        }
        for s in &self.forbidden {
            f += &format!("- it said \"{s}\"; never mention tests, commits or patches.\n");
        }
        f
    }
}

pub fn check_instruction(text: &str, src_patch: &str, required: &[Required]) -> DescribeCheck {
    let sigs: Vec<String> = required.iter().filter(|r| !r.signature.is_empty())
        .map(|r| r.signature.clone()).collect();
    DescribeCheck {
        missing_symbols: required.iter().filter(|r| !scan::has_word(text, &r.symbol))
            .map(|r| r.symbol.clone()).collect(),
        forbidden: scan::forbidden_mentions(text).into_iter()
            .chain((!text.starts_with("# ")).then(|| "text before the # title".to_string())).collect(),
        leak: scan::leak_check(text, src_patch, &sigs),
    }
}

pub const DESCRIBE_ATTEMPTS: usize = 2;

/// The parent's package source for the writer (so it knows what already
/// exists and what must keep working), or only its exported API when the
/// whole source is too large.
pub fn existing_code(repo: &Path, t: &Task) -> Result<String> {
    let files = parent_sources(repo, t)?;
    let whole: String = files.iter().map(|(p, s)| format!("// ==== {p}\n{s}\n")).collect();
    if whole.len() <= MAX_EXISTING_SOURCE {
        return Ok(whole);
    }
    parent_api(repo, t)
}

pub const MAX_EXISTING_SOURCE: usize = 40_000;

pub fn describe(repo: Option<&Path>, t: &Task, llm: &Copilot) -> Result<Step> {
    let Some(required) = required_from_verdict(t) else {
        return Ok(Step::Retry("needs the api stage".into()));
    };
    let (tests, src) = (t.graded_tests()?, t.read("src.patch")?);
    let existing = match repo {
        Some(r) => existing_code(r, t)?,
        None => String::new(),
    };
    let mut feedback = String::new();
    let mut last = None;
    for attempt in 1..=DESCRIBE_ATTEMPTS {
        let text = unfence(&llm.complete(&describe_prompt(t, &tests, &existing, &required, &feedback))?);
        let check = check_instruction(&text, &src, &required);
        std::fs::write(t.dir.join(INSTRUCTION), format!("{text}\n"))?;
        let ok = check.ok();
        feedback = check.feedback();
        last = Some((check, attempt, text.len()));
        if ok {
            break;
        }
    }
    let (check, attempts, chars) = last.context("no attempt")?;
    Ok(verdict(check.ok(), check.reason(), json!({
        "model": llm.model, "attempts": attempts, "chars": chars, "check": check})))
}

// ---- probe ----

pub const MAX_API_CONTEXT: usize = 60_000;

/// The exported declarations of the task's packages at the parent, one
/// block per file, capped at MAX_API_CONTEXT bytes.
pub fn parent_api(repo: &Path, t: &Task) -> Result<String> {
    let mut out = String::new();
    for (path, src) in parent_sources(repo, t)? {
        let decls = scan::exported_api(&src, &path);
        if decls.is_empty() {
            continue;
        }
        out += &format!("// {path}\n");
        for d in decls {
            out += &format!("{}{}\n", if d.kind == "field" { "    " } else { "" }, d.signature);
        }
        if out.len() > MAX_API_CONTEXT {
            return Ok(capped(&out, MAX_API_CONTEXT));
        }
    }
    Ok(out)
}

/// (path, text) of the non-test Go files of the task's packages at the parent.
pub fn parent_sources(repo: &Path, t: &Task) -> Result<Vec<(String, String)>> {
    let parent = &t.rec.cand.parent;
    let mut out = Vec::new();
    for dir in t.package_dirs() {
        let spec = if dir.is_empty() { parent.clone() } else { format!("{parent}:{dir}") };
        let Ok(list) = crate::git(repo, &["ls-tree", "--name-only", &spec]) else { continue };
        for name in list.lines().filter(|n| n.ends_with(".go") && !n.ends_with("_test.go")) {
            let path = if dir.is_empty() { name.to_string() } else { format!("{dir}/{name}") };
            out.push((path.clone(), crate::git(repo, &["show", &format!("{parent}:{path}")])?));
        }
    }
    Ok(out)
}

pub fn probe_prompt(instruction: &str, api: &str) -> String {
    format!("\
You are reviewing a problem statement before it is given to a developer. The
developer will see ONLY this statement and the repository (whose existing
exported API for the affected packages is listed below). Judge whether a
competent Go developer could implement exactly what is asked, with the exact
names and signatures other code will call, WITHOUT guessing.

Answer with ONE JSON object and nothing else:
{{\"sufficient\": true|false,
 \"ambiguities\": [\"each point where two reasonable readings would lead to different observable behaviour\"],
 \"guessed_names\": [\"each identifier, signature, error text or value the developer would have to invent because the statement does not fix it\"]}}
Use empty lists when there is nothing to report. Do not report stylistic
preferences or internal details that no caller can observe. Report only
questions an implementer would actually face in the situations the statement
describes; do not list hypothetical inputs or library corner cases the
statement never brings up (where it says other behaviour stays unchanged,
that is a complete answer). A name the statement explicitly leaves to the
implementer (internal helpers, unexported types) is not a guessed name.

=== Problem statement ===
{instruction}

=== Existing exported API of the affected packages ===
{api}")
}

pub fn probe(repo: &Path, t: &Task, llm: &Copilot) -> Result<Step> {
    let Some(desc) = read_verdict(&t.dir, "describe") else {
        return Ok(Step::Retry("needs the describe stage".into()));
    };
    if desc["pass"].as_bool() != Some(true) {
        return Ok(verdict(false, "no fair statement to probe (describe failed)", json!({})));
    }
    let instruction = t.read(INSTRUCTION)?;
    let api = parent_api(repo, t)?;
    let reply = llm.complete(&probe_prompt(&instruction, &api))?;
    let p = match scan::parse_probe(&reply) {
        Ok(p) => p,
        Err(e) => return Ok(Step::Retry(format!("unparseable reviewer reply: {e}"))),
    };
    let pass = p.sufficient && p.ambiguities.is_empty() && p.guessed_names.is_empty();
    let reason = if pass { "sufficient, unambiguous, no guessed names".to_string() } else {
        format!("sufficient={} ambiguities={} guessed_names={}", p.sufficient, p.ambiguities.len(),
                p.guessed_names.len())
    };
    Ok(verdict(pass, reason, json!({"model": llm.model, "review": p})))
}

// ---- specificity ----

pub fn specificity(repo: Option<&Path>, t: &Task) -> Result<Step> {
    let instruction = t.read(INSTRUCTION).unwrap_or_default();
    let existing: String = match repo {
        Some(r) => parent_sources(r, t)?.into_iter().map(|(_, s)| s).collect::<Vec<_>>().join("\n"),
        None => String::new(),
    };
    let findings = scan::specificity(&t.graded_tests()?, &t.read("src.patch")?, &instruction,
                                     &existing, &t.package_dirs());
    let open: Vec<&scan::Finding> = findings.iter().filter(|f| !f.covered).collect();
    let reason = if open.is_empty() {
        format!("{} findings, all covered by the statement", findings.len())
    } else {
        let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &open {
            *kinds.entry(f.kind.as_str()).or_default() += 1;
        }
        format!("uncovered: {}", kinds.iter().map(|(k, n)| format!("{n} {k}")).collect::<Vec<_>>().join(", "))
    };
    Ok(verdict(open.is_empty(), reason, json!({"findings": findings})))
}

// ---- gate ----

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Failing {
    pub stage: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate {
    pub sha12: String,
    pub exam_ready: bool,
    pub failing: Vec<Failing>,
}

/// A task is exam ready when it is valid and every blocking stage passed.
/// A stage that has not run counts as failing; api never blocks.
pub fn gate_of(t: &Task) -> Gate {
    let mut failing = Vec::new();
    if !t.rec.valid {
        failing.push(Failing { stage: "valid".into(), reason: t.rec.reason.clone() });
    } else {
        for s in BLOCKING {
            match read_verdict(&t.dir, s) {
                None => failing.push(Failing { stage: s.into(), reason: "not run".into() }),
                Some(v) if v["pass"].as_bool() != Some(true) => failing.push(Failing {
                    stage: s.into(), reason: v["reason"].as_str().unwrap_or("").to_string() }),
                _ => {}
            }
        }
    }
    Gate { sha12: t.short().to_string(), exam_ready: failing.is_empty(), failing }
}

pub fn gate(t: &Task) -> Result<Step> {
    let g = gate_of(t);
    let reason = if g.exam_ready { "exam ready".to_string() } else {
        g.failing.iter().map(|f| f.stage.as_str()).collect::<Vec<_>>().join(", ")
    };
    Ok(verdict(g.exam_ready, reason, serde_json::to_value(&g)?))
}

/// Write `<tasks>/exam.jsonl` (one line per task) and print how many tasks
/// each stage keeps out of the exam.
pub fn write_exam(tasks_dir: &Path, tasks: &[Task]) -> Result<Vec<Gate>> {
    let gates: Vec<Gate> = tasks.iter().map(gate_of).collect();
    let mut f = std::fs::File::create(tasks_dir.join("exam.jsonl"))?;
    for g in &gates {
        writeln!(f, "{}", serde_json::to_string(g)?)?;
    }
    let mut by: BTreeMap<&str, usize> = BTreeMap::new();
    for g in &gates {
        for s in &g.failing {
            *by.entry(s.stage.as_str()).or_default() += 1;
        }
    }
    println!("[gate] {:<12} {:>5}", "stage", "tasks");
    for s in ["valid"].iter().chain(BLOCKING.iter()) {
        println!("[gate] {:<12} {:>5} failing", s, by.get(s).copied().unwrap_or(0));
    }
    println!("[gate] {:<12} {:>5} / {}", "exam_ready", gates.iter().filter(|g| g.exam_ready).count(), gates.len());
    Ok(gates)
}

// ---- exam ----

/// One exam question: what an agent under test is given, plus what the
/// harness needs to check its answer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExamTask {
    pub sha: String,
    pub sha12: String,
    pub parent: String,
    pub module_root: String,
    pub packages: Vec<String>,
    pub instruction: String,
}

/// The exam-ready tasks under `tasks_dir`, judged afresh from the verdicts.
pub fn load_exam(tasks_dir: &Path) -> Result<Vec<ExamTask>> {
    let mut out = Vec::new();
    for t in load_tasks(tasks_dir, None)? {
        if !gate_of(&t).exam_ready {
            continue;
        }
        let c = &t.rec.cand;
        out.push(ExamTask { sha: c.sha.clone(), sha12: t.short().to_string(), parent: c.parent.clone(),
                            module_root: c.module_root.clone(), packages: c.packages.clone(),
                            instruction: t.read(INSTRUCTION)? });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Ran, Runner};
    use crate::Candidate;
    use std::sync::Mutex;
    use std::time::Duration;

    const SRC: &str = "diff --git a/m/pkg/box/box.go b/m/pkg/box/box.go\n@@ -1,3 +1,8 @@ package box\n \n\
        +// Limit is the most a box holds.\n+const Limit = 3\n+\n\
        +func (b *Box) Full() bool { return len(b.items) >= Limit && b.sealedAgainstFurtherItems }\n";
    const TESTS: &str = "diff --git a/m/pkg/box/box_test.go b/m/pkg/box/box_test.go\n@@ -1,1 +1,3 @@\n \n\
        +\tExpect(b.Full()).To(BeTrue())\n+\tExpect(Limit).To(Equal(3))\n";
    const LOG: &str = "$ go test ./pkg/box\nexit=1\n# example.invalid/m/pkg/box\n\
        pkg/box/box_test.go:2:9: b.Full undefined (type *Box has no field or method Full)\n\
        FAIL\texample.invalid/m/pkg/box [build failed]\n";

    fn task(root: &Path, name: &str, valid: bool) -> Task {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let rec = TaskRecord {
            cand: Candidate { sha: format!("{name}{}", "0".repeat(28)), parent: "p".repeat(40),
                              subject: "box: report when full".into(), body: String::new(),
                              module_root: "m".into(), packages: vec!["./pkg/box".into()],
                              src_files: vec!["m/pkg/box/box.go".into()],
                              test_files: vec!["m/pkg/box/box_test.go".into()], src_churn: 4, csf: None },
            valid, reason: if valid { "ok".into() } else { "tests already pass on parent".into() },
            fails_before: None, passes_after: None, parent_outcome: Some(Outcome::BuildFail),
            commit_outcome: Some(Outcome::Pass), detail: None, csf_guards: None, seconds: 1.0 };
        std::fs::write(dir.join("task.json"), serde_json::to_string(&rec).unwrap()).unwrap();
        for (f, s) in [("src.patch", SRC), ("tests.patch", TESTS), ("parent.log", LOG)] {
            std::fs::write(dir.join(f), s).unwrap();
        }
        Task { dir, rec }
    }

    struct Canned {
        replies: Mutex<Vec<String>>,
        prompts: Mutex<Vec<(Vec<String>, String)>>,
    }

    impl Runner for Canned {
        fn run(&self, argv: &[String], stdin: &str, _cwd: &Path, _t: Duration) -> Result<Ran> {
            self.prompts.lock().unwrap().push((argv.to_vec(), stdin.to_string()));
            let reply = self.replies.lock().unwrap().remove(0);
            Ok(Ran { code: 0, stdout: reply, stderr: String::new() })
        }
    }

    fn canned(replies: &[&str]) -> Canned {
        Canned { replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
                 prompts: Mutex::new(vec![]) }
    }

    fn llm(r: &Canned) -> Copilot<'_> {
        Copilot { runner: r, model: "writer".into(), reasoning: "low".into(), timeout: Duration::from_secs(1) }
    }

    #[test]
    fn the_module_path_comes_from_the_logs_fail_line() {
        let d = tempfile::tempdir().unwrap();
        let t = task(d.path(), "aaaaaaaaaaaa", true);
        assert_eq!(module_from_log(LOG, &t).as_deref(), Some("example.invalid/m"));
        assert_eq!(t.package_dirs(), ["m/pkg/box"]);
        let Step::Verdict { pass, details, .. } = api(&t).unwrap() else { panic!() };
        assert!(pass);
        let syms: Vec<&str> = details["required"].as_array().unwrap().iter()
            .map(|r| r["symbol"].as_str().unwrap()).collect();
        assert_eq!(syms, ["Full", "Limit"]);
    }

    #[test]
    fn describe_sends_no_fix_and_checks_the_statement() {
        let d = tempfile::tempdir().unwrap();
        let t = task(d.path(), "aaaaaaaaaaaa", true);
        write_verdict(&t.dir, "api", true, "", json!({"required": required_of(&t).unwrap()})).unwrap();
        // First draft leaks the body and omits Limit; the retry is clean.
        let leaky = "# Box\n\n`func (b *Box) Full() bool` returns len(b.items) >= Limit && b.sealedAgainstFurtherItems";
        let clean = "```markdown\n# Boxes should say when they are full\n\nIn `m/pkg/box`, add \
                     `func (b *Box) Full() bool` and `const Limit = 3`: Full reports whether the box holds \
                     Limit items and is sealed.\n```";
        let r = canned(&[leaky, clean]);
        let Step::Verdict { pass, reason, details } = describe(None, &t, &llm(&r)).unwrap() else { panic!() };
        assert!(pass, "{reason}");
        assert_eq!(details["attempts"], 2);
        let prompts = r.prompts.lock().unwrap();
        for (argv, stdin) in prompts.iter() {
            assert!(!stdin.contains("sealedAgainstFurtherItems"), "src.patch must never reach the writer");
            assert!(stdin.contains("Expect(b.Full())"), "the writer sees the tests");
            assert!(argv.iter().all(|a| !a.contains("Box")), "the prompt is on stdin");
        }
        assert!(prompts[1].1.contains("implementation code") && prompts[1].1.contains("`Limit`"),
                "the retry says what was wrong without quoting the fix");
        let text = std::fs::read_to_string(t.dir.join(INSTRUCTION)).unwrap();
        assert!(text.starts_with("# Boxes should say"), "fences are removed: {text}");
    }

    #[test]
    fn describe_waits_for_api_and_fails_a_statement_that_keeps_leaking() {
        let d = tempfile::tempdir().unwrap();
        let t = task(d.path(), "aaaaaaaaaaaa", true);
        let r = canned(&[]);
        assert!(matches!(describe(None, &t, &llm(&r)).unwrap(), Step::Retry(_)));
        write_verdict(&t.dir, "api", true, "", json!({"required": required_of(&t).unwrap()})).unwrap();
        let leaky = "Full and Limit: return len(b.items) >= Limit && b.sealedAgainstFurtherItems";
        let r = canned(&[leaky, leaky]);
        let Step::Verdict { pass, reason, .. } = describe(None, &t, &llm(&r)).unwrap() else { panic!() };
        assert!(!pass && reason.contains("leaks"), "{reason}");
    }

    #[test]
    fn a_reply_is_cut_to_the_issue() {
        assert_eq!(unfence("Not present locally; writing from the spec.\n\n# Title\n\nBody"), "# Title\n\nBody");
        assert_eq!(unfence("```markdown\n# Title\nBody\n```"), "# Title\nBody");
        assert_eq!(unfence("Here:\n```markdown\n# Title\nBody\n```"), "# Title\nBody");
        assert_eq!(unfence("# T\n```go\nfunc F()\n```"), "# T\n```go\nfunc F()\n```");
        let c = check_instruction("Sure.\nno title here", "", &[]);
        assert!(!c.ok() && c.forbidden == ["text before the # title"]);
    }

    #[test]
    fn every_blocking_stage_can_keep_a_task_out_and_api_never_does() {
        let d = tempfile::tempdir().unwrap();
        let t = task(d.path(), "aaaaaaaaaaaa", true);
        for s in BLOCKING {
            write_verdict(&t.dir, s, true, "fine", json!({})).unwrap();
        }
        write_verdict(&t.dir, "api", false, "extraction failed", json!({})).unwrap();
        assert!(gate_of(&t).exam_ready, "api is informational");
        for s in BLOCKING {
            write_verdict(&t.dir, s, false, "bad", json!({})).unwrap();
            let g = gate_of(&t);
            assert!(!g.exam_ready);
            assert_eq!(g.failing, [Failing { stage: s.into(), reason: "bad".into() }]);
            std::fs::remove_file(verdict_path(&t.dir, s)).unwrap();
            assert_eq!(gate_of(&t).failing[0].reason, "not run", "a missing stage blocks");
            write_verdict(&t.dir, s, true, "fine", json!({})).unwrap();
        }
        let invalid = task(d.path(), "bbbbbbbbbbbb", false);
        for s in BLOCKING {
            write_verdict(&invalid.dir, s, true, "fine", json!({})).unwrap();
        }
        assert_eq!(gate_of(&invalid).failing[0].stage, "valid");
    }

    #[test]
    fn exam_lists_only_ready_tasks_with_their_statement() {
        let d = tempfile::tempdir().unwrap();
        let ready = task(d.path(), "aaaaaaaaaaaa", true);
        task(d.path(), "bbbbbbbbbbbb", true);
        for s in BLOCKING {
            write_verdict(&ready.dir, s, true, "fine", json!({})).unwrap();
        }
        std::fs::write(ready.dir.join(INSTRUCTION), "# Do the thing\n").unwrap();
        let exam = load_exam(d.path()).unwrap();
        assert_eq!(exam.len(), 1);
        assert_eq!((exam[0].sha12.as_str(), exam[0].instruction.as_str()), ("aaaaaaaaaaaa", "# Do the thing\n"));
        let tasks = load_tasks(d.path(), None).unwrap();
        run_stage("gate", &tasks, 2, false, gate).unwrap();
        let gates = write_exam(d.path(), &tasks).unwrap();
        assert_eq!(gates.iter().filter(|g| g.exam_ready).count(), 1);
        let lines = std::fs::read_to_string(d.path().join("exam.jsonl")).unwrap();
        assert_eq!(lines.lines().count(), 2);
    }

    #[test]
    fn stages_skip_fresh_verdicts_and_redo_stale_or_forced_ones() {
        let d = tempfile::tempdir().unwrap();
        task(d.path(), "aaaaaaaaaaaa", true);
        task(d.path(), "bbbbbbbbbbbb", false);
        let tasks = load_tasks(d.path(), None).unwrap();
        let calls = AtomicUsize::new(0);
        let count = |_: &Task| { calls.fetch_add(1, Ordering::SeqCst); Ok(verdict(true, "ok", json!({}))) };
        run_stage("api", &tasks, 2, false, count).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "invalid tasks are not examined");
        run_stage("api", &tasks, 2, false, count).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "a fresh verdict is kept");
        run_stage("api", &tasks, 2, true, count).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "--force redoes it");
        // describe depends on api: a newer api verdict makes describe stale.
        let t = &tasks[0].dir;
        write_verdict(t, "describe", true, "ok", json!({})).unwrap();
        assert!(is_fresh(t, "describe"));
        std::thread::sleep(Duration::from_millis(20));
        write_verdict(t, "api", true, "ok", json!({})).unwrap();
        assert!(!is_fresh(t, "describe") && !is_fresh(t, "specificity"));
        // A retry writes nothing, so the next run tries again.
        run_stage("flake", &tasks, 1, false, |_| Ok(Step::Retry("infra".into()))).unwrap();
        assert!(!verdict_path(t, "flake").exists());
        assert_eq!(load_tasks(d.path(), Some("bbbb")).unwrap().len(), 1);
    }

    #[test]
    fn verdicts_are_refused_inside_a_work_tree() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join(".git")).unwrap();
        let tasks = vec![task(&d.path().join("tasks"), "aaaaaaaaaaaa", true)];
        assert!(run_stage("api", &tasks, 1, false, api).is_err());
        assert!(!verdict_path(&tasks[0].dir, "api").exists());
    }

    #[test]
    fn a_probe_needs_no_ambiguity_and_no_guessing() {
        let d = tempfile::tempdir().unwrap();
        let t = task(d.path(), "aaaaaaaaaaaa", true);
        std::fs::write(t.dir.join(INSTRUCTION), "# Full boxes").unwrap();
        write_verdict(&t.dir, "describe", true, "ok", json!({})).unwrap();
        let repo = d.path(); // no git: parent sources are simply absent
        for (reply, want) in [
            ("{\"sufficient\": true, \"ambiguities\": [], \"guessed_names\": []}", true),
            ("{\"sufficient\": true, \"ambiguities\": [\"order?\"], \"guessed_names\": []}", false),
            ("{\"sufficient\": true, \"ambiguities\": [], \"guessed_names\": [\"ErrFull\"]}", false),
            ("{\"sufficient\": false, \"ambiguities\": [], \"guessed_names\": []}", false),
        ] {
            let r = canned(&[reply]);
            let Step::Verdict { pass, .. } = probe(repo, &t, &llm(&r)).unwrap() else { panic!() };
            assert_eq!(pass, want, "{reply}");
            let prompt = &r.prompts.lock().unwrap()[0].1;
            assert!(!prompt.contains("Expect(") && !prompt.contains("sealedAgainst"), "no tests, no fix");
        }
    }
}
