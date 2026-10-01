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

//! Mine the "active agent without a PR" gap from Claude Code session
//! transcripts (kaashmonee/candace-server#294): agents that commit but do not
//! push, push but do not open a pull request, and the briefs and operator
//! corrections around them.
//!
//! ```text
//! rrsi-mine pr-gap --out DIR [--root ~/.claude/projects] [--since DATE] [--jobs 8]
//!                  [--exclude S] [--github]
//! ```
//!
//! One *run* is one transcript (a main session or one subagent). Per run,
//! from its tool calls (heredoc bodies and quoted messages are ignored, so a
//! commit message that says "git push" is not a push):
//!
//! - commits: successful `git commit` calls (not `--amend` / `--dry-run`);
//! - pushes: successful `git push` calls after the first commit, with the
//!   repository and branch from the push output and whether GitHub answered
//!   "Create a pull request for ..." (the branch had no PR at that moment);
//! - PRs: successful `gh pr create`, `gh api ... /pulls` POSTs and
//!   `*create_pull_request` tools.
//!
//! Each active run (at least one commit) gets one outcome: `pr_opened`,
//! `pr_existing` (GitHub has a PR for the pushed branch created by the end of
//! the run; needs `--github`), `pushed_default` (pushed to main/master, no PR
//! expected), `pushed_no_pr` or `never_pushed`. The last two are the gap.
//!
//! Signals ([`Signal`]), all deterministic: `never_pushed`, `slow_push`
//! (first push more than [`SLACK_SECS`] after the first commit),
//! `pushed_no_pr`, `slow_pr` (PR more than [`SLACK_SECS`] after the first
//! push), `brief_defers_pr` (an `Agent` prompt or a brief file read by a
//! subagent says "do not open a PR", "when complete, open", "don't push",
//! ...) and `operator_pr_correction` (a later human turn such as "agents need
//! to always have a PR").
//!
//! `--github` joins pushed branches against `gh pr list --head` (cached in
//! `DIR/github-cache.json`) to confirm gaps and find PRs opened by others.
//!
//! The summary also scores a family of trigger rules ("when an agent's first
//! commit is N minutes old with no push, or its pushed branch has had no PR
//! for N minutes, open a draft PR for it and tell the agent") on the measured
//! runs, and `DIR/pr-gap-task.json` is the top rule as a harness task.
//! Records are plain JSON in the episode shape of the `traces` miner; `DIR`
//! must be outside every git work tree.

use crate::miners::handoffs::{parse_session, Kind};
use crate::transcript::*;
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// A push or PR later than this after the step before it is "slow".
pub const SLACK_SECS: u64 = 10 * 60;
/// The trigger-rule thresholds scored, in minutes.
pub const RULE_MINUTES: [u64; 6] = [2, 5, 10, 15, 30, 60];
pub const TEXT_MAX: usize = 400;
pub const BRIEF_MAX: usize = 600;
pub const DEFAULT_BRANCHES: &[&str] = &["main", "master"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    NeverPushed,
    SlowPush,
    PushedNoPr,
    SlowPr,
    BriefDefersPr,
    OperatorPrCorrection,
}

impl Signal {
    pub const ALL: [Signal; 6] = [Signal::NeverPushed, Signal::SlowPush, Signal::PushedNoPr, Signal::SlowPr,
        Signal::BriefDefersPr, Signal::OperatorPrCorrection];

    pub fn name(self) -> String {
        serde_json::to_value(self).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
    }
}

// ------------------------------------------------------------ shell parsing

/// The code of a shell command: heredoc bodies dropped, quoted strings with
/// whitespace dropped (commit messages, PR bodies), one-word quoted strings
/// kept unquoted (`"repos/o/r/pulls"`).
pub fn shell_code(cmd: &str) -> String {
    static HEREDOC: OnceLock<Regex> = OnceLock::new();
    let heredoc = HEREDOC.get_or_init(|| Regex::new(r#"<<-?\s*['"]?([A-Za-z_][A-Za-z0-9_]*)['"]?"#).expect("static regex"));
    let mut lines = vec![];
    let mut until: Option<String> = None;
    for line in cmd.lines() {
        if let Some(d) = &until {
            if line.trim() == d {
                until = None;
            }
            continue;
        }
        if let Some(c) = heredoc.captures(line) {
            until = Some(c[1].to_string());
        }
        lines.push(line);
    }
    let src = lines.join("\n");
    let mut out = String::with_capacity(src.len());
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let q = cs[i];
        if q == '\'' || q == '"' {
            let mut j = i + 1;
            let mut s = String::new();
            while j < cs.len() && cs[j] != q {
                if q == '"' && cs[j] == '\\' && j + 1 < cs.len() {
                    j += 1;
                }
                s.push(cs[j]);
                j += 1;
            }
            if !s.chars().any(char::is_whitespace) {
                out.push_str(&s);
            }
            out.push(' ');
            i = j + 1;
        } else {
            out.push(q);
            i += 1;
        }
    }
    out
}

/// Shell segments (split on `&&`, `||`, `;`, `|`, newlines) of a command's code.
fn segments(code: &str) -> Vec<String> {
    code.split(['\n', ';', '&', '|']).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn git_op(seg: &str) -> Option<(&'static str, String)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(
        r"(?:^|[\s(`]|\$\()git(?:\s+(?:-C|-c|--git-dir|--work-tree)\s+\S+|\s+--no-pager)*\s+(commit|push)\b(.*)$")
        .expect("static regex"));
    let c = re.captures(seg)?;
    let op = if &c[1] == "commit" { "commit" } else { "push" };
    Some((op, c[2].to_string()))
}

fn has_flag(args: &str, flags: &[&str]) -> bool {
    args.split_whitespace().any(|w| flags.iter().any(|f| w == *f || w.starts_with(&format!("{f}="))))
}

/// What one shell command does to the gap: (new commits, pushes, PR creates).
pub fn shell_ops(cmd: &str) -> (usize, usize, usize) {
    static PR_CREATE: OnceLock<Regex> = OnceLock::new();
    static API_PULLS: OnceLock<Regex> = OnceLock::new();
    let pr_create = PR_CREATE.get_or_init(|| Regex::new(r"(?:^|[\s(`]|\$\()gh\s+pr\s+create\b").expect("static regex"));
    let api_pulls = API_PULLS.get_or_init(|| Regex::new(r"(?:^|[\s(`]|\$\()gh\s+api\b.*\brepos/[^/\s]+/[^/\s]+/pulls(?:\s|$)").expect("static regex"));
    let (mut commits, mut pushes, mut prs) = (0, 0, 0);
    for seg in segments(&shell_code(cmd)) {
        match git_op(&seg) {
            Some(("commit", args)) if !has_flag(&args, &["--amend", "--dry-run", "-h", "--help"]) => commits += 1,
            Some(("push", args)) if !has_flag(&args, &["--dry-run", "-n", "--delete", "-d", "-h", "--help"]) => pushes += 1,
            _ => {}
        }
        if pr_create.is_match(&seg) && !has_flag(&seg, &["--help", "-h", "--dry-run"]) {
            prs += 1;
        }
        let get_or_patch = has_flag(&seg, &["-X", "--method"]) && !seg.contains("POST");
        if api_pulls.is_match(&seg) && !get_or_patch
            && (seg.contains("POST") || has_flag(&seg, &["-f", "-F", "--field", "--raw-field", "--input"])) {
            prs += 1;
        }
    }
    (commits, pushes, prs)
}

/// One branch a push updated.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    /// `owner/repo` on GitHub; empty until resolved from `dir`.
    pub repo: String,
    pub branch: String,
    /// GitHub answered "Create a pull request for ...": no PR at that push.
    pub said_no_pr: bool,
    /// The checkout the push ran in, for resolving `repo` from its origin.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dir: String,
}

/// The targets a push output names (`To <repo>` and its ref-update lines).
pub fn push_targets(out: &str) -> Vec<Target> {
    static TO: OnceLock<Regex> = OnceLock::new();
    static REF: OnceLock<Regex> = OnceLock::new();
    let to = TO.get_or_init(|| Regex::new(r"(?m)^To\s+\S*github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?\s*$").expect("static regex"));
    let rf = REF.get_or_init(|| Regex::new(r"(?m)^\s*[*+ ]?\s*(?:\[new branch\]|[0-9a-f]{6,}\.\.\.?[0-9a-f]{6,})\s+\S+\s+->\s+(\S+)").expect("static regex"));
    let repo = to.captures(out).map(|c| format!("{}/{}", &c[1], &c[2])).unwrap_or_default();
    rf.captures_iter(out).map(|c| {
        let br = c[1].to_string();
        let said = out.contains(&format!("Create a pull request for '{br}'"));
        Target { repo: repo.clone(), branch: br, said_no_pr: said, dir: String::new() }
    }).collect()
}

/// The target of a push whose output was silenced (`-q`, `| tail`): the
/// branch from its refspec (or `fallback_branch` for none / `HEAD`), the
/// directory from `git -C` or the last `cd` before it (or `fallback_dir`).
pub fn push_guess(cmd: &str, fallback_dir: &str, fallback_branch: &str) -> Option<Target> {
    static CD: OnceLock<Regex> = OnceLock::new();
    static DASH_C: OnceLock<Regex> = OnceLock::new();
    let cd = CD.get_or_init(|| Regex::new(r"^cd\s+(\S+)").expect("static regex"));
    let dash_c = DASH_C.get_or_init(|| Regex::new(r"git\s+-C\s+(\S+)").expect("static regex"));
    let mut dir = fallback_dir.to_string();
    for seg in segments(&shell_code(cmd)) {
        if let Some(c) = cd.captures(&seg) {
            dir = c[1].to_string();
        }
        let Some(("push", args)) = git_op(&seg) else { continue };
        if let Some(c) = dash_c.captures(&seg) {
            dir = c[1].to_string();
        }
        let pos: Vec<&str> = args.split_whitespace().filter(|w| !w.starts_with('-')).collect();
        let refspec = pos.get(1).map(|r| r.rsplit(':').next().unwrap_or(r).trim_start_matches('+'))
            .filter(|r| !r.is_empty() && *r != "HEAD");
        let branch = refspec.map(|r| r.trim_start_matches("refs/heads/").to_string())
            .unwrap_or_else(|| fallback_branch.to_string());
        if branch.is_empty() || branch == "HEAD" {
            return None;
        }
        let home = std::env::var("HOME").unwrap_or_default();
        let dir = if dir.starts_with('~') { dir.replacen('~', &home, 1) } else { dir };
        return Some(Target { repo: String::new(), branch, said_no_pr: false, dir });
    }
    None
}

pub fn pr_urls(out: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+").expect("static regex"));
    re.find_iter(out).map(|m| m.as_str().to_string()).collect()
}

// --------------------------------------------------------- brief phrasing

/// Brief phrases that defer or forbid the PR (or the push).
pub const DEFER: &[&str] = &[
    r"do not open a (?:separate )?(?:draft )?(?:pr|pull request)", r"don'?t open a (?:separate )?(?:draft )?(?:pr|pull request)",
    r"do not create a (?:separate )?(?:pr|pull request)", r"don'?t create a (?:separate )?(?:pr|pull request)",
    r"no separate (?:pr|pull request)", r"do not open (?:a |any )?prs?\b", r"never open a (?:pr|pull request)",
    r"do not push\b", r"don'?t push\b", r"never push\b",
    r"when (?:the slice is |it is |it's |you are |you're |everything is )?(?:complete|done|finished)[^.\n]{0,40}\bopen\b",
    r"once (?:the slice is |it is |it's |you are |you're |everything is )?(?:complete|done|finished)[^.\n]{0,40}\bopen\b",
    r"open (?:the|a) (?:draft )?(?:pr|pull request) (?:when|once|after) (?:the slice |it |you |everything )?(?:is |are )?(?:complete|done|finished)",
    r"at the end,? open (?:the|a) (?:draft )?(?:pr|pull request)",
];
/// Pushes that are not about this branch ("don't push to main").
pub const DEFER_EXEMPT: &[&str] = &[" to main", " to master", " to origin/main", " to the default", " directly",
    " --force", " -f", " force", " secrets", " to production", " to prod"];
/// Brief phrases that ask for the PR early.
pub const EARLY: &[&str] = &[
    r"draft (?:pr|pull request)[^.\n]{0,80}\b(?:immediately|right after|right away|first commit|at the start|within minutes|early|first push)",
    r"(?:immediately|right after|right away|first commit|at the start|within minutes|first push)[^.\n]{0,80}\bdraft (?:pr|pull request)",
    r"always have a (?:draft )?(?:pr|pull request)", r"push after every commit",
];
/// Operator turns that point at a missing push or PR after work began.
pub const CORRECTION: &[&str] = &[
    r"always (?:have|has) a (?:draft )?(?:pr|pull request)", r"\bneeds? (?:a|their|its) (?:draft )?(?:pr|pull request)",
    r"where(?:'s| is| are) (?:the|your|their) (?:draft )?(?:pr|prs|pull request)", r"\bno (?:draft )?(?:pr|prs)\b",
    r"without (?:a|any) (?:draft )?(?:pr|prs|pull request)", r"(?:don'?t|doesn'?t|didn'?t|do not|does not) (?:even )?have (?:a|any) (?:pr|prs|pull request)",
    r"(?:haven'?t|hasn'?t|didn'?t|not) (?:even )?(?:been )?push(?:ed)?\b", r"\bunpushed\b", r"pushed (?:it )?without (?:opening )?(?:a|an?y) (?:pr|pull request)",
];

fn re_any(pats: &[&str]) -> Regex {
    Regex::new(&format!("(?i)(?:{})", pats.join("|"))).expect("static regex")
}

/// Brief phrases that defer the PR (exempting "don't push to main" etc.).
pub fn defer_phrases(text: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| re_any(DEFER));
    let lower = text.to_lowercase();
    re.find_iter(&lower).filter(|m| {
        let after: String = lower[m.end()..].chars().take(24).collect();
        !(m.as_str().contains("push") && DEFER_EXEMPT.iter().any(|x| after.starts_with(x)))
    }).map(|m| m.as_str().to_string()).collect()
}

pub fn asks_early_pr(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| re_any(EARLY)).is_match(text)
}

pub fn is_correction(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| re_any(CORRECTION)).is_match(text)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Brief {
    /// No brief (a main session).
    #[default]
    None,
    /// The brief defers or forbids the PR / push.
    Defers,
    /// The brief asks for a draft PR early.
    Early,
    /// A brief that says neither.
    Silent,
}

pub fn classify_brief(text: &str) -> Brief {
    if text.trim().is_empty() {
        Brief::None
    } else if !defer_phrases(text).is_empty() {
        Brief::Defers
    } else if asks_early_pr(text) {
        Brief::Early
    } else {
        Brief::Silent
    }
}

// ------------------------------------------------------------------ runs

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    #[default]
    NoCommits,
    PrOpened,
    PrExisting,
    PushedDefault,
    PushedNoPr,
    NeverPushed,
}

impl Outcome {
    pub fn is_gap(self) -> bool {
        matches!(self, Outcome::PushedNoPr | Outcome::NeverPushed)
    }
    pub fn name(self) -> String {
        serde_json::to_value(self).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
    }
}

/// One step of a run that matters to the gap.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub i: usize,
    pub ts: String,
    pub t: u64,
    /// "commit" | "push" | "pr"
    pub kind: String,
    pub n: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<Target>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub urls: Vec<String>,
    pub text: String,
}

/// What GitHub says about a pushed branch.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GithubPr {
    pub url: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub project: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub subagent: bool,
    pub file: String,
    pub start: String,
    pub end: String,
    pub events: usize,
    pub commits: usize,
    /// Commits made before the first push (all of them when never pushed).
    pub commits_before_first_push: usize,
    /// Commits after the last push, at the end of the transcript.
    pub unpushed_at_end: usize,
    pub pushes: usize,
    pub prs_created: usize,
    pub first_commit: Option<String>,
    pub first_push: Option<String>,
    pub first_pr: Option<String>,
    pub commit_to_push_secs: Option<u64>,
    pub push_to_pr_secs: Option<u64>,
    /// From the first commit to the last event.
    pub commit_to_end_secs: Option<u64>,
    /// From the first push to the last event.
    pub push_to_end_secs: Option<u64>,
    /// The branches pushed (merged by repo and branch).
    pub pushed: Vec<Target>,
    pub pr_urls: Vec<String>,
    pub brief: Brief,
    pub brief_phrases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub github: Vec<GithubPr>,
    pub outcome: Outcome,
    pub steps: Vec<Step>,
}

/// An orchestrator-side moment: a brief that defers the PR, or an operator
/// correction.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Moment {
    pub i: usize,
    pub ts: String,
    pub signal: Option<Signal>,
    pub text: String,
    pub phrases: Vec<String>,
}

/// The first user message of a transcript when it is plain text (a
/// subagent's brief).
pub fn first_prompt(bytes: &[u8]) -> String {
    for line in bytes.split(|b| *b == b'\n') {
        let Ok(e) = serde_json::from_slice::<Value>(line) else { continue };
        if e.get("type").and_then(Value::as_str) != Some("user") {
            continue;
        }
        return match e.get("message").and_then(|m| m.get("content")) {
            Some(Value::String(s)) => s.clone(),
            Some(c @ Value::Array(_)) if !c.as_array().is_some_and(|a| a.iter().any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))) =>
                content_text(c),
            _ => String::new(),
        };
    }
    String::new()
}

fn is_brief_read(k: &Kind) -> bool {
    match k {
        Kind::ToolUse { name, input, .. } if name == "Bash" =>
            input.get("command").and_then(Value::as_str).is_some_and(|c| c.contains("brief") && (c.contains("cat ") || c.contains("sed ") || c.contains("head "))),
        Kind::ToolUse { name, input, .. } if name == "Read" =>
            input.get("file_path").and_then(Value::as_str).is_some_and(|p| p.to_lowercase().contains("brief")),
        _ => false,
    }
}

/// Measures one transcript.
pub fn measure(bytes: &[u8], project: &str, file: &str) -> (Run, Vec<Moment>) {
    let ses = parse_session(BufReader::new(bytes));
    let evs = &ses.events;
    let subagent = file.contains("/subagents/");
    let mut brief = if subagent { first_prompt(bytes) } else { String::new() };
    let mut steps: Vec<Step> = vec![];
    let mut moments = vec![];
    let mut work_began = false;
    let result_of = |id: &str| evs.iter().find_map(|e| match &e.kind {
        Kind::ToolResult { id: x, is_error, text } if x == id => Some((*is_error, text.clone())),
        _ => None,
    });
    for (i, e) in evs.iter().enumerate() {
        let t = iso_secs(&e.ts).unwrap_or(0);
        match &e.kind {
            Kind::ToolUse { id, name, input } => {
                if subagent && is_brief_read(&e.kind) {
                    if let Some((false, text)) = result_of(id) {
                        brief.push('\n');
                        brief.push_str(&text);
                    }
                }
                if name == "Agent" || name == "Task" {
                    work_began = true;
                    let prompt = field(input, "prompt");
                    let phrases = defer_phrases(&prompt);
                    if !phrases.is_empty() {
                        moments.push(Moment { i, ts: e.ts.clone(), signal: Some(Signal::BriefDefersPr),
                                              text: truncate(&prompt, BRIEF_MAX), phrases });
                    }
                    continue;
                }
                let (commits, pushes, prs) = if name == "Bash" {
                    shell_ops(input.get("command").and_then(Value::as_str).unwrap_or_default())
                } else if name.ends_with("create_pull_request") {
                    (0, 0, 1)
                } else {
                    (0, 0, 0)
                };
                if commits + pushes + prs == 0 {
                    continue;
                }
                let (err, out) = result_of(id).unwrap_or((true, String::new()));
                let text = truncate(&format!("{} => {}", input.get("command").and_then(Value::as_str).unwrap_or(name),
                                             out.trim()), TEXT_MAX);
                let mut targets = push_targets(&out);
                let cmd = input.get("command").and_then(Value::as_str).unwrap_or_default();
                if targets.is_empty() && pushes > 0 && !err {
                    targets.extend(push_guess(cmd, &e.cwd, &e.branch));
                }
                // An errored chained call: count only what its output proves.
                let commits = if err { 0 } else { commits };
                let pushes = if err && push_targets(&out).is_empty() { 0 } else { pushes };
                let urls = pr_urls(&out);
                let prs = if err { 0 } else { prs };
                for (kind, n) in [("commit", commits), ("push", pushes), ("pr", prs)] {
                    if n > 0 {
                        work_began = true;
                        steps.push(Step { i, ts: e.ts.clone(), t, kind: kind.into(), n,
                                          targets: if kind == "push" { targets.clone() } else { vec![] },
                                          urls: if kind == "pr" { urls.clone() } else { vec![] }, text: text.clone() });
                    }
                }
            }
            Kind::Human { text } if work_began && is_correction(text) => {
                moments.push(Moment { i, ts: e.ts.clone(), signal: Some(Signal::OperatorPrCorrection),
                                      text: truncate(text, TEXT_MAX), phrases: vec![] });
            }
            _ => {}
        }
    }
    let start = evs.first().map(|e| e.ts.clone()).unwrap_or_default();
    let end = evs.last().map(|e| e.ts.clone()).unwrap_or_default();
    let mut run = Run {
        id: format!("{:016x}", fnv64(format!("pr-gap#{file}").as_bytes())),
        project: project.into(), session_id: ses.session_id.clone(), agent_id: ses.agent_id.clone(), subagent,
        file: file.into(), start, end: end.clone(), events: evs.len(), brief: classify_brief(&brief),
        brief_phrases: defer_phrases(&brief), ..Run::default()
    };
    fill(&mut run, steps, iso_secs(&end).unwrap_or(0));
    (run, moments)
}

/// Derives the counts and the outcome of a run from its steps.
pub fn fill(run: &mut Run, steps: Vec<Step>, end_t: u64) {
    let first_commit = steps.iter().position(|s| s.kind == "commit");
    let after = |k: &str| first_commit.and_then(|c| steps[c..].iter().position(|s| s.kind == k).map(|p| p + c));
    let (fp, fr) = (after("push"), after("pr"));
    run.commits = steps.iter().filter(|s| s.kind == "commit").map(|s| s.n).sum();
    run.pushes = fp.map(|p| steps[p..].iter().filter(|s| s.kind == "push").count()).unwrap_or(0);
    run.prs_created = fr.map(|p| steps[p..].iter().filter(|s| s.kind == "pr").count()).unwrap_or(0);
    run.commits_before_first_push = steps[..fp.unwrap_or(steps.len())].iter().filter(|s| s.kind == "commit").map(|s| s.n).sum();
    let last_push = steps.iter().rposition(|s| s.kind == "push");
    run.unpushed_at_end = steps[last_push.map(|p| p + 1).unwrap_or(0)..].iter().filter(|s| s.kind == "commit").map(|s| s.n).sum();
    let at = |i: Option<usize>| i.map(|i| steps[i].t);
    run.first_commit = first_commit.map(|i| steps[i].ts.clone());
    run.first_push = fp.map(|i| steps[i].ts.clone());
    run.first_pr = fr.map(|i| steps[i].ts.clone());
    run.commit_to_push_secs = at(fp).zip(at(first_commit)).map(|(p, c)| p.saturating_sub(c));
    run.push_to_pr_secs = at(fr).zip(at(fp)).map(|(r, p)| r.saturating_sub(p));
    run.commit_to_end_secs = at(first_commit).map(|c| end_t.saturating_sub(c));
    run.push_to_end_secs = at(fp).map(|p| end_t.saturating_sub(p));
    let mut pushed: Vec<Target> = vec![];
    for s in steps.iter().filter(|s| s.kind == "push") {
        for t in &s.targets {
            match pushed.iter_mut().find(|p| p.branch == t.branch && (p.repo == t.repo || p.repo.is_empty() || t.repo.is_empty())) {
                Some(p) => {
                    p.said_no_pr |= t.said_no_pr;
                    if p.repo.is_empty() {
                        p.repo = t.repo.clone();
                    }
                    if p.dir.is_empty() {
                        p.dir = t.dir.clone();
                    }
                }
                None => pushed.push(t.clone()),
            }
        }
    }
    // A silenced push in a run that named its repo elsewhere is that repo.
    let named: Vec<String> = pushed.iter().map(|p| p.repo.clone()).filter(|r| !r.is_empty()).collect();
    if named.len() == 1 || named.windows(2).all(|w| w[0] == w[1]) {
        if let Some(r) = named.first() {
            for p in pushed.iter_mut().filter(|p| p.repo.is_empty()) {
                p.repo = r.clone();
            }
        }
    }
    run.pushed = pushed;
    run.pr_urls = steps.iter().flat_map(|s| s.urls.clone()).collect::<Vec<_>>();
    run.pr_urls.dedup();
    run.steps = steps;
    run.outcome = outcome(run, end_t);
}

pub fn outcome(run: &Run, end_t: u64) -> Outcome {
    if run.commits == 0 {
        Outcome::NoCommits
    } else if run.prs_created > 0 {
        Outcome::PrOpened
    } else if run.github.iter().any(|g| iso_secs(&g.created_at).is_some_and(|c| c <= end_t)) {
        Outcome::PrExisting
    } else if run.pushes > 0 && !run.pushed.is_empty() && run.pushed.iter().all(|p| DEFAULT_BRANCHES.contains(&p.branch.as_str())) {
        Outcome::PushedDefault
    } else if run.pushes > 0 {
        Outcome::PushedNoPr
    } else {
        Outcome::NeverPushed
    }
}

/// The signals one run fires.
pub fn run_signals(run: &Run) -> Vec<Signal> {
    let mut s = vec![];
    match run.outcome {
        Outcome::NeverPushed => s.push(Signal::NeverPushed),
        Outcome::PushedNoPr => s.push(Signal::PushedNoPr),
        _ => {}
    }
    if run.commit_to_push_secs.is_some_and(|d| d > SLACK_SECS) {
        s.push(Signal::SlowPush);
    }
    if run.push_to_pr_secs.is_some_and(|d| d > SLACK_SECS) {
        s.push(Signal::SlowPr);
    }
    if run.subagent && run.commits > 0 && run.brief == Brief::Defers {
        s.push(Signal::BriefDefersPr);
    }
    s
}

// ------------------------------------------------------------- the rules

/// One scored trigger rule.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuleScore {
    pub minutes: u64,
    pub rule: String,
    /// Active runs it fires on.
    pub fires: usize,
    /// Fires on runs that ended in the gap (pushed_no_pr / never_pushed).
    pub gaps_caught: usize,
    /// Fires on runs that got their PR anyway, later.
    pub nags: usize,
    /// Gap runs it never fires on (they ended within N minutes).
    pub gaps_missed: usize,
    /// gaps_caught - nags.
    pub score: i64,
}

pub fn rule_text(n: u64) -> String {
    format!("when an agent's first commit is {n} minutes old with no push, or its pushed branch has had no PR for \
             {n} minutes, open a draft PR for it and tell the agent")
}

/// Whether the rule with threshold `secs` fires during a run.
pub fn fires(run: &Run, secs: u64) -> bool {
    if run.commits == 0 || matches!(run.outcome, Outcome::PushedDefault) {
        return false;
    }
    let no_push = match run.commit_to_push_secs {
        Some(d) => d > secs,
        None => run.commit_to_end_secs.unwrap_or(0) >= secs,
    };
    let no_pr = match (run.push_to_end_secs, run.push_to_pr_secs, run.outcome) {
        (_, _, Outcome::PrExisting) => false,
        (_, Some(d), _) => d > secs,
        (Some(e), None, _) => e >= secs,
        _ => false,
    };
    no_push || no_pr
}

pub fn score_rules(runs: &[&Run]) -> Vec<RuleScore> {
    RULE_MINUTES.iter().map(|&m| {
        let mut r = RuleScore { minutes: m, rule: rule_text(m), ..RuleScore::default() };
        for run in runs {
            let f = fires(run, m * 60);
            r.fires += usize::from(f);
            match (f, run.outcome.is_gap()) {
                (true, true) => r.gaps_caught += 1,
                (true, false) => r.nags += 1,
                (false, true) => r.gaps_missed += 1,
                _ => {}
            }
        }
        r.score = r.gaps_caught as i64 - r.nags as i64;
        r
    }).collect()
}

/// The best rule: highest score, then most gaps caught, then the larger N.
pub fn top_rule(scores: &[RuleScore]) -> Option<&RuleScore> {
    scores.iter().max_by(|a, b| (a.score, a.gaps_caught, a.minutes).cmp(&(b.score, b.gaps_caught, b.minutes)))
}

// ------------------------------------------------------------ GitHub join

/// Looks up the PRs whose head is a branch.
pub trait Github: Sync {
    fn prs_for_head(&self, repo: &str, branch: &str) -> Result<Vec<GithubPr>>;
    /// `owner/repo` of a checkout's GitHub `origin`, if it still exists.
    fn origin_repo(&self, dir: &str) -> Option<String>;
}

/// `owner/repo` from a GitHub remote URL.
pub fn github_repo(url: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$").expect("static regex"));
    re.captures(url.trim()).map(|c| format!("{}/{}", &c[1], &c[2]))
}

/// `gh pr list --head` with the operator's own `gh` login.
pub struct GhCli;

impl Github for GhCli {
    fn prs_for_head(&self, repo: &str, branch: &str) -> Result<Vec<GithubPr>> {
        let out = std::process::Command::new("gh")
            .args(["pr", "list", "--repo", repo, "--head", branch, "--state", "all", "--json", "url,createdAt"])
            .output().context("running gh")?;
        anyhow::ensure!(out.status.success(), "gh pr list {repo} {branch}: {}", String::from_utf8_lossy(&out.stderr).trim());
        let v: Vec<Value> = serde_json::from_slice(&out.stdout)?;
        Ok(v.iter().map(|p| GithubPr { url: field(p, "url"), created_at: field(p, "createdAt") }).collect())
    }
    fn origin_repo(&self, dir: &str) -> Option<String> {
        if dir.is_empty() || !Path::new(dir).is_dir() {
            return None;
        }
        let out = std::process::Command::new("git").args(["-C", dir, "remote", "get-url", "origin"]).output().ok()?;
        out.status.success().then(|| github_repo(&String::from_utf8_lossy(&out.stdout))).flatten()
    }
}

/// Joins pushed branches against GitHub (cached in `cache`; a branch with no
/// PR is asked again next time). Returns (lookups, failures).
pub fn join_github(runs: &mut [Run], gh: &dyn Github, cache: &mut BTreeMap<String, Vec<GithubPr>>) -> (usize, usize) {
    let (mut asked, mut failed) = (0, 0);
    let mut origins: BTreeMap<String, Option<String>> = BTreeMap::new();
    for run in runs.iter_mut().filter(|r| r.commits > 0 && r.prs_created == 0) {
        for p in run.pushed.iter_mut().filter(|p| p.repo.is_empty()) {
            p.repo = origins.entry(p.dir.clone()).or_insert_with(|| gh.origin_repo(&p.dir)).clone().unwrap_or_default();
        }
        let mut found = vec![];
        for (repo, br) in run.pushed.iter().filter(|p| !p.repo.is_empty() && !DEFAULT_BRANCHES.contains(&p.branch.as_str()))
            .map(|p| (&p.repo, &p.branch)) {
            let key = format!("{repo}#{br}");
            if cache.get(&key).is_none_or(Vec::is_empty) {
                asked += 1;
                match gh.prs_for_head(repo, br) {
                    Ok(v) => { cache.insert(key.clone(), v); }
                    Err(e) => { failed += 1; eprintln!("[pr-gap] github: {e:#}"); }
                }
            }
            found.extend(cache.get(&key).cloned().unwrap_or_default());
        }
        run.github = found;
        let end_t = iso_secs(&run.end).unwrap_or(0);
        run.outcome = outcome(run, end_t);
    }
    (asked, failed)
}

/// How GitHub saw a `pushed_no_pr` run: `confirmed_no_pr` (no PR for any
/// pushed branch, even now), `pr_opened_after_run` (someone opened one after
/// the run ended) or `unresolved` (no pushed branch could be looked up).
pub fn github_status(run: &Run, cache: &BTreeMap<String, Vec<GithubPr>>) -> &'static str {
    let keys: Vec<String> = run.pushed.iter().filter(|p| !p.repo.is_empty() && !DEFAULT_BRANCHES.contains(&p.branch.as_str()))
        .map(|p| format!("{}#{}", p.repo, p.branch)).filter(|k| cache.contains_key(k)).collect();
    if keys.is_empty() {
        "unresolved"
    } else if keys.iter().any(|k| !cache[k].is_empty()) {
        "pr_opened_after_run"
    } else {
        "confirmed_no_pr"
    }
}

// --------------------------------------------------------------- episodes

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CtxEv {
    pub i: usize,
    pub ts: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub signals: Vec<Signal>,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub span_events: usize,
    pub tool_calls: usize,
    pub tool_errors: usize,
    pub human_turns: usize,
    pub session_events: usize,
}

/// Same shape as a `traces` / `handoffs` episode, plus `extra`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub project: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub subagent: bool,
    pub file: String,
    pub start: String,
    pub end: String,
    pub start_event: usize,
    pub end_event: usize,
    pub signals: BTreeMap<Signal, usize>,
    pub user_turn: String,
    pub context: Vec<CtxEv>,
    pub counts: Counts,
    pub extra: Value,
}

fn mins(s: Option<u64>) -> String {
    s.map(|s| format!("{:.1} min", s as f64 / 60.0)).unwrap_or_else(|| "never".into())
}

/// One episode per run that fires a signal, one per orchestrator moment.
pub fn episodes(run: &Run, moments: &[Moment], brief_text: &str) -> Vec<Episode> {
    let mut out = vec![];
    let sigs = run_signals(run);
    if !sigs.is_empty() {
        let mut ctx: Vec<CtxEv> = run.steps.iter().take(12).map(|s| CtxEv {
            i: s.i, ts: s.ts.clone(), kind: "tool_use".into(), tool: Some(s.kind.clone()), signals: vec![], text: s.text.clone(),
        }).collect();
        ctx.push(CtxEv { i: run.events.saturating_sub(1), ts: run.end.clone(), kind: "run_end".into(), tool: None,
            signals: sigs.clone(),
            text: format!("outcome {}: {} commits ({} before the first push, {} unpushed at the end), {} pushes, {} PRs \
                           created; first commit -> push {}, push -> PR {}; brief {:?}",
                          run.outcome.name(), run.commits, run.commits_before_first_push, run.unpushed_at_end,
                          run.pushes, run.prs_created, mins(run.commit_to_push_secs), mins(run.push_to_pr_secs), run.brief) });
        let a = run.steps.first().map(|s| s.i).unwrap_or(0);
        out.push(Episode {
            id: run.id.clone(), project: run.project.clone(), session_id: run.session_id.clone(),
            agent_id: run.agent_id.clone(), subagent: run.subagent, file: run.file.clone(),
            start: run.first_commit.clone().unwrap_or_else(|| run.start.clone()), end: run.end.clone(),
            start_event: a, end_event: run.events.saturating_sub(1),
            signals: sigs.iter().map(|s| (*s, 1)).collect(),
            user_turn: truncate(brief_text, 800), context: ctx,
            counts: Counts { span_events: run.events.saturating_sub(a), tool_calls: run.steps.len(), session_events: run.events, ..Counts::default() },
            extra: json!({"outcome": run.outcome, "commits": run.commits, "commits_before_first_push": run.commits_before_first_push,
                          "unpushed_at_end": run.unpushed_at_end, "commit_to_push_secs": run.commit_to_push_secs,
                          "push_to_pr_secs": run.push_to_pr_secs, "brief": run.brief, "brief_phrases": run.brief_phrases,
                          "pushed": run.pushed, "github": run.github}),
        });
    }
    for m in moments {
        let Some(sig) = m.signal else { continue };
        out.push(Episode {
            id: format!("{:016x}", fnv64(format!("pr-gap#{}#{}", run.file, m.i).as_bytes())),
            project: run.project.clone(), session_id: run.session_id.clone(), agent_id: run.agent_id.clone(),
            subagent: run.subagent, file: run.file.clone(), start: m.ts.clone(), end: m.ts.clone(),
            start_event: m.i, end_event: m.i, signals: [(sig, 1)].into(),
            user_turn: if sig == Signal::OperatorPrCorrection { m.text.clone() } else { String::new() },
            context: vec![CtxEv { i: m.i, ts: m.ts.clone(),
                kind: if sig == Signal::BriefDefersPr { "tool_use".into() } else { "human".into() },
                tool: (sig == Signal::BriefDefersPr).then(|| "Agent".to_string()), signals: vec![sig], text: m.text.clone() }],
            counts: Counts { span_events: 1, session_events: run.events,
                             human_turns: usize::from(sig == Signal::OperatorPrCorrection), ..Counts::default() },
            extra: json!({"phrases": m.phrases}),
        });
    }
    out
}

// ------------------------------------------------------------------ driver

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileState {
    /// [`FORMAT`] of the cached record; another value re-parses the transcript.
    #[serde(default)]
    pub format: u32,
    pub mtime: u64,
    pub size: u64,
    pub hash: String,
    pub events: usize,
}

/// Bump when a cached per-transcript record changes shape or meaning.
pub const FORMAT: u32 = 2;

#[derive(Debug, Default, Serialize, Deserialize)]
struct FileOut {
    run: Run,
    moments: Vec<Moment>,
    brief: String,
}

/// Distribution of a quantity over runs (nearest-rank quantiles), divided by `unit`.
#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Dist {
    pub n: usize,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}

pub fn dist(mut xs: Vec<u64>, unit: f64) -> Dist {
    if xs.is_empty() {
        return Dist::default();
    }
    xs.sort_unstable();
    let q = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize] as f64 / unit;
    Dist { n: xs.len(), median: q(0.5), p90: q(0.9), max: xs[xs.len() - 1] as f64 / unit }
}

/// Counts for one population of runs (all, main sessions, subagents).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Population {
    pub runs: usize,
    pub active_runs: usize,
    pub outcomes: BTreeMap<String, usize>,
    pub gap_runs: usize,
    /// Pushed runs where GitHub said the branch had no PR at the push.
    pub github_said_no_pr_at_push: usize,
    /// Minutes from the first commit to the first push (pushed runs).
    pub commit_to_push_min: Dist,
    /// Minutes from the first push to the first PR created (runs that made one).
    pub push_to_pr_min: Dist,
    pub commits_before_first_push: Dist,
    pub runs_with_unpushed_commits_at_end: usize,
}

fn population<'a>(runs: impl Iterator<Item = &'a Run>) -> Population {
    let mut p = Population::default();
    let (mut c2p, mut p2r, mut before) = (vec![], vec![], vec![]);
    for r in runs {
        p.runs += 1;
        if r.commits == 0 {
            continue;
        }
        p.active_runs += 1;
        *p.outcomes.entry(r.outcome.name()).or_insert(0) += 1;
        p.gap_runs += usize::from(r.outcome.is_gap());
        p.github_said_no_pr_at_push += usize::from(r.pushed.iter().any(|x| x.said_no_pr));
        c2p.extend(r.commit_to_push_secs);
        p2r.extend(r.push_to_pr_secs);
        before.push(r.commits_before_first_push as u64);
        p.runs_with_unpushed_commits_at_end += usize::from(r.unpushed_at_end > 0);
    }
    p.commit_to_push_min = dist(c2p, 60.0);
    p.push_to_pr_min = dist(p2r, 60.0);
    p.commits_before_first_push = dist(before, 1.0);
    p
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub transcripts: usize,
    pub processed: usize,
    pub skipped_unchanged: usize,
    pub sessions: usize,
    pub projects: usize,
    pub events: usize,
    pub episodes: usize,
    pub episodes_per_signal: BTreeMap<String, usize>,
    pub hits_per_signal: BTreeMap<String, usize>,
    pub sessions_with_episodes: usize,
    pub all: Population,
    pub main_sessions: Population,
    pub subagents: Population,
    /// Active subagent runs by brief class: (runs, gap runs).
    pub subagent_gap_by_brief: BTreeMap<String, (usize, usize)>,
    pub github: Value,
    pub rules: Vec<RuleScore>,
    pub top_rule: Option<RuleScore>,
    pub seconds: f64,
}

fn process(root: &Path, out: &Path, rel: &str, prev: Option<&FileState>) -> Result<(FileState, bool)> {
    let path = root.join(rel);
    let (mtime, size) = mtime_size(&path)?;
    let outfile = out.join("files").join(format!("{}.json", file_key(rel)));
    let prev = prev.filter(|p| p.format == FORMAT && outfile.exists());
    if let Some(p) = prev {
        if p.mtime == mtime && p.size == size {
            return Ok((p.clone(), false));
        }
    }
    let bytes = std::fs::read(&path)?;
    let hash = format!("{:016x}", fnv64(&bytes));
    if let Some(p) = prev.filter(|p| p.hash == hash) {
        return Ok((FileState { mtime, size, ..p.clone() }, false));
    }
    let project = rel.split('/').next().unwrap_or_default();
    let (run, moments) = measure(&bytes, project, rel);
    let brief = if run.subagent { truncate(&first_prompt(&bytes), BRIEF_MAX) } else { String::new() };
    let events = run.events;
    std::fs::write(&outfile, serde_json::to_string(&FileOut { run, moments, brief })?)?;
    Ok((FileState { format: FORMAT, mtime, size, hash, events }, true))
}

pub fn mine_pr_gap(root: &Path, out: &Path, since: &str, jobs: usize, exclude: &[String],
                   gh: Option<&dyn Github>) -> Result<Summary> {
    crate::miner::ensure_private_out("pr-gap", out)?;
    let t0 = Instant::now();
    std::fs::create_dir_all(out.join("files"))?;
    let state_path = out.join("pr-gap-state.json");
    let state: BTreeMap<String, FileState> = match std::fs::read_to_string(&state_path) {
        Ok(t) => serde_json::from_str(&t).context("pr-gap-state.json")?,
        Err(_) => BTreeMap::new(),
    };
    let since_secs = since_epoch(since);
    let files: Vec<String> = transcripts(root)?.into_iter()
        .filter(|r| !exclude.iter().any(|x| !x.is_empty() && r.contains(x.as_str())))
        .filter(|r| since_secs.is_none_or(|s| mtime_size(&root.join(r)).map(|(m, _)| m >= s).unwrap_or(true)))
        .collect();
    let next = AtomicUsize::new(0);
    let processed = AtomicUsize::new(0);
    let results: Mutex<BTreeMap<String, FileState>> = Mutex::new(BTreeMap::new());
    let errors: Mutex<Vec<String>> = Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| {
                while let Some(rel) = files.get(next.fetch_add(1, Ordering::SeqCst)) {
                    match process(root, out, rel, state.get(rel)) {
                        Ok((st, did)) => {
                            if did {
                                processed.fetch_add(1, Ordering::SeqCst);
                            }
                            results.lock().unwrap().insert(rel.clone(), st);
                        }
                        Err(e) => errors.lock().unwrap().push(format!("{rel}: {e:#}")),
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    for e in &errors {
        eprintln!("[pr-gap] skipped {e}");
    }
    let results = results.into_inner().unwrap();
    let keep: HashSet<String> = results.keys().map(|r| format!("{}.json", file_key(r))).collect();
    for ent in std::fs::read_dir(out.join("files"))? {
        let p = ent?.path();
        if !keep.contains(&p.file_name().unwrap_or_default().to_string_lossy().into_owned()) {
            std::fs::remove_file(p)?;
        }
    }
    let mut sum = Summary { transcripts: files.len(), processed: processed.load(Ordering::SeqCst), ..Default::default() };
    sum.skipped_unchanged = sum.transcripts - sum.processed - errors.len();
    let mut fos: Vec<FileOut> = vec![];
    for rel in results.keys() {
        fos.push(serde_json::from_str(&std::fs::read_to_string(out.join("files").join(format!("{}.json", file_key(rel))))?)?);
    }
    // Runs that began before `since` are kept only if they were active after it.
    if !since.is_empty() {
        fos.retain(|f| f.run.end.as_str() >= since);
    }
    let mut runs: Vec<Run> = fos.iter().map(|f| f.run.clone()).collect();
    sum.github = match gh {
        Some(gh) => {
            let cache_path = out.join("github-cache.json");
            let mut cache: BTreeMap<String, Vec<GithubPr>> = std::fs::read_to_string(&cache_path).ok()
                .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            let (asked, failed) = join_github(&mut runs, gh, &mut cache);
            std::fs::write(&cache_path, serde_json::to_string_pretty(&cache)?)?;
            let mut c = BTreeMap::<&str, usize>::new();
            for r in runs.iter().filter(|r| r.outcome == Outcome::PushedNoPr) {
                *c.entry(github_status(r, &cache)).or_insert(0) += 1;
            }
            json!({"joined": true, "lookups": asked, "failed": failed, "branches_cached": cache.len(),
                   "pr_existing": runs.iter().filter(|r| r.outcome == Outcome::PrExisting).count(),
                   "pushed_no_pr": c})
        }
        None => json!({"joined": false}),
    };
    let mut eps = vec![];
    let (mut sessions, mut projects, mut hit_sessions) = (HashSet::new(), HashSet::new(), HashSet::new());
    let mut text = String::new();
    let mut runs_text = String::new();
    let mut exams = String::new();
    for (f, run) in fos.iter().zip(&runs) {
        sessions.insert(run.session_id.clone());
        projects.insert(run.project.clone());
        sum.events += run.events;
        let moments: Vec<Moment> = f.moments.iter().filter(|m| since.is_empty() || m.ts.as_str() >= since).cloned().collect();
        eps.extend(episodes(run, &moments, &f.brief));
        if run.commits > 0 {
            runs_text.push_str(&serde_json::to_string(run)?);
            runs_text.push('\n');
        }
        if run.subagent && run.outcome.is_gap() {
            exams.push_str(&serde_json::to_string(&json!({
                "run": run.id, "file": run.file, "brief": run.brief, "brief_phrases": run.brief_phrases,
                "brief_head": truncate(&f.brief, 300), "outcome": run.outcome,
                "before": format!("the agent ended with {} commits and {}", run.commits,
                                  if run.pushes > 0 { "a pushed branch but no PR" } else { "nothing pushed" }),
                "check": "replay the brief; after the agent's first `git push`, `gh pr list --head <branch>` returns a PR \
                          (and a draft PR exists within the first push)"}))?);
            exams.push('\n');
        }
    }
    for e in &eps {
        hit_sessions.insert(e.session_id.clone());
        for (sig, n) in &e.signals {
            *sum.episodes_per_signal.entry(sig.name()).or_insert(0) += 1;
            *sum.hits_per_signal.entry(sig.name()).or_insert(0) += n;
        }
        text.push_str(&serde_json::to_string(e)?);
        text.push('\n');
    }
    for s in Signal::ALL {
        sum.episodes_per_signal.entry(s.name()).or_insert(0);
        sum.hits_per_signal.entry(s.name()).or_insert(0);
    }
    sum.all = population(runs.iter());
    sum.main_sessions = population(runs.iter().filter(|r| !r.subagent));
    sum.subagents = population(runs.iter().filter(|r| r.subagent));
    for r in runs.iter().filter(|r| r.subagent && r.commits > 0) {
        let e = sum.subagent_gap_by_brief.entry(format!("{:?}", r.brief).to_lowercase()).or_insert((0, 0));
        e.0 += 1;
        e.1 += usize::from(r.outcome.is_gap());
    }
    let active: Vec<&Run> = runs.iter().filter(|r| r.commits > 0).collect();
    sum.rules = score_rules(&active);
    sum.top_rule = top_rule(&sum.rules).cloned();
    sum.episodes = eps.len();
    sum.sessions = sessions.len();
    sum.projects = projects.len();
    sum.sessions_with_episodes = hit_sessions.len();
    std::fs::write(out.join("episodes.jsonl"), text)?;
    std::fs::write(out.join("runs.jsonl"), runs_text)?;
    std::fs::write(out.join("pr-gap-exam-candidates.jsonl"), exams)?;
    std::fs::write(&state_path, serde_json::to_string_pretty(&results)?)?;
    std::fs::write(out.join("pr-gap-task.json"), serde_json::to_string_pretty(&task(&sum))?)?;
    sum.seconds = t0.elapsed().as_secs_f64();
    std::fs::write(out.join("pr-gap-summary.json"), serde_json::to_string_pretty(&sum)?)?;
    Ok(sum)
}

/// The top rule as a harness task, in the shape `rrsi harness mine` writes.
pub fn task(sum: &Summary) -> Value {
    let top = sum.top_rule.clone().unwrap_or_default();
    let a = &sum.all;
    json!({
        "id": "active_agent_without_pr",
        "title": "Active agents run without a pushed branch or a PR",
        "struggle_pattern": format!("{} of {} active runs (at least one commit) ended with no PR: {} never pushed, {} pushed without a PR",
            a.gap_runs, a.active_runs, a.outcomes.get("never_pushed").unwrap_or(&0), a.outcomes.get("pushed_no_pr").unwrap_or(&0)),
        "root_cause_hypothesis": "briefs defer the PR to the end of the slice (or forbid pushing), and nothing notices an agent sitting on unpushed commits or a PR-less branch",
        "proposed_fix": {"kind": "claude_md_rule",
            "change": "agent operating rule and brief snippet: commit, push and open a DRAFT PR within the first minutes; a hook watches for the trigger rule below"},
        "trigger_rule": {"when": format!("first commit {} min old with no push, or a pushed branch {} min without a PR", top.minutes, top.minutes),
            "owner_resolution": "the agent whose transcript made the commit (its worktree branch)",
            "payload": "open a draft PR for the branch and tell the agent its URL",
            "rule": top.rule},
        "acceptance_check": "re-run `rrsi-mine pr-gap`: gap runs and the rule's fires fall toward zero for runs started after the rule shipped",
        "priority": "P1",
        "exam_candidate": {"checkable": true, "before": "a replayed brief ends with commits and no PR",
            "after": "a draft PR exists for the branch within the first push",
            "check": "after the first `git push` in the replay, `gh pr list --head <branch>` is non-empty"},
        "evidence": {"active_runs": a.active_runs, "gap_runs": a.gap_runs, "outcomes": a.outcomes,
                     "rule": {"fires": top.fires, "gaps_caught": top.gaps_caught, "nags": top.nags, "gaps_missed": top.gaps_missed},
                     "episodes_per_signal": sum.episodes_per_signal},
    })
}

// ------------------------------------------------------------------ plugin

pub struct PrGap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    out: PathBuf,
    #[serde(default)]
    root: Option<PathBuf>,
    #[serde(default)]
    since: String,
    #[serde(default = "default_jobs")]
    jobs: usize,
    #[serde(default, deserialize_with = "crate::miner::one_or_many")]
    exclude: Vec<String>,
    #[serde(default)]
    github: bool,
}

fn default_jobs() -> usize {
    8
}

impl crate::miner::Miner for PrGap {
    fn name(&self) -> &'static str {
        "pr-gap"
    }
    fn about(&self) -> &'static str {
        "Active agents without a PR: commit->push and push->PR latency, unpushed commits, briefs that defer the PR, operator corrections, scored trigger rules"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("out", "output directory (outside every work tree)"),
          ("root", "transcript root, default ~/.claude/projects"),
          ("since", "only runs active on/after this ISO date"), ("jobs", "parser threads, default 8"),
          ("exclude", "skip transcripts whose path contains this (repeatable)"),
          ("github", "join pushed branches against `gh pr list --head` (network; cached)")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("episodes.jsonl", "one gap episode (a run, a deferring brief or an operator correction) in the traces shape"),
          ("runs.jsonl", "one active run (transcript with a commit): steps, latencies, outcome, brief class"),
          ("pr-gap-summary.json", "per-signal counts, outcome and latency distributions, scored trigger rules"),
          ("pr-gap-task.json", "the top trigger rule as a harness task"),
          ("pr-gap-exam-candidates.jsonl", "subagent briefs that ended in the gap, to replay"),
          ("github-cache.json", "PRs per pushed branch (with --github)"),
          ("pr-gap-state.json", "per transcript mtime/size/hash for incremental runs")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = crate::miner::parse_args(self.name(), args)?;
        let root = a.root.unwrap_or_else(root_default);
        let gh = a.github.then_some(&GhCli as &dyn Github);
        Ok(serde_json::to_value(mine_pr_gap(&root, &a.out, &a.since, a.jobs, &a.exclude, gh)?)?)
    }
}

#[cfg(test)]
mod tests;
