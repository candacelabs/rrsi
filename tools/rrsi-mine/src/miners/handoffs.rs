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

//! Mine agent-to-agent coordination from Claude Code session transcripts:
//! moments where an agent should have messaged another session, and the
//! cross-session traffic that did happen (did it help or cause churn?).
//!
//! ```text
//! rrsi-mine handoffs --out DIR [--root ~/.claude/projects] [--since DATE] [--jobs 8] [--exclude S]
//! ```
//!
//! Detectors ([`Signal`]), all deterministic:
//!
//! - in one transcript: `user_relay` (the operator relays between sessions),
//!   `peer_message` / `coordinator_message` (messages that arrived),
//!   `peer_retraction` (a message withdraws, voids or renames an earlier
//!   decision), `peer_churn` (the second and later retractions from one
//!   sender), `message_out` (SendMessage / send_message calls),
//!   `vcs_conflict`, `worktree_collision`, `already_done`, `blocked_on_other`,
//!   `polling` (the same status check run again and again instead of
//!   asking), `ownership_question`, `claim` (claim/release comments);
//! - across transcripts: `file_overlap` (two sessions edit the same
//!   repository file within [`OVERLAP_SLACK_SECS`] of each other),
//!   `branch_overlap` (two sessions active on one branch at once),
//!   `duplicate_work` (two sessions open issues/PRs with near-identical
//!   titles).
//!
//! Every arrived peer message is scored: did the receiver act (a tool call
//! before the next turn) and did it reply (a message tool call soon after)?
//!
//! Records are plain JSON in the episode shape of the `traces` miner
//! (`DIR/episodes.jsonl`, `DIR/handoffs-summary.json`), plus an `extra`
//! object per episode. `DIR` must be outside every git work tree.

use crate::transcript::*;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const EPISODE_GAP: usize = 6;
pub const CONTEXT_BEFORE: usize = 3;
pub const CONTEXT_AFTER: usize = 3;
pub const CONTEXT_MAX: usize = 24;
pub const TEXT_MAX: usize = 400;
pub const PEER_TEXT_MAX: usize = 700;
/// A receiver "acted" on a message if it called a tool within this many events.
pub const ACT_WINDOW: usize = 20;
/// It "replied" if it sent a message within this many events.
pub const REPLY_WINDOW: usize = 60;
pub const POLL_LOOKBACK: usize = 12;
pub const POLL_REPEATS: usize = 3;
pub const POLL_SIMILARITY: f64 = 0.8;
/// Two sessions' edits of one file this close in time count as overlapping.
pub const OVERLAP_SLACK_SECS: u64 = 30 * 60;
pub const TITLE_JACCARD: f64 = 0.5;
pub const TITLE_MIN_WORDS: usize = 3;

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Human { text: String },
    PeerIn { from_session: String, from_name: String, body: String },
    CoordinatorIn { body: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { id: String, is_error: bool, text: String },
    AssistantText { text: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ev {
    pub ts: String,
    pub cwd: String,
    pub branch: String,
    pub kind: Kind,
}

#[derive(Clone, Debug, Default)]
pub struct Session {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub events: Vec<Ev>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    UserRelay,
    PeerMessage,
    CoordinatorMessage,
    PeerRetraction,
    PeerChurn,
    MessageOut,
    VcsConflict,
    WorktreeCollision,
    AlreadyDone,
    BlockedOnOther,
    Polling,
    OwnershipQuestion,
    Claim,
    FileOverlap,
    BranchOverlap,
    DuplicateWork,
}

impl Signal {
    /// The detectors that look at one transcript.
    pub const IN_SESSION: [Signal; 13] = [Signal::UserRelay, Signal::PeerMessage, Signal::CoordinatorMessage,
        Signal::PeerRetraction, Signal::PeerChurn, Signal::MessageOut, Signal::VcsConflict,
        Signal::WorktreeCollision, Signal::AlreadyDone, Signal::BlockedOnOther, Signal::Polling,
        Signal::OwnershipQuestion, Signal::Claim];

    pub fn name(self) -> String {
        serde_json::to_value(self).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
    }

    pub fn detect(self, evs: &[Ev]) -> Vec<usize> {
        match self {
            Signal::UserRelay => humans_matching(evs, is_relay),
            Signal::PeerMessage => hits(evs, |k| matches!(k, Kind::PeerIn { .. })),
            Signal::CoordinatorMessage => hits(evs, |k| matches!(k, Kind::CoordinatorIn { .. })),
            Signal::PeerRetraction => hits(evs, |k| incoming(k).is_some_and(|(_, b)| is_retraction(b))),
            Signal::PeerChurn => detect_churn(evs),
            Signal::MessageOut => hits(evs, is_message_out),
            Signal::VcsConflict => hits(evs, |k| matches!(k, Kind::ToolResult { text, .. } if is_vcs_conflict(text))),
            Signal::WorktreeCollision => hits(evs, |k| matches!(k, Kind::ToolResult { text, .. }
                if WORKTREE_MARKERS.iter().any(|m| text.contains(m)))),
            Signal::AlreadyDone => said_matching(evs, ALREADY_DONE),
            Signal::BlockedOnOther => hits(evs, |k| matches!(k, Kind::AssistantText { text }
                if contains_any(&text.to_lowercase(), BLOCKED_ON_OTHER))),
            Signal::Polling => detect_polling(evs),
            Signal::OwnershipQuestion => said_matching(evs, OWNERSHIP),
            Signal::Claim => hits(evs, |k| bash_command(k).is_some_and(is_claim)),
            Signal::FileOverlap | Signal::BranchOverlap | Signal::DuplicateWork => vec![],
        }
    }
}

// ---------------------------------------------------------------- parsing

/// `(from_session, from_name, body)` of a `<cross-session-message ...>` block.
pub fn parse_peer(text: &str) -> Option<(String, String, String)> {
    let a = text.find("<cross-session-message")?;
    let rest = &text[a..];
    let attr = |k: &str| {
        let pat = format!("{k}=\"");
        rest.find(&pat).map(|i| {
            let v = &rest[i + pat.len()..];
            v[..v.find('"').unwrap_or(v.len())].to_string()
        }).unwrap_or_default()
    };
    let open_end = rest.find('>')? + 1;
    let body = &rest[open_end..];
    let body = &body[..body.find("</cross-session-message>").unwrap_or(body.len())];
    Some((attr("from-session"), attr("from-name"), body.trim().to_string()))
}

pub const COORDINATOR_PREFIX: &str = "The coordinator sent a message while you were working";

fn origin(e: &Value) -> Option<&Value> {
    e.get("origin")
}

fn push(ses: &mut Session, ev: Ev) {
    // Queued and delivered copies of one message: keep one.
    let dup = ses.events.iter().rev().take(8).any(|p| match (&p.kind, &ev.kind) {
        (Kind::PeerIn { from_session: a, body: x, .. }, Kind::PeerIn { from_session: b, body: y, .. }) =>
            a == b && x.chars().take(120).eq(y.chars().take(120)),
        (Kind::Human { text: x }, Kind::Human { text: y }) => x == y,
        (Kind::CoordinatorIn { body: x }, Kind::CoordinatorIn { body: y }) => x == y,
        _ => false,
    });
    if !dup {
        ses.events.push(ev);
    }
}

fn incoming_from_text(text: &str, origin: Option<&Value>) -> Option<Kind> {
    if let Some(o) = origin.filter(|o| o.get("kind").and_then(Value::as_str) == Some("peer")) {
        let body = field(o, "body");
        let (fs, fname, b) = parse_peer(text).unwrap_or_default();
        return Some(Kind::PeerIn {
            from_session: Some(field(o, "fromSession")).filter(|s| !s.is_empty()).unwrap_or(fs),
            from_name: Some(field(o, "name")).filter(|s| !s.is_empty()).unwrap_or(fname),
            body: if body.is_empty() { b } else { body },
        });
    }
    if let Some((from_session, from_name, body)) = parse_peer(text) {
        return Some(Kind::PeerIn { from_session, from_name, body });
    }
    let coord = origin.and_then(|o| o.get("kind")).and_then(Value::as_str) == Some("coordinator")
        || text.trim_start().starts_with(COORDINATOR_PREFIX);
    coord.then(|| Kind::CoordinatorIn { body: text.trim().to_string() })
}

pub fn parse_session<R: BufRead>(r: R) -> Session {
    let mut ses = Session::default();
    for line in r.lines() {
        let Ok(line) = line else { continue };
        let Ok(e) = serde_json::from_str::<Value>(&line) else { continue };
        parse_event(&e, &mut ses);
    }
    ses
}

fn parse_event(e: &Value, ses: &mut Session) {
    if ses.session_id.is_empty() {
        ses.session_id = field(e, "sessionId");
    }
    if ses.agent_id.is_none() {
        ses.agent_id = e.get("agentId").and_then(Value::as_str).map(str::to_string);
    }
    let base = |kind| Ev { ts: field(e, "timestamp"), cwd: field(e, "cwd"), branch: field(e, "gitBranch"), kind };
    let sidechain = e.get("isSidechain").and_then(Value::as_bool).unwrap_or(false);
    match e.get("type").and_then(Value::as_str).unwrap_or_default() {
        "assistant" => {
            let Some(Value::Array(bs)) = e.get("message").and_then(|m| m.get("content")) else { return };
            for b in bs {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") if !field(b, "text").trim().is_empty() =>
                        push(ses, base(Kind::AssistantText { text: field(b, "text") })),
                    Some("tool_use") => push(ses, base(Kind::ToolUse {
                        id: field(b, "id"), name: field(b, "name"), input: b.get("input").cloned().unwrap_or(Value::Null),
                    })),
                    _ => {}
                }
            }
        }
        "user" => {
            let content = e.get("message").and_then(|m| m.get("content")).cloned().unwrap_or(Value::Null);
            if let Value::Array(bs) = &content {
                let results: Vec<&Value> = bs.iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result")).collect();
                if !results.is_empty() {
                    for b in results {
                        push(ses, base(Kind::ToolResult {
                            id: field(b, "tool_use_id"),
                            is_error: b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                            text: content_text(b.get("content").unwrap_or(&Value::Null)),
                        }));
                    }
                    return;
                }
            }
            let text = content_text(&content);
            if let Some(k) = incoming_from_text(&text, origin(e)) {
                push(ses, base(k));
                return;
            }
            if e.get("isMeta").and_then(Value::as_bool).unwrap_or(false) || text.trim_start().starts_with("[Request interrupted") {
                return;
            }
            let human = match origin_kind(e).as_deref() {
                Some("human") => true,
                Some(_) => false,
                None => !sidechain && !text.trim_start().starts_with(['<', '[']),
            };
            let text = strip_reminders(&text);
            if human && !text.is_empty() {
                push(ses, base(Kind::Human { text }));
            }
        }
        "attachment" => {
            let Some(a) = e.get("attachment") else { return };
            if a.get("type").and_then(Value::as_str) != Some("queued_command") {
                return;
            }
            let prompt = field(a, "prompt");
            if let Some(k) = incoming_from_text(&prompt, a.get("origin")) {
                push(ses, base(k));
            } else if a.get("origin").and_then(|o| o.get("kind")).and_then(Value::as_str) == Some("human")
                || a.get("humanTurn").and_then(Value::as_bool).unwrap_or(false) {
                let text = strip_reminders(&prompt);
                if !text.is_empty() {
                    push(ses, base(Kind::Human { text }));
                }
            }
        }
        _ => {}
    }
}

// -------------------------------------------------------------- detectors

pub const RELAY: &[&str] = &["other agent", "other session", "other claude", "another session", "another agent",
    "another claude", "other chat", "other window", "other instance", "wrong session", "wrong chat",
    "wrong window", "tell the other", "relay this", "pass this to", "pass this along", "forward this",
    "from-session=", "cross-session-message", "other conversation", "parallel session"];
pub const RETRACTION: &[&str] = &["withdraw", "void", "disregard", "ignore my", "ignore the previous",
    "ignore that", "retract", "scratch that", "never mind", "nevermind", "correction:", "supersede",
    "no longer", "rename", "final name", "changed my mind", "reversed", "revert my", "instead of what i"];
pub const VCS_CONFLICT: &[&str] = &["CONFLICT (", "Merge conflict in", "could not apply", "Automatic merge failed",
    "needs merge", "Unmerged paths", "! [rejected]", "Updates were rejected", "non-fast-forward",
    "is already checked out at", "is already used by worktree", "rebase in progress",
    "stale info", "cannot lock ref"];
pub const WORKTREE_MARKERS: &[&str] = &["belongs to a different worktree", "Do not write to other worktrees"];
pub const ALREADY_DONE: &[&str] = &["already done by", "already been done", "already implemented",
    "already landed", "already merged", "already fixed", "already open", "already filed", "duplicate of",
    "duplicates #", "another session already", "other session already", "already working on",
    "already being worked", "already in flight", "already handled by", "superseded by", "already did this",
    "already claimed"];
pub const BLOCKED_ON_OTHER: &[&str] = &["waiting on the other", "waiting for the other", "waiting on another",
    "waiting for another", "blocked on the other", "blocked by the other", "blocked on another",
    "blocked by another", "owned by another", "another session owns", "other session owns",
    "owned by the other", "until the other", "once the other", "waiting for #", "waiting on #",
    "blocked on #", "blocked by #", "keep waiting for", "still waiting for", "waiting for that",
    "until it lands", "until that lands", "until it merges", "until that merges"];
pub const OWNERSHIP: &[&str] = &["who owns", "who's working", "who is working", "is anyone working",
    "is someone working", "anyone else working", "anyone already", "which session owns", "which session is",
    "whose is", "owner of this", "who's doing", "who is doing", "is this mine", "is that mine", "not mine",
    "isn't mine", "someone else's"];
pub const POLL_COMMANDS: &[&str] = &["gh pr view", "gh pr checks", "gh issue view", "gh run view", "gh run list",
    "gh run watch", "gh api repos", "git fetch", "git ls-remote", "git log origin", "gh pr list"];

fn contains_any(t: &str, pats: &[&str]) -> bool {
    pats.iter().any(|p| t.contains(p))
}

pub fn is_relay(t: &str) -> bool {
    contains_any(&t.to_lowercase(), RELAY)
}

pub fn is_retraction(t: &str) -> bool {
    contains_any(&t.to_lowercase(), RETRACTION)
}

pub fn is_vcs_conflict(t: &str) -> bool {
    contains_any(t, VCS_CONFLICT)
}

pub fn is_message_out(k: &Kind) -> bool {
    matches!(k, Kind::ToolUse { name, .. } if name == "SendMessage" || name.ends_with("send_message"))
}

pub fn bash_command(k: &Kind) -> Option<&str> {
    match k {
        Kind::ToolUse { name, input, .. } if name == "Bash" => input.get("command").and_then(Value::as_str),
        _ => None,
    }
}

pub fn is_claim(cmd: &str) -> bool {
    let c = cmd.to_lowercase();
    (c.contains("gh issue comment") || c.contains("gh pr comment") || c.contains("/comments"))
        && (c.contains("claimed by") || c.contains("released by") || c.contains("claiming") || c.contains("releasing"))
}

fn incoming(k: &Kind) -> Option<(&str, &str)> {
    match k {
        Kind::PeerIn { from_session, body, .. } => Some((from_session.as_str(), body.as_str())),
        Kind::CoordinatorIn { body } => Some(("coordinator", body.as_str())),
        _ => None,
    }
}

fn hits(evs: &[Ev], f: impl Fn(&Kind) -> bool) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| f(&e.kind).then_some(i)).collect()
}

fn humans_matching(evs: &[Ev], f: impl Fn(&str) -> bool) -> Vec<usize> {
    hits(evs, |k| matches!(k, Kind::Human { text } if f(text)))
}

/// Human or assistant text containing one of `pats` (case-insensitive).
fn said_matching(evs: &[Ev], pats: &[&str]) -> Vec<usize> {
    hits(evs, |k| match k {
        Kind::Human { text } | Kind::AssistantText { text } => contains_any(&text.to_lowercase(), pats),
        _ => false,
    })
}

/// The second and later retractions from one sender in one transcript
/// (rename storms, decisions voided and re-made).
pub fn detect_churn(evs: &[Ev]) -> Vec<usize> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let mut out = vec![];
    for (i, e) in evs.iter().enumerate() {
        if let Some((from, body)) = incoming(&e.kind) {
            if is_retraction(body) {
                let n = seen.entry(from).or_insert(0);
                *n += 1;
                if *n >= 2 {
                    out.push(i);
                }
            }
        }
    }
    out
}

/// The same status check ([`POLL_COMMANDS`]) run [`POLL_REPEATS`] or more
/// times within [`POLL_LOOKBACK`] tool calls, with no message sent between.
pub fn detect_polling(evs: &[Ev]) -> Vec<usize> {
    let calls: Vec<(usize, Option<&str>, bool)> = evs.iter().enumerate()
        .filter(|(_, e)| matches!(e.kind, Kind::ToolUse { .. }))
        .map(|(i, e)| (i, bash_command(&e.kind).filter(|c| contains_any(c, POLL_COMMANDS)), is_message_out(&e.kind)))
        .collect();
    let mut out = vec![];
    for (k, (i, cmd, _)) in calls.iter().enumerate() {
        let Some(cmd) = cmd else { continue };
        let tc = trigrams(cmd);
        let mut same = 0;
        for (_, prev, msg) in calls[k.saturating_sub(POLL_LOOKBACK)..k].iter().rev() {
            if *msg {
                break;
            }
            if prev.is_some_and(|p| p == *cmd || jaccard(&trigrams(p), &tc) >= POLL_SIMILARITY) {
                same += 1;
            }
        }
        if same + 1 >= POLL_REPEATS {
            out.push(*i);
        }
    }
    out
}

/// How the receiver handled one arrived message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub event: usize,
    pub from: String,
    pub coordinator: bool,
    pub retraction: bool,
    /// A tool call (other than a message) before the next turn, within [`ACT_WINDOW`].
    pub acted: bool,
    /// A message sent within [`REPLY_WINDOW`] events.
    pub replied: bool,
}

pub fn outcomes(evs: &[Ev]) -> Vec<Outcome> {
    evs.iter().enumerate().filter_map(|(i, e)| {
        let (from, body) = incoming(&e.kind)?;
        let after = &evs[i + 1..];
        let acted = after.iter().take(ACT_WINDOW)
            .take_while(|n| !matches!(n.kind, Kind::Human { .. } | Kind::PeerIn { .. } | Kind::CoordinatorIn { .. }))
            .any(|n| matches!(n.kind, Kind::ToolUse { .. }) && !is_message_out(&n.kind));
        let replied = after.iter().take(REPLY_WINDOW).any(|n| is_message_out(&n.kind));
        Some(Outcome { event: i, from: from.to_string(), coordinator: matches!(e.kind, Kind::CoordinatorIn { .. }),
                       retraction: is_retraction(body), acted, replied })
    }).collect()
}

// ------------------------------------------------------- cross-session facts

/// What one transcript contributes to the cross-session detectors.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Facts {
    pub session_id: String,
    pub project: String,
    pub file: String,
    /// (repository file key, epoch seconds) per edit.
    pub edits: Vec<(String, u64)>,
    /// (repository, branch) -> [first, last] epoch seconds seen there.
    pub branches: Vec<(String, String, u64, u64)>,
    /// ("pr" | "issue", title, epoch seconds) per create command.
    pub titles: Vec<(String, String, u64)>,
}

/// A repository-relative key for a path, so the same file in two worktrees
/// of one repository matches: `<repo>//<path in repo>`.
pub fn repo_key(path: &str) -> String {
    match path.find("/.claude/worktrees/") {
        Some(i) => {
            let rest = &path[i + "/.claude/worktrees/".len()..];
            let inner = rest.find('/').map(|j| &rest[j + 1..]).unwrap_or("");
            format!("{}//{}", &path[..i], inner)
        }
        None => path.to_string(),
    }
}

/// The repository root a cwd belongs to (worktrees fold into their repo).
pub fn repo_of(cwd: &str) -> String {
    repo_key(cwd).split("//").next().unwrap_or_default().to_string()
}

/// `--title "..."`, `--title '...'`, `--title=...` or `-t "..."` in a command.
pub fn title_arg(cmd: &str) -> Option<String> {
    for flag in ["--title=", "--title ", "-t "] {
        if let Some(i) = cmd.find(flag) {
            let rest = cmd[i + flag.len()..].trim_start();
            let q = rest.chars().next()?;
            let t = if q == '"' || q == '\'' {
                let r = &rest[1..];
                &r[..r.find(q).unwrap_or(r.len())]
            } else {
                rest.split_whitespace().next().unwrap_or("")
            };
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

pub const EDIT_TOOLS: &[&str] = &["Edit", "Write", "MultiEdit", "NotebookEdit"];
pub const MAIN_BRANCHES: &[&str] = &["", "main", "master", "HEAD"];

pub fn facts(ses: &Session, project: &str, file: &str, since: u64) -> Facts {
    let mut f = Facts { session_id: ses.session_id.clone(), project: project.into(), file: file.into(), ..Facts::default() };
    let mut br: BTreeMap<(String, String), (u64, u64)> = BTreeMap::new();
    for e in &ses.events {
        let Some(t) = iso_secs(&e.ts).filter(|t| *t >= since) else { continue };
        if !MAIN_BRANCHES.contains(&e.branch.as_str()) && !e.cwd.is_empty() {
            let r = br.entry((repo_of(&e.cwd), e.branch.clone())).or_insert((t, t));
            r.0 = r.0.min(t);
            r.1 = r.1.max(t);
        }
        if let Kind::ToolUse { name, input, .. } = &e.kind {
            if EDIT_TOOLS.contains(&name.as_str()) {
                if let Some(p) = input.get("file_path").or_else(|| input.get("notebook_path")).and_then(Value::as_str) {
                    f.edits.push((repo_key(p), t));
                }
            }
            if let Some(cmd) = bash_command(&e.kind) {
                for (kind, pat) in [("pr", "gh pr create"), ("issue", "gh issue create")] {
                    if cmd.contains(pat) {
                        if let Some(title) = title_arg(cmd) {
                            f.titles.push((kind.into(), title, t));
                        }
                    }
                }
            }
        }
    }
    f.branches = br.into_iter().map(|((r, b), (a, z))| (r, b, a, z)).collect();
    f
}

fn ts_of(secs: u64) -> String {
    // Days to civil (Howard Hinnant), for cross-session episode timestamps.
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let s = secs % 86400;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.000Z", s / 3600, s / 60 % 60, s % 60)
}

fn overlaps(a: (u64, u64), b: (u64, u64), slack: u64) -> bool {
    a.0 <= b.1 + slack && b.0 <= a.1 + slack
}

/// Cross-session episodes from every transcript's facts. Transcripts of one
/// session (main + subagents) are one party; pairs are unordered.
pub fn cross_episodes(all: &[Facts]) -> Vec<Episode> {
    #[derive(Default)]
    struct Party<'a> {
        project: &'a str,
        file: &'a str,
        edits: BTreeMap<&'a str, (u64, u64)>,
        branches: BTreeMap<(&'a str, &'a str), (u64, u64)>,
        titles: Vec<(&'a str, &'a str, u64)>,
    }
    let mut parties: BTreeMap<&str, Party> = BTreeMap::new();
    for f in all {
        if f.session_id.is_empty() {
            continue;
        }
        let p = parties.entry(&f.session_id).or_default();
        if p.file.is_empty() || !f.file.contains("/subagents/") {
            p.project = &f.project;
            p.file = &f.file;
        }
        for (k, t) in &f.edits {
            let r = p.edits.entry(k).or_insert((*t, *t));
            *r = (r.0.min(*t), r.1.max(*t));
        }
        for (repo, b, a, z) in &f.branches {
            let r = p.branches.entry((repo, b)).or_insert((*a, *z));
            *r = (r.0.min(*a), r.1.max(*z));
        }
        p.titles.extend(f.titles.iter().map(|(k, t, s)| (k.as_str(), t.as_str(), *s)));
    }
    let ids: Vec<&str> = parties.keys().copied().collect();
    let mut out = vec![];
    for (x, a) in ids.iter().enumerate() {
        for b in &ids[x + 1..] {
            let (pa, pb) = (&parties[a], &parties[b]);
            // Files: group by repository.
            let mut by_repo: BTreeMap<String, Vec<(&str, u64, u64)>> = BTreeMap::new();
            for (k, ia) in &pa.edits {
                if let Some(ib) = pb.edits.get(k) {
                    if overlaps(*ia, *ib, OVERLAP_SLACK_SECS) {
                        by_repo.entry(repo_of(k)).or_default().push((k, ia.0.min(ib.0), ia.1.max(ib.1)));
                    }
                }
            }
            for (repo, files) in by_repo {
                let (s, e) = (files.iter().map(|f| f.1).min().unwrap_or(0), files.iter().map(|f| f.2).max().unwrap_or(0));
                let lines: Vec<String> = files.iter().take(12).map(|f| f.0.split("//").nth(1).unwrap_or(f.0).to_string()).collect();
                out.push(cross(Signal::FileOverlap, (a, pa.project, pa.file), (b, pb.project), (s, e), files.len(),
                    format!("both sessions edited {} file(s) of {} within {} min: {}", files.len(),
                            repo.rsplit('/').next().unwrap_or(&repo), OVERLAP_SLACK_SECS / 60, lines.join(", ")),
                    json!({"repo_files": files.len()})));
            }
            for ((repo, br), ia) in &pa.branches {
                if let Some(ib) = pb.branches.get(&(*repo, *br)) {
                    if overlaps(*ia, *ib, 0) {
                        out.push(cross(Signal::BranchOverlap, (a, pa.project, pa.file), (b, pb.project),
                            (ia.0.max(ib.0), ia.1.min(ib.1)), 1,
                            format!("both sessions were on branch {br} of {} at the same time",
                                    repo.rsplit('/').next().unwrap_or(repo)),
                            json!({"branch": br})));
                    }
                }
            }
            for (ka, ta, sa) in &pa.titles {
                let wa = content_words(ta);
                if wa.len() < TITLE_MIN_WORDS {
                    continue;
                }
                for (kb, tb, sb) in &pb.titles {
                    let wb = content_words(tb);
                    if wb.len() >= TITLE_MIN_WORDS && jaccard(&wa, &wb) >= TITLE_JACCARD {
                        out.push(cross(Signal::DuplicateWork, (a, pa.project, pa.file), (b, pb.project),
                            ((*sa).min(*sb), (*sa).max(*sb)), 1,
                            format!("{ka} \"{}\" and {kb} \"{}\" opened by two sessions", truncate(ta, 120), truncate(tb, 120)),
                            json!({"kinds": [ka, kb]})));
                    }
                }
            }
        }
    }
    out
}

fn cross(sig: Signal, a: (&str, &str, &str), b: (&str, &str), (start, end): (u64, u64), n: usize,
         text: String, mut extra: Value) -> Episode {
    extra["other_session"] = json!(b.0);
    extra["other_project"] = json!(b.1);
    let ctx = CtxEv { i: 0, ts: ts_of(start), kind: "cross_session".into(), tool: None, is_error: None,
                      signals: vec![sig], text };
    Episode {
        id: format!("{:016x}", fnv64(format!("{}#{}#{}#{}", sig.name(), a.0, b.0, ctx.text).as_bytes())),
        project: a.1.into(), session_id: a.0.into(), agent_id: None, subagent: false, file: a.2.into(),
        start: ts_of(start), end: ts_of(end), start_event: 0, end_event: 0,
        signals: [(sig, n)].into(), user_turn: String::new(), context: vec![ctx],
        counts: Counts::default(), extra,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
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

/// Same shape as a `traces` episode, plus `extra`.
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
    /// Peer-message outcomes in this episode, or the other party of a
    /// cross-session episode.
    pub extra: Value,
}

fn ctx(evs: &[Ev], i: usize, hits: &BTreeMap<usize, Vec<Signal>>) -> CtxEv {
    let e = &evs[i];
    let (kind, tool, is_error, text) = match &e.kind {
        Kind::Human { text } => ("human", None, None, truncate(text, TEXT_MAX)),
        Kind::PeerIn { from_name, body, .. } =>
            ("peer_message", None, None, truncate(&format!("from \"{from_name}\": {body}"), PEER_TEXT_MAX)),
        Kind::CoordinatorIn { body } => ("coordinator_message", None, None, truncate(body, PEER_TEXT_MAX)),
        Kind::ToolUse { name, input, .. } => ("tool_use", Some(name.clone()), None, truncate(&input.to_string(), TEXT_MAX)),
        Kind::ToolResult { id, is_error, text } => {
            let tool = evs[..i].iter().rev().find_map(|p| match &p.kind {
                Kind::ToolUse { id: u, name, .. } if u == id => Some(name.clone()),
                _ => None,
            });
            ("tool_result", tool, Some(*is_error), truncate(text, TEXT_MAX))
        }
        Kind::AssistantText { text } => ("assistant", None, None, truncate(text, TEXT_MAX)),
    };
    CtxEv { i, ts: e.ts.clone(), kind: kind.into(), tool, is_error,
            signals: hits.get(&i).cloned().unwrap_or_default(), text }
}

pub fn detect_all(evs: &[Ev]) -> BTreeMap<usize, Vec<Signal>> {
    let mut hits: BTreeMap<usize, Vec<Signal>> = BTreeMap::new();
    for sig in Signal::IN_SESSION {
        for i in sig.detect(evs) {
            hits.entry(i).or_default().push(sig);
        }
    }
    hits
}

pub fn episodes(ses: &Session, project: &str, file: &str) -> Vec<Episode> {
    let evs = &ses.events;
    let hits = detect_all(evs);
    let outs = outcomes(evs);
    let mut spans: Vec<(usize, usize)> = vec![];
    for &i in hits.keys() {
        match spans.last_mut() {
            Some((_, end)) if i - *end <= EPISODE_GAP => *end = i,
            _ => spans.push((i, i)),
        }
    }
    spans.into_iter().map(|(a, b)| {
        let mut signals = BTreeMap::new();
        for (_, sigs) in hits.range(a..=b) {
            for s in sigs {
                *signals.entry(*s).or_insert(0) += 1;
            }
        }
        let lo = a.saturating_sub(CONTEXT_BEFORE);
        let hi = (b + CONTEXT_AFTER).min(evs.len() - 1);
        let idxs: Vec<usize> = if hi - lo < CONTEXT_MAX { (lo..=hi).collect() }
            else { (lo..lo + CONTEXT_MAX / 2).chain(hi + 1 - CONTEXT_MAX / 2..=hi).collect() };
        let user_turn = evs[..=a].iter().rev().find_map(|e| match &e.kind {
            Kind::Human { text } => Some(truncate(text, 800)),
            _ => None,
        }).unwrap_or_default();
        let span = &evs[a..=b];
        let here: Vec<&Outcome> = outs.iter().filter(|o| (a..=b).contains(&o.event)).collect();
        Episode {
            id: format!("{:016x}", fnv64(format!("handoffs#{file}#{a}").as_bytes())),
            project: project.into(), session_id: ses.session_id.clone(), agent_id: ses.agent_id.clone(),
            subagent: file.contains("/subagents/"), file: file.into(),
            start: evs[a].ts.clone(), end: evs[b].ts.clone(), start_event: a, end_event: b,
            signals, user_turn, context: idxs.into_iter().map(|i| ctx(evs, i, &hits)).collect(),
            counts: Counts {
                span_events: span.len(),
                tool_calls: span.iter().filter(|e| matches!(e.kind, Kind::ToolUse { .. })).count(),
                tool_errors: span.iter().filter(|e| matches!(e.kind, Kind::ToolResult { is_error: true, .. })).count(),
                human_turns: span.iter().filter(|e| matches!(e.kind, Kind::Human { .. })).count(),
                session_events: evs.len(),
            },
            extra: if here.is_empty() { json!({}) } else { json!({"outcomes": here}) },
        }
    }).collect()
}

// ------------------------------------------------------------------ driver

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileState {
    pub mtime: u64,
    pub size: u64,
    pub hash: String,
    pub since: String,
    pub events: usize,
}

/// One transcript's cached result.
#[derive(Debug, Default, Serialize, Deserialize)]
struct FileOut {
    episodes: Vec<Episode>,
    outcomes: Vec<Outcome>,
    message_out: usize,
    message_out_failed: usize,
    facts: Facts,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PeerSummary {
    pub messages: usize,
    pub from_peers: usize,
    pub from_coordinator: usize,
    pub acted: usize,
    pub replied: usize,
    pub retractions: usize,
    pub senders: usize,
    pub message_out_calls: usize,
    pub message_out_failed: usize,
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
    pub peer: PeerSummary,
    pub seconds: f64,
}

fn process(root: &Path, out: &Path, rel: &str, since: &str, prev: Option<&FileState>) -> Result<(FileState, bool)> {
    let path = root.join(rel);
    let (mtime, size) = mtime_size(&path)?;
    let outfile = out.join("files").join(format!("{}.json", file_key(rel)));
    if let Some(p) = prev.filter(|p| p.since == since && outfile.exists()) {
        if p.mtime == mtime && p.size == size {
            return Ok((p.clone(), false));
        }
    }
    let bytes = std::fs::read(&path)?;
    let hash = format!("{:016x}", fnv64(&bytes));
    if let Some(p) = prev.filter(|p| p.since == since && p.hash == hash && outfile.exists()) {
        return Ok((FileState { mtime, size, ..p.clone() }, false));
    }
    let ses = parse_session(BufReader::new(&bytes[..]));
    let project = rel.split('/').next().unwrap_or_default();
    let keep = |ts: &str| since.is_empty() || ts >= since;
    let outs: Vec<Outcome> = outcomes(&ses.events).into_iter().filter(|o| keep(&ses.events[o.event].ts)).collect();
    let msgs: Vec<&Ev> = ses.events.iter().filter(|e| is_message_out(&e.kind) && keep(&e.ts)).collect();
    let failed = msgs.iter().filter(|e| match &e.kind {
        Kind::ToolUse { id, .. } => ses.events.iter().any(|r| matches!(&r.kind, Kind::ToolResult { id: x, is_error: true, .. } if x == id)),
        _ => false,
    }).count();
    let fo = FileOut {
        episodes: episodes(&ses, project, rel).into_iter().filter(|e| keep(&e.start)).collect(),
        outcomes: outs, message_out: msgs.len(), message_out_failed: failed,
        facts: facts(&ses, project, rel, since_epoch(since).unwrap_or(0)),
    };
    std::fs::write(&outfile, serde_json::to_string(&fo)?)?;
    Ok((FileState { mtime, size, hash, since: since.into(), events: ses.events.len() }, true))
}

pub fn mine_handoffs(root: &Path, out: &Path, since: &str, jobs: usize, exclude: &[String]) -> Result<Summary> {
    crate::miner::ensure_private_out("handoffs", out)?;
    let t0 = Instant::now();
    std::fs::create_dir_all(out.join("files"))?;
    let state_path = out.join("handoffs-state.json");
    let state: BTreeMap<String, FileState> = match std::fs::read_to_string(&state_path) {
        Ok(t) => serde_json::from_str(&t).context("handoffs-state.json")?,
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
                    match process(root, out, rel, since, state.get(rel)) {
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
        eprintln!("[handoffs] skipped {e}");
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
    let (mut sessions, mut projects) = (HashSet::new(), HashSet::new());
    let mut eps = vec![];
    let mut all_facts = vec![];
    let mut senders = BTreeSet::new();
    for rel in results.keys() {
        let fo: FileOut = serde_json::from_str(&std::fs::read_to_string(out.join("files").join(format!("{}.json", file_key(rel))))?)?;
        projects.insert(rel.split('/').next().unwrap_or_default().to_string());
        sessions.insert(rel.split('/').take(2).collect::<Vec<_>>().join("/").trim_end_matches(".jsonl").to_string());
        for o in &fo.outcomes {
            sum.peer.messages += 1;
            if o.coordinator { sum.peer.from_coordinator += 1 } else { sum.peer.from_peers += 1 }
            sum.peer.acted += usize::from(o.acted);
            sum.peer.replied += usize::from(o.replied);
            sum.peer.retractions += usize::from(o.retraction);
            senders.insert(o.from.clone());
        }
        sum.peer.message_out_calls += fo.message_out;
        sum.peer.message_out_failed += fo.message_out_failed;
        eps.extend(fo.episodes);
        all_facts.push(fo.facts);
    }
    sum.peer.senders = senders.len();
    eps.extend(cross_episodes(&all_facts).into_iter().filter(|e| since.is_empty() || e.start.as_str() >= since));
    let mut text = String::new();
    let mut hit_sessions = HashSet::new();
    for e in &eps {
        hit_sessions.insert(e.session_id.clone());
        for (sig, n) in &e.signals {
            *sum.episodes_per_signal.entry(sig.name()).or_insert(0) += 1;
            *sum.hits_per_signal.entry(sig.name()).or_insert(0) += n;
        }
        text.push_str(&serde_json::to_string(e)?);
        text.push('\n');
    }
    sum.episodes = eps.len();
    sum.sessions = sessions.len();
    sum.projects = projects.len();
    sum.sessions_with_episodes = hit_sessions.len();
    sum.events = results.values().map(|s| s.events).sum();
    std::fs::write(out.join("episodes.jsonl"), text)?;
    std::fs::write(&state_path, serde_json::to_string_pretty(&results)?)?;
    sum.seconds = t0.elapsed().as_secs_f64();
    std::fs::write(out.join("handoffs-summary.json"), serde_json::to_string_pretty(&sum)?)?;
    Ok(sum)
}

// ------------------------------------------------------------- plugin

pub struct Handoffs;

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
}

fn default_jobs() -> usize {
    8
}

impl crate::miner::Miner for Handoffs {
    fn name(&self) -> &'static str {
        "handoffs"
    }
    fn about(&self) -> &'static str {
        "Cross-session coordination in Claude Code transcripts: relays, peer messages and their outcomes, overlaps, blocked-on-other, ownership"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("out", "output directory (outside every work tree)"),
          ("root", "transcript root, default ~/.claude/projects"),
          ("since", "only episodes on/after this ISO date"), ("jobs", "parser threads, default 8"),
          ("exclude", "skip transcripts whose path contains this (repeatable)")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("episodes.jsonl", "one coordination episode: signals, bounded context, counts, extra (outcomes / other session)"),
          ("handoffs-summary.json", "counts per signal and peer-message outcomes"),
          ("handoffs-state.json", "per transcript mtime/size/hash for incremental runs")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = crate::miner::parse_args(self.name(), args)?;
        let root = a.root.unwrap_or_else(root_default);
        Ok(serde_json::to_value(mine_handoffs(&root, &a.out, &a.since, a.jobs, &a.exclude)?)?)
    }
}

#[cfg(test)]
mod tests;
