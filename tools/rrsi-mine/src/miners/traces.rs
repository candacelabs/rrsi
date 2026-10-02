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

//! Mine agent *struggles* from Claude Code session transcripts.
//!
//! ```text
//! rrsi-mine traces --out DIR [--root ~/.claude/projects] [--since 2026-09-01] [--jobs 8] [--exclude S]
//! ```
//!
//! Every `<project>/<session>.jsonl` (and nested subagent transcript) is
//! parsed into a flat event list ([`Ev`]); named detectors ([`Signal`]), each a
//! pure function over that list, mark the events where the agent struggled;
//! hits close together are merged into one [`Episode`] with a bounded context
//! window. Nothing here calls a model.
//!
//! Output (the transcripts are private, so `DIR` must be outside every git
//! work tree, which is enforced):
//!
//! - `DIR/episodes/<file-key>.jsonl` — the episodes of one transcript
//! - `DIR/episodes.jsonl` — all episodes, rebuilt every run
//! - `DIR/traces-state.json` — per transcript: mtime, size, content hash
//!   (an unchanged transcript is skipped on the next run, resume-safe) and
//!   its [`Facts`]: tool calls per UTC hour and the Claude Desktop host
//!   markers, which `python -m rrsi harness measure` turns into the daily
//!   struggle rate per 1k tool calls and the host attribution
//! - `DIR/traces-summary.json` — counts per signal, sessions, projects

use crate::enclosing_work_tree;
pub use crate::transcript::*;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Hits at most this many events apart belong to one episode.
pub const EPISODE_GAP: usize = 6;
/// Context events kept around an episode (before its first / after its last hit).
pub const CONTEXT_BEFORE: usize = 3;
pub const CONTEXT_AFTER: usize = 2;
/// Most context events per episode; longer windows keep head and tail.
pub const CONTEXT_MAX: usize = 24;
pub const TEXT_MAX: usize = 400;
pub const USER_TURN_MAX: usize = 800;
/// A user turn longer than this is an instruction, not a correction.
pub const CORRECTION_MAX_CHARS: usize = 280;
pub const REASK_MIN_WORDS: usize = 4;
pub const REASK_JACCARD: f64 = 0.6;
pub const REASK_LOOKBACK: usize = 10;
pub const RETRY_LOOKBACK: usize = 4;
pub const RETRY_SIMILARITY: f64 = 0.85;

/// One normalized transcript event.
#[derive(Clone, Debug, PartialEq)]
pub enum EvKind {
    /// A turn typed by the operator (not a tool result, notification or meta text).
    Human { text: String },
    ToolUse { id: String, name: String, input: String },
    ToolResult { id: String, is_error: bool, text: String, denial: Option<String> },
    AssistantText { text: String },
    /// `[Request interrupted by user...]`.
    Interrupt,
    /// The harness told the agent the user has not heard from it in a while.
    SilenceReminder,
    /// A hook (e.g. the host's Stop hook) did not respond.
    HookNoResponse { text: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ev {
    pub ts: String,
    pub kind: EvKind,
}

/// One parsed transcript.
#[derive(Clone, Debug, Default)]
pub struct Session {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub events: Vec<Ev>,
}

/// The named struggle detectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    ToolError,
    Retry,
    HookTimeout,
    PermissionDenial,
    UserInterrupt,
    UserCorrection,
    Reask,
    Silence,
    TestFailure,
}

impl Signal {
    pub const ALL: [Signal; 9] = [Signal::ToolError, Signal::Retry, Signal::HookTimeout,
        Signal::PermissionDenial, Signal::UserInterrupt, Signal::UserCorrection, Signal::Reask,
        Signal::Silence, Signal::TestFailure];

    pub fn name(self) -> &'static str {
        match self {
            Signal::ToolError => "tool_error",
            Signal::Retry => "retry",
            Signal::HookTimeout => "hook_timeout",
            Signal::PermissionDenial => "permission_denial",
            Signal::UserInterrupt => "user_interrupt",
            Signal::UserCorrection => "user_correction",
            Signal::Reask => "reask",
            Signal::Silence => "silence",
            Signal::TestFailure => "test_failure",
        }
    }

    pub fn detect(self, evs: &[Ev]) -> Vec<usize> {
        match self {
            Signal::ToolError => detect_tool_error(evs),
            Signal::Retry => detect_retry(evs),
            Signal::HookTimeout => detect_hook_timeout(evs),
            Signal::PermissionDenial => detect_permission_denial(evs),
            Signal::UserInterrupt => detect_user_interrupt(evs),
            Signal::UserCorrection => detect_user_correction(evs),
            Signal::Reask => detect_reask(evs),
            Signal::Silence => detect_silence(evs),
            Signal::TestFailure => detect_test_failure(evs),
        }
    }
}

// ---------------------------------------------------------------- parsing





/// Parses one transcript (JSON lines; malformed lines are skipped).
pub fn parse_session<R: BufRead>(r: R) -> Session {
    let mut ses = Session::default();
    for line in r.lines() {
        let Ok(line) = line else { continue };
        let Ok(e) = serde_json::from_str::<Value>(&line) else { continue };
        parse_event(&e, &mut ses);
    }
    ses
}

fn push_human(ses: &mut Session, ts: String, text: String) {
    if text.is_empty() {
        return;
    }
    // A queued mid-turn message can be recorded twice; keep one.
    if let Some(Ev { kind: EvKind::Human { text: prev }, .. }) = ses.events.last() {
        if *prev == text {
            return;
        }
    }
    ses.events.push(Ev { ts, kind: EvKind::Human { text } });
}

fn parse_event(e: &Value, ses: &mut Session) {
    let ts = field(e, "timestamp");
    if ses.session_id.is_empty() {
        ses.session_id = field(e, "sessionId");
    }
    if ses.agent_id.is_none() {
        ses.agent_id = e.get("agentId").and_then(Value::as_str).map(str::to_string);
    }
    let sidechain = e.get("isSidechain").and_then(Value::as_bool).unwrap_or(false);
    match e.get("type").and_then(Value::as_str).unwrap_or_default() {
        "assistant" => {
            let Some(Value::Array(bs)) = e.get("message").and_then(|m| m.get("content")) else { return };
            for b in bs {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = field(b, "text");
                        if !text.trim().is_empty() {
                            ses.events.push(Ev { ts: ts.clone(), kind: EvKind::AssistantText { text } });
                        }
                    }
                    Some("tool_use") => ses.events.push(Ev { ts: ts.clone(), kind: EvKind::ToolUse {
                        id: field(b, "id"), name: field(b, "name"),
                        input: b.get("input").map(Value::to_string).unwrap_or_default(),
                    }}),
                    _ => {}
                }
            }
        }
        "user" => {
            let content = e.get("message").and_then(|m| m.get("content")).cloned().unwrap_or(Value::Null);
            let denial = e.get("toolDenialKind").and_then(Value::as_str).map(str::to_string);
            let mut had_result = false;
            if let Value::Array(bs) = &content {
                for b in bs {
                    if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                        had_result = true;
                        ses.events.push(Ev { ts: ts.clone(), kind: EvKind::ToolResult {
                            id: field(b, "tool_use_id"),
                            is_error: b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                            text: content_text(b.get("content").unwrap_or(&Value::Null)),
                            denial: denial.clone(),
                        }});
                    }
                }
            }
            if had_result {
                return;
            }
            let text = content_text(&content);
            if text.trim_start().starts_with("[Request interrupted by user") {
                ses.events.push(Ev { ts, kind: EvKind::Interrupt });
                return;
            }
            if e.get("isMeta").and_then(Value::as_bool).unwrap_or(false) {
                return;
            }
            let human = match origin_kind(e).as_deref() {
                Some("human") => true,
                Some(_) => false,
                // Older/plain main-session prompts carry no origin; a subagent's
                // first turn is its brief, never the operator.
                None => !sidechain && !text.trim_start().starts_with(['<', '[']),
            };
            if human {
                push_human(ses, ts, strip_reminders(&text));
            }
        }
        "attachment" => {
            let Some(a) = e.get("attachment") else { return };
            match a.get("type").and_then(Value::as_str) {
                Some("silent_turn_reminder") => ses.events.push(Ev { ts, kind: EvKind::SilenceReminder }),
                Some("hook_system_message") => {
                    let text = field(a, "content");
                    if text.contains("didn't respond") || text.contains("did not respond") {
                        ses.events.push(Ev { ts, kind: EvKind::HookNoResponse { text } });
                    }
                }
                Some("queued_command") => {
                    let human = a.get("origin").and_then(|o| o.get("kind")).and_then(Value::as_str) == Some("human")
                        || a.get("humanTurn").and_then(Value::as_bool).unwrap_or(false);
                    if human && a.get("commandMode").and_then(Value::as_str).unwrap_or("prompt") == "prompt" {
                        let ts = a.get("timestamp").and_then(Value::as_str).map(str::to_string).unwrap_or(ts);
                        push_human(ses, ts, strip_reminders(&field(a, "prompt")));
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

// -------------------------------------------------------------- detectors

pub const HOOK_TIMEOUT_MARKERS: [&str; 2] = ["hook did not respond", "hook didn't respond"];
/// The Claude Desktop host's worktree isolation guard rejecting a tool call.
pub const GUARD_MARKERS: [&str; 2] = ["isolated in the worktree", "running in an isolated git worktree"];
pub const DENIAL_MARKERS: [&str; 4] = [
    "doesn't want to proceed with this tool use",
    "denied by the Claude Code auto mode classifier",
    "Permission to use",
    "permission was denied",
];

fn is_hook_timeout(text: &str) -> bool {
    HOOK_TIMEOUT_MARKERS.iter().any(|m| text.contains(m))
}

pub fn is_guard_rejection(text: &str) -> bool {
    GUARD_MARKERS.iter().any(|m| text.contains(m))
}

fn is_denial(text: &str, denial: &Option<String>) -> bool {
    match denial.as_deref() {
        Some("interrupted") => false,
        Some(_) => true,
        None => DENIAL_MARKERS[..2].iter().any(|m| text.contains(m))
            || (text.starts_with(DENIAL_MARKERS[2]) && text.contains("denied"))
            || text.starts_with(DENIAL_MARKERS[3]),
    }
}

/// `tool_result.is_error`, except the kinds other detectors own (hook
/// timeouts, permission denials, a call cut off by the session ending).
pub fn detect_tool_error(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::ToolResult { is_error: true, text, denial, .. }
            if !is_hook_timeout(text) && !is_denial(text, denial) && denial.is_none() => Some(i),
        _ => None,
    }).collect()
}

pub fn detect_hook_timeout(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::ToolResult { text, .. } if is_hook_timeout(text) => Some(i),
        EvKind::HookNoResponse { .. } => Some(i),
        _ => None,
    }).collect()
}

pub fn detect_permission_denial(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::ToolResult { text, denial, .. } if !is_hook_timeout(text) && is_denial(text, denial) => Some(i),
        _ => None,
    }).collect()
}

pub fn detect_user_interrupt(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| (e.kind == EvKind::Interrupt).then_some(i)).collect()
}

pub fn detect_silence(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| (e.kind == EvKind::SilenceReminder).then_some(i)).collect()
}


pub const CORRECTION_WORDS: [&str; 13] = ["no", "nope", "nah", "wtf", "why", "stop", "again", "bruh",
    "wrong", "ugh", "huh", "wait", "undo"];
pub const CORRECTION_PHRASES: [&str; 18] = ["i said", "i told you", "i asked", "not what i", "that's not",
    "thats not", "you didn't", "you did not", "didnt", "still not", "still broken", "doesn't work",
    "didn't work", "not working", "why did you", "don't do", "you were supposed", "instead of"];
/// All-caps words that are names, not shouting.
pub const ACRONYMS: [&str; 24] = ["JSON", "HTML", "HTTP", "HTTPS", "YAML", "TODO", "README", "CLAUDE",
    "AGENTS", "NVIDIA", "API", "CLI", "SDK", "URL", "UUID", "SQL", "GPU", "CPU", "DNS", "TLS", "SSH",
    "LLM", "RRSI", "MCP"];

/// Shouting: two or more all-caps words of 4+ letters that are not
/// acronyms, or mostly upper-case text with at least 10 letters.
pub fn is_shouting(t: &str) -> bool {
    let caps = t.split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 4 && w.chars().all(char::is_uppercase) && !ACRONYMS.contains(w))
        .count();
    let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
    let upper = letters.iter().filter(|c| c.is_uppercase()).count();
    caps >= 2 || (letters.len() >= 10 && upper * 10 >= letters.len() * 7)
}

/// A short turn that pushes back on what the agent just did.
pub fn is_correction(t: &str) -> bool {
    if t.chars().count() > CORRECTION_MAX_CHARS {
        return false;
    }
    let ws = words(t);
    let joined = format!(" {} ", ws.join(" "));
    ws.iter().any(|w| CORRECTION_WORDS.contains(&w.as_str()))
        || CORRECTION_PHRASES.iter().any(|p| joined.contains(&format!(" {p} ")))
        || is_shouting(t)
}

/// A human turn right after the agent acted (tool call, result or reply).
fn after_agent_action(evs: &[Ev], i: usize) -> bool {
    evs[..i].iter().rev()
        .find(|e| !matches!(e.kind, EvKind::SilenceReminder | EvKind::HookNoResponse { .. }))
        .is_some_and(|e| matches!(e.kind, EvKind::ToolUse { .. } | EvKind::ToolResult { .. }
            | EvKind::AssistantText { .. } | EvKind::Interrupt))
}

pub fn detect_user_correction(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::Human { text } if after_agent_action(evs, i) && is_correction(text) => Some(i),
        _ => None,
    }).collect()
}




/// The operator asks (nearly) the same thing again after the agent already
/// answered it: content-word Jaccard >= [`REASK_JACCARD`] against one of the
/// last [`REASK_LOOKBACK`] human turns, with agent output in between.
pub fn detect_reask(evs: &[Ev]) -> Vec<usize> {
    let humans: Vec<(usize, BTreeSet<String>)> = evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::Human { text } => Some((i, content_words(text))),
        _ => None,
    }).collect();
    let mut hits = vec![];
    for (k, (j, wj)) in humans.iter().enumerate() {
        if wj.len() < REASK_MIN_WORDS {
            continue;
        }
        let again = humans[k.saturating_sub(REASK_LOOKBACK)..k].iter().any(|(i, wi)| {
            wi.len() >= REASK_MIN_WORDS && jaccard(wi, wj) >= REASK_JACCARD
                && evs[*i + 1..*j].iter().any(|e| matches!(e.kind, EvKind::AssistantText { .. } | EvKind::ToolUse { .. }))
        });
        if again {
            hits.push(*j);
        }
    }
    hits
}


fn result_of<'a>(evs: &'a [Ev], id: &str) -> Option<&'a EvKind> {
    evs.iter().map(|e| &e.kind).find(|k| matches!(k, EvKind::ToolResult { id: r, .. } if r == id))
}

fn failed(k: Option<&EvKind>) -> bool {
    matches!(k, Some(EvKind::ToolResult { is_error, text, .. }) if *is_error || is_test_failure(text))
}

/// The same tool called again with near-identical input (trigram Jaccard >=
/// [`RETRY_SIMILARITY`]) when the latest such call within [`RETRY_LOOKBACK`]
/// calls failed.
pub fn detect_retry(evs: &[Ev]) -> Vec<usize> {
    let uses: Vec<(usize, &str, &str, &str)> = evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::ToolUse { id, name, input } => Some((i, id.as_str(), name.as_str(), input.as_str())),
        _ => None,
    }).collect();
    let mut hits = vec![];
    for (k, (j, _, name, input)) in uses.iter().enumerate() {
        let tj = trigrams(input);
        // The latest similar earlier call decides: a retry only if it failed.
        let latest = uses[k.saturating_sub(RETRY_LOOKBACK)..k].iter().rev().find(|(_, _, pname, pinput)| {
            pname == name && (pinput == input || jaccard(&trigrams(pinput), &tj) >= RETRY_SIMILARITY)
        });
        if latest.is_some_and(|(_, pid, _, _)| failed(result_of(evs, pid))) {
            hits.push(*j);
        }
    }
    hits
}

/// Test or CI failures reported in tool output: go test, cargo, pytest,
/// bazel, GitHub Actions / `gh pr checks`.
pub fn is_test_failure(text: &str) -> bool {
    text.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("--- FAIL") || t.starts_with("FAIL\t") || (t.starts_with("FAIL ") && t.contains("[build failed]"))
            || t.contains("test result: FAILED") || t.starts_with("FAILED (failures")
            || (t.starts_with('=') && t.contains(" failed") && t.chars().any(|c| c.is_ascii_digit()))
            || (t.starts_with("FAILED ") && t.contains("::"))
            || t.contains(" FAILED in ") || t.contains("\tfail\t")
            || t.contains("\"conclusion\":\"failure\"") || t.contains("\"conclusion\": \"failure\"")
            || t.contains("Process completed with exit code")
            || t.starts_with("Tests failed") || t.contains("tests failed")
    })
}

pub fn detect_test_failure(evs: &[Ev]) -> Vec<usize> {
    evs.iter().enumerate().filter_map(|(i, e)| match &e.kind {
        EvKind::ToolResult { text, .. } if is_test_failure(text) => Some(i),
        _ => None,
    }).collect()
}

// --------------------------------------------------------------- episodes

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    /// Events in the episode's span (first to last hit).
    pub span_events: usize,
    pub tool_calls: usize,
    pub tool_errors: usize,
    pub human_turns: usize,
    /// Events in the whole transcript.
    pub session_events: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub project: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub subagent: bool,
    /// Transcript path relative to --root.
    pub file: String,
    pub start: String,
    pub end: String,
    pub start_event: usize,
    pub end_event: usize,
    /// Hits per signal in this episode.
    pub signals: BTreeMap<Signal, usize>,
    /// The operator turn the agent was working on (truncated).
    pub user_turn: String,
    pub context: Vec<CtxEv>,
    pub counts: Counts,
}



fn ctx(evs: &[Ev], i: usize, hits: &BTreeMap<usize, Vec<Signal>>) -> CtxEv {
    let e = &evs[i];
    let (kind, tool, is_error, text) = match &e.kind {
        EvKind::Human { text } => ("human", None, None, truncate(text, TEXT_MAX)),
        EvKind::ToolUse { name, input, .. } => ("tool_use", Some(name.clone()), None, truncate(input, TEXT_MAX)),
        EvKind::ToolResult { id, is_error, text, .. } => {
            let tool = evs[..i].iter().rev().find_map(|p| match &p.kind {
                EvKind::ToolUse { id: u, name, .. } if u == id => Some(name.clone()),
                _ => None,
            });
            ("tool_result", tool, Some(*is_error), truncate(text, TEXT_MAX))
        }
        EvKind::AssistantText { text } => ("assistant", None, None, truncate(text, TEXT_MAX)),
        EvKind::Interrupt => ("interrupt", None, None, String::new()),
        EvKind::SilenceReminder => ("silence_reminder", None, None, String::new()),
        EvKind::HookNoResponse { text } => ("hook_no_response", None, None, truncate(text, TEXT_MAX)),
    };
    CtxEv { i, ts: e.ts.clone(), kind: kind.into(), tool, is_error,
            signals: hits.get(&i).cloned().unwrap_or_default(), text }
}

/// All detector hits: event index -> signals.
pub fn detect_all(evs: &[Ev]) -> BTreeMap<usize, Vec<Signal>> {
    let mut hits: BTreeMap<usize, Vec<Signal>> = BTreeMap::new();
    for sig in Signal::ALL {
        for i in sig.detect(evs) {
            hits.entry(i).or_default().push(sig);
        }
    }
    hits
}

/// Merges hits at most [`EPISODE_GAP`] events apart into episodes.
pub fn episodes(ses: &Session, project: &str, file: &str) -> Vec<Episode> {
    let evs = &ses.events;
    let hits = detect_all(evs);
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
            EvKind::Human { text } => Some(truncate(text, USER_TURN_MAX)),
            _ => None,
        }).unwrap_or_default();
        let span = &evs[a..=b];
        let counts = Counts {
            span_events: span.len(),
            tool_calls: span.iter().filter(|e| matches!(e.kind, EvKind::ToolUse { .. })).count(),
            tool_errors: span.iter().filter(|e| matches!(e.kind, EvKind::ToolResult { is_error: true, .. })).count(),
            human_turns: span.iter().filter(|e| matches!(e.kind, EvKind::Human { .. })).count(),
            session_events: evs.len(),
        };
        Episode {
            id: format!("{:016x}", fnv64(format!("{file}#{a}").as_bytes())),
            project: project.into(), session_id: ses.session_id.clone(), agent_id: ses.agent_id.clone(),
            subagent: file.contains("/subagents/"), file: file.into(),
            start: evs[a].ts.clone(), end: evs[b].ts.clone(), start_event: a, end_event: b,
            signals, user_turn, context: idxs.into_iter().map(|i| ctx(evs, i, &hits)).collect(), counts,
        }
    }).collect()
}

// ------------------------------------------------------------------ facts

/// Bump when [`facts`] changes meaning: every transcript is then re-read
/// once, however unchanged, so the state never mixes two definitions.
pub const FACTS_VERSION: u32 = 1;

/// What one transcript contributes to the struggle-rate measurement: the
/// denominator (tool calls, bucketed by UTC hour so the measurement can
/// close its days in any whole-hour zone) and the Claude Desktop host
/// markers that attribute the session to a host.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Facts {
    /// [`FACTS_VERSION`] these facts were computed under.
    pub version: u32,
    /// `YYYY-MM-DDTHH` (UTC) -> tool calls started in that hour.
    pub tool_calls_by_utc_hour: BTreeMap<String, usize>,
    /// Tool results saying a host hook did not respond ([`HOOK_TIMEOUT_MARKERS`]).
    pub hook_timeouts: usize,
    /// Tool results rejected by the host's worktree guard ([`GUARD_MARKERS`]).
    pub guard_rejections: usize,
}

pub fn facts(evs: &[Ev]) -> Facts {
    let mut f = Facts { version: FACTS_VERSION, ..Facts::default() };
    for e in evs {
        match &e.kind {
            EvKind::ToolUse { .. } if e.ts.len() >= 13 => *f.tool_calls_by_utc_hour.entry(e.ts[..13].to_string()).or_insert(0) += 1,
            EvKind::ToolResult { text, .. } if is_guard_rejection(text) => f.guard_rejections += 1,
            _ => {}
        }
    }
    f.hook_timeouts = detect_hook_timeout(evs).len();
    f
}

// ------------------------------------------------------------------ driver

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileState {
    pub mtime: u64,
    pub size: u64,
    pub hash: String,
    pub since: String,
    pub episodes: usize,
    pub events: usize,
    /// Absent in state files written before the measurement existed; such a
    /// transcript is re-read once (its `version` is then 0).
    #[serde(default)]
    pub facts: Facts,
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
    pub seconds: f64,
}




/// Processes one transcript; returns its new state and writes its episodes.
fn process(root: &Path, out: &Path, rel: &str, since: &str, prev: Option<&FileState>) -> Result<(FileState, bool)> {
    let path = root.join(rel);
    let (mtime, size) = mtime_size(&path)?;
    let outfile = out.join("episodes").join(format!("{}.jsonl", file_key(rel)));
    // The cursor: the previous state stands while the transcript, the
    // --since window and the facts definition are all unchanged.
    let fresh = |p: &FileState| p.since == since && p.facts.version == FACTS_VERSION && outfile.exists();
    if let Some(p) = prev {
        if p.mtime == mtime && p.size == size && fresh(p) {
            return Ok((p.clone(), false));
        }
    }
    let bytes = std::fs::read(&path)?;
    let hash = format!("{:016x}", fnv64(&bytes));
    if let Some(p) = prev {
        if p.hash == hash && fresh(p) {
            return Ok((FileState { mtime, size, ..p.clone() }, false));
        }
    }
    let ses = parse_session(BufReader::new(&bytes[..]));
    let project = rel.split('/').next().unwrap_or_default();
    let eps: Vec<Episode> = episodes(&ses, project, rel).into_iter()
        .filter(|e| since.is_empty() || e.start.as_str() >= since).collect();
    let mut text = String::new();
    for e in &eps {
        text.push_str(&serde_json::to_string(e)?);
        text.push('\n');
    }
    std::fs::write(&outfile, text)?;
    Ok((FileState { mtime, size, hash, since: since.into(), episodes: eps.len(), events: ses.events.len(),
                    facts: facts(&ses.events) }, true))
}

/// Mines every transcript under `root` into `out` (see the module docs).
/// `since` (an ISO date, or empty) drops episodes that start earlier and
/// transcripts last modified before it; `exclude` skips transcripts whose
/// relative path contains any of the given substrings.
pub fn mine_traces(root: &Path, out: &Path, since: &str, jobs: usize, exclude: &[String]) -> Result<Summary> {
    if let Some(tree) = enclosing_work_tree(out) {
        bail!("refusing to write trace episodes to {} inside the git work tree {}: \
               they quote private session transcripts", out.display(), tree.display());
    }
    let t0 = Instant::now();
    std::fs::create_dir_all(out.join("episodes"))?;
    let state_path = out.join("traces-state.json");
    let state: BTreeMap<String, FileState> = match std::fs::read_to_string(&state_path) {
        Ok(t) => serde_json::from_str(&t).context("traces-state.json")?,
        Err(_) => BTreeMap::new(),
    };
    let since_secs = since_epoch(since);
    let all = transcripts(root)?;
    let files: Vec<String> = all.into_iter()
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
    for e in errors.lock().unwrap().iter() {
        eprintln!("[traces] skipped {e}");
    }
    let results = results.into_inner().unwrap();
    // Drop episode files of transcripts that disappeared or were excluded.
    let keep: HashSet<String> = results.keys().map(|r| format!("{}.jsonl", file_key(r))).collect();
    for ent in std::fs::read_dir(out.join("episodes"))? {
        let p = ent?.path();
        if !keep.contains(&p.file_name().unwrap_or_default().to_string_lossy().into_owned()) {
            std::fs::remove_file(p)?;
        }
    }
    let mut sum = Summary { transcripts: files.len(), processed: processed.load(Ordering::SeqCst), ..Default::default() };
    sum.skipped_unchanged = sum.transcripts - sum.processed - errors.lock().unwrap().len();
    let (mut sessions, mut projects, mut hit_sessions) = (HashSet::new(), HashSet::new(), HashSet::new());
    let mut all_text = String::new();
    for rel in results.keys() {
        let text = std::fs::read_to_string(out.join("episodes").join(format!("{}.jsonl", file_key(rel))))?;
        for line in text.lines() {
            let ep: Episode = serde_json::from_str(line)?;
            sum.episodes += 1;
            hit_sessions.insert(ep.session_id.clone());
            for (sig, n) in &ep.signals {
                *sum.episodes_per_signal.entry(sig.name().into()).or_insert(0) += 1;
                *sum.hits_per_signal.entry(sig.name().into()).or_insert(0) += n;
            }
        }
        all_text.push_str(&text);
        projects.insert(rel.split('/').next().unwrap_or_default().to_string());
        // A subagent transcript belongs to its parent session.
        sessions.insert(rel.split('/').take(2).collect::<Vec<_>>().join("/").trim_end_matches(".jsonl").to_string());
    }
    sum.events = results.values().map(|s| s.events).sum();
    sum.sessions = sessions.len();
    sum.projects = projects.len();
    sum.sessions_with_episodes = hit_sessions.len();
    std::fs::write(out.join("episodes.jsonl"), all_text)?;
    std::fs::write(&state_path, serde_json::to_string_pretty(&results)?)?;
    sum.seconds = t0.elapsed().as_secs_f64();
    std::fs::write(out.join("traces-summary.json"), serde_json::to_string_pretty(&sum)?)?;
    Ok(sum)
}



// ------------------------------------------------------------- plugin

/// The traces miner as a plugin: `rrsi-mine traces --out DIR [--root R]
/// [--since DATE] [--jobs N] [--exclude S]...`.
pub struct Traces;

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

impl crate::miner::Miner for Traces {
    fn name(&self) -> &'static str {
        "traces"
    }
    fn about(&self) -> &'static str {
        "Struggle episodes from Claude Code session transcripts (nine named detectors, no LLM)"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("out", "output directory (outside every work tree)"),
          ("root", "transcript root, default ~/.claude/projects"),
          ("since", "only episodes on/after this ISO date"), ("jobs", "parser threads, default 8"),
          ("exclude", "skip transcripts whose path contains this (repeatable)")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("episodes.jsonl", "one struggle episode: session, signals, bounded context, counts"),
          ("traces-summary.json", "counts per signal, sessions, projects"),
          ("traces-state.json", "per transcript: mtime/size/hash for incremental runs, tool calls per UTC hour, \
                                 Desktop host markers (hook timeouts, worktree-guard rejections)")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = crate::miner::parse_args(self.name(), args)?;
        let root = a.root.unwrap_or_else(root_default);
        Ok(serde_json::to_value(mine_traces(&root, &a.out, &a.since, a.jobs, &a.exclude)?)?)
    }
}

#[cfg(test)]
mod tests;
