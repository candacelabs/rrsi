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

//! GitHub Copilot CLI sessions as Claude Code-shaped transcripts.
//!
//! ```text
//! rrsi-mine copilot-transcripts --out DIR [--root ~/.copilot/session-state] [--since 2026-09-01] [--jobs 8]
//! ```
//!
//! The Copilot CLI writes one `<root>/<session>/events.jsonl` per session:
//! one `{type, data, id, timestamp, parentId}` object per line, with every
//! subagent's activity interleaved in its parent's file and keyed by
//! `parentToolCallId`. The transcript miners (`traces`, `handoffs`, `pr-gap`)
//! and `python -m rrsi harness measure` read Claude Code's layout. This
//! plugin projects each Copilot session onto that layout once, so the
//! mapping lives in one place and the miners stay as they are; point their
//! `--root` at `DIR`.
//!
//! - `DIR/copilot-<cwd-slug>/<session>.jsonl` — the main agent
//! - `DIR/copilot-<cwd-slug>/<session>/subagents/agent-<toolCallId>.jsonl` —
//!   one subagent run: a sidechain whose first line is its brief
//! - `DIR/copilot-sessions.ndjson` — per session: repository, cwd, client,
//!   version, tool calls and human turns per UTC hour (counts, no text; not
//!   `.jsonl`, so no miner mistakes it for a transcript)
//! - `DIR/copilot-state.json` — per source: mtime, size and outputs; an
//!   unchanged session is skipped on the next run, a vanished one removed
//! - `DIR/copilot-summary.json` — counts
//!
//! | Copilot event | Claude Code line |
//! |---|---|
//! | `user.message`, `source` `user` | `user`, `origin.kind` `human` |
//! | `user.message` without `source` (older CLIs) | `user` without origin: the parsers' plain-prompt rule |
//! | `user.message`, `source` `agent-<id>` | `user`, `origin.kind` `peer` from `<id>` |
//! | `user.message`, `source` `schedule-*` / any other | `user`, `origin.kind` `scheduled` / `system`: not human |
//! | `assistant.message` text | `assistant` text block |
//! | `tool.execution_start` | `assistant` `tool_use`, named and shaped as the Claude Code tool the miners read ([`map_tool`]) |
//! | `tool.execution_complete` | `user` `tool_result`; `is_error` is `!success`, or a `bash` result ending in a non-zero exit code, as Claude Code marks it |
//! | error code `denied` / `rejected` | `toolDenialKind`; the user's rejection feedback becomes a human turn |
//! | error `Cancelled` / `Agent is cancelled` | `toolDenialKind` `interrupted` |
//! | `abort` by the user | `[Request interrupted by user]` |
//! | `session.start` / `session.resume` / `session.context_changed` | `cwd` and `gitBranch` of the lines after it |

use crate::transcript::{content_text, field, since_epoch};
pub use crate::transcript::mtime_size;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, UNIX_EPOCH};

pub const SESSIONS_FILE: &str = "copilot-sessions.ndjson";
pub const STATE_FILE: &str = "copilot-state.json";
pub const SUMMARY_FILE: &str = "copilot-summary.json";
pub const PROJECT_PREFIX: &str = "copilot";
pub const INTERRUPT_TEXT: &str = "[Request interrupted by user]";
/// What follows this in a rejection's error message is the user's own text.
pub const REJECTION_FEEDBACK: &str = "User feedback:";
/// Tool errors that mean the user stopped the call, not that it failed.
pub const CANCELLED: [&str; 2] = ["Cancelled", "Agent is cancelled"];
/// `abort` reasons that are the user's.
pub const USER_ABORTS: [&str; 3] = ["user_initiated", "user initiated", "user_abort"];
/// The event types the projection reads; every other line is skipped unparsed.
const NEEDED: [&str; 9] = ["session.start", "session.resume", "session.context_changed", "user.message",
    "assistant.message", "tool.execution_start", "tool.execution_complete", "abort", "hook.end"];

/// Claude Code names a project directory after its cwd with every character
/// that is not ASCII alphanumeric replaced by `-`; a Copilot session gets the
/// same name behind the `copilot` prefix.
pub fn project_of(cwd: &str) -> String {
    if cwd.is_empty() {
        return format!("{PROJECT_PREFIX}-unknown");
    }
    let slug: String = cwd.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    if slug.starts_with('-') { format!("{PROJECT_PREFIX}{slug}") } else { format!("{PROJECT_PREFIX}-{slug}") }
}

/// The exit code a `bash` result ends with: `<shellId: N completed with exit
/// code C>`, or the older `<exited with exit code C>`.
pub fn exit_code(text: &str) -> Option<i64> {
    let last = text.trim_end().lines().last()?.trim();
    let inner = last.strip_prefix('<')?.strip_suffix('>')?;
    let marked = inner.starts_with("exited with exit code ")
        || (inner.starts_with("shellId: ") && inner.contains(" completed with exit code "));
    if !marked {
        return None;
    }
    inner.rsplit_once("exit code ")?.1.trim().parse().ok()
}

/// Files named by an `apply_patch` body (`*** Update|Add|Delete File: <path>`).
pub fn patch_files(patch: &str) -> Vec<String> {
    patch.lines().filter_map(|l| {
        ["*** Update File: ", "*** Add File: ", "*** Delete File: "].iter()
            .find_map(|p| l.strip_prefix(p)).map(|f| f.trim().to_string())
    }).collect()
}

/// A Copilot tool call as the Claude Code tool the miners read: the name
/// (`bash` -> `Bash`, `view` -> `Read`, `edit`/`apply_patch` -> `Edit`,
/// `create` -> `Write`, `task` -> `Agent`, `write_agent` -> `SendMessage`)
/// and the input fields that tool carries (`file_path`, `old_string`, ...),
/// added next to the Copilot fields. Other tools keep their name; an input
/// that is not an object is wrapped as `{"input": ...}`.
pub fn map_tool(name: &str, args: &Value) -> (String, Value) {
    let mut m = match args {
        Value::Object(m) => m.clone(),
        Value::Null => Map::new(),
        v => Map::from_iter([("input".to_string(), v.clone())]),
    };
    let copy = |m: &mut Map<String, Value>, from: &str, to: &str| {
        if let Some(v) = m.get(from).cloned() {
            m.insert(to.into(), v);
        }
    };
    let claude = match name {
        "bash" => "Bash",
        "view" => {
            copy(&mut m, "path", "file_path");
            "Read"
        }
        "edit" => {
            copy(&mut m, "path", "file_path");
            copy(&mut m, "old_str", "old_string");
            copy(&mut m, "new_str", "new_string");
            "Edit"
        }
        "create" => {
            copy(&mut m, "path", "file_path");
            copy(&mut m, "file_text", "content");
            "Write"
        }
        "task" => {
            copy(&mut m, "agent_type", "subagent_type");
            "Agent"
        }
        "write_agent" => {
            copy(&mut m, "agent_ids", "to");
            copy(&mut m, "agent_id", "to");
            "SendMessage"
        }
        "apply_patch" => {
            let files = patch_files(args.as_str().unwrap_or_default());
            if let Some(first) = files.first() {
                m.insert("file_path".into(), json!(first));
            }
            m.insert("files".into(), json!(files));
            "Edit"
        }
        other => other,
    };
    (claude.to_string(), Value::Object(m))
}

/// One session as counts (no text): the measurement's covariates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    /// The transcript directory: [`project_of`] the first context's cwd.
    pub project: String,
    pub repository: String,
    pub cwd: String,
    pub branch: String,
    /// `client_name` from the session's `workspace.yaml` (`github/cli`, `sdk`, ...).
    pub client: String,
    pub copilot_version: String,
    pub start: String,
    pub end: String,
    /// `YYYY-MM-DDTHH` (UTC) -> tool calls started, subagents included.
    pub tool_calls_by_utc_hour: BTreeMap<String, usize>,
    /// `YYYY-MM-DDTHH` (UTC) -> turns the miners read as the operator's.
    pub human_turns_by_utc_hour: BTreeMap<String, usize>,
    /// User messages by kind: human, plain, plain_meta, peer, scheduled, system.
    pub user_messages: BTreeMap<String, usize>,
    pub subagent_runs: usize,
    pub tool_errors: usize,
    pub denials: usize,
    pub aborts: usize,
    /// Copilot hooks that failed (`hook.end` with `success: false`).
    pub hook_failures: usize,
    pub models: Vec<String>,
}

/// One converted session: JSON lines of the main agent and of each subagent run.
#[derive(Clone, Debug, Default)]
pub struct Converted {
    pub record: SessionRecord,
    pub main: Vec<String>,
    /// `parentToolCallId` -> the run's lines (brief first).
    pub subagents: BTreeMap<String, Vec<String>>,
}

/// The event type, read from the line's head without parsing it (Copilot
/// writes `type` first); `None` when it is not there.
fn head_type(line: &str) -> Option<&str> {
    let head = line.get(..line.len().min(48))?;
    let at = head.find("\"type\":\"")? + 8;
    let rest = &line[at..];
    Some(&rest[..rest.find('"')?])
}

fn hour(ts: &str) -> Option<String> {
    (ts.len() >= 13).then(|| ts[..13].to_string())
}

/// The origin a user message gets and the kind it is counted as.
fn origin(source: &str, text: &str) -> (Option<Value>, &'static str) {
    if source.is_empty() {
        let meta = text.trim_start().starts_with(['<', '[']);
        return (None, if meta { "plain_meta" } else { "plain" });
    }
    if source == "user" {
        return (Some(json!({"kind": "human"})), "human");
    }
    if let Some(peer) = source.strip_prefix("agent-") {
        return (Some(json!({"kind": "peer", "fromSession": peer, "name": peer, "body": text})), "peer");
    }
    if source.starts_with("schedule") {
        return (Some(json!({"kind": "scheduled", "source": source})), "scheduled");
    }
    (Some(json!({"kind": "system", "source": source})), "system")
}

struct Builder {
    sid: String,
    cwd: String,
    branch: String,
    version: String,
    main: Vec<String>,
    subs: BTreeMap<String, Vec<String>>,
    /// `task` call id -> its prompt (the subagent's brief).
    briefs: HashMap<String, String>,
    /// tool call id -> Copilot tool name.
    tools: HashMap<String, String>,
    rec: SessionRecord,
}

impl Builder {
    fn line(&self, kind: &str, mut v: Value, ts: &str, e: &Value, agent: Option<&str>) -> String {
        v["type"] = json!(kind);
        v["timestamp"] = json!(ts);
        v["sessionId"] = json!(self.sid);
        v["uuid"] = json!(field(e, "id"));
        v["parentUuid"] = e.get("parentId").cloned().unwrap_or(Value::Null);
        v["cwd"] = json!(self.cwd);
        v["gitBranch"] = json!(self.branch);
        v["version"] = json!(format!("copilot-cli/{}", self.version));
        v["isSidechain"] = json!(agent.is_some());
        if let Some(a) = agent {
            v["agentId"] = json!(a);
        }
        v.to_string()
    }

    /// Appends a line to the main transcript or to the subagent run `agent`,
    /// opening the run with its brief.
    fn push(&mut self, agent: Option<&str>, kind: &str, v: Value, ts: &str, e: &Value) {
        let line = self.line(kind, v, ts, e, agent);
        let Some(a) = agent else {
            self.main.push(line);
            return;
        };
        if !self.subs.contains_key(a) {
            let brief = self.briefs.get(a).cloned().unwrap_or_default();
            let first = self.line("user", json!({"message": {"role": "user", "content": brief}}), ts, e, Some(a));
            self.subs.insert(a.to_string(), vec![first]);
        }
        self.subs.get_mut(a).expect("opened above").push(line);
    }

    fn context(&mut self, c: &Value) {
        let cwd = field(c, "cwd");
        if !cwd.is_empty() {
            self.cwd = cwd;
        }
        let branch = field(c, "branch");
        if !branch.is_empty() {
            self.branch = branch;
        }
        if self.rec.cwd.is_empty() {
            self.rec.cwd = self.cwd.clone();
            self.rec.branch = self.branch.clone();
        }
        let repo = field(c, "repository");
        if self.rec.repository.is_empty() && !repo.is_empty() {
            self.rec.repository = repo;
        }
    }

    fn human_turn(&mut self, ts: &str) {
        if let Some(h) = hour(ts) {
            *self.rec.human_turns_by_utc_hour.entry(h).or_insert(0) += 1;
        }
    }

    fn event(&mut self, ty: &str, e: &Value) {
        let ts = field(e, "timestamp");
        let d = e.get("data").cloned().unwrap_or(Value::Null);
        let agent = d.get("parentToolCallId").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let agent = agent.as_deref();
        match ty {
            "session.start" | "session.resume" => {
                if let Some(c) = d.get("context") {
                    self.context(c);
                }
                let v = field(&d, "copilotVersion");
                if !v.is_empty() {
                    self.version = v.clone();
                    self.rec.copilot_version = v;
                }
                if self.rec.start.is_empty() {
                    self.rec.start = ts.clone();
                }
            }
            "session.context_changed" => self.context(&d),
            "user.message" => {
                let text = field(&d, "content");
                let (o, kind) = origin(&field(&d, "source"), &text);
                *self.rec.user_messages.entry(kind.to_string()).or_insert(0) += 1;
                if matches!(kind, "human" | "plain") && !text.trim().is_empty() {
                    self.human_turn(&ts);
                }
                let mut v = json!({"message": {"role": "user", "content": text}});
                if let Some(o) = o {
                    v["origin"] = o;
                }
                self.push(None, "user", v, &ts, e);
            }
            "assistant.message" => {
                let model = field(&d, "model");
                if !model.is_empty() && !self.rec.models.contains(&model) {
                    self.rec.models.push(model);
                }
                let text = field(&d, "content");
                if !text.trim().is_empty() {
                    let v = json!({"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}});
                    self.push(agent, "assistant", v, &ts, e);
                }
            }
            "tool.execution_start" => {
                let (id, name) = (field(&d, "toolCallId"), field(&d, "toolName"));
                let args = d.get("arguments").cloned().unwrap_or(Value::Null);
                if name == "task" {
                    self.briefs.insert(id.clone(), field(&args, "prompt"));
                }
                self.tools.insert(id.clone(), name.clone());
                if let Some(h) = hour(&ts) {
                    *self.rec.tool_calls_by_utc_hour.entry(h).or_insert(0) += 1;
                }
                let (claude, input) = map_tool(&name, &args);
                let v = json!({"message": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": id, "name": claude, "input": input}]}, "copilotTool": name});
                self.push(agent, "assistant", v, &ts, e);
            }
            "tool.execution_complete" => self.complete(&d, agent, &ts, e),
            "abort" => {
                if USER_ABORTS.contains(&field(&d, "reason").as_str()) {
                    self.rec.aborts += 1;
                    self.push(None, "user", json!({"message": {"role": "user", "content": INTERRUPT_TEXT}}), &ts, e);
                }
            }
            "hook.end" => {
                if d.get("success").and_then(Value::as_bool) == Some(false) {
                    self.rec.hook_failures += 1;
                }
            }
            _ => {}
        }
    }

    fn complete(&mut self, d: &Value, agent: Option<&str>, ts: &str, e: &Value) {
        let id = field(d, "toolCallId");
        let success = d.get("success").and_then(Value::as_bool).unwrap_or(true);
        let result = content_text(d.get("result").and_then(|r| r.get("content")).unwrap_or(&Value::Null));
        let err = d.get("error").cloned().unwrap_or(Value::Null);
        let (code, msg) = (field(&err, "code"), field(&err, "message"));
        let text = match (success, result.is_empty()) {
            (true, _) => result,
            (false, true) => msg.clone(),
            (false, false) => format!("{msg}\n{result}"),
        };
        let mut denial = None;
        let mut feedback = None;
        if !success {
            if code == "denied" || code == "rejected" {
                denial = Some(code.clone());
                feedback = msg.find(REJECTION_FEEDBACK).map(|i| msg[i + REJECTION_FEEDBACK.len()..].trim().to_string());
            } else if CANCELLED.contains(&msg.trim()) {
                denial = Some("interrupted".to_string());
            }
        }
        let bash = self.tools.get(&id).is_some_and(|n| n == "bash");
        let is_error = !success || (bash && exit_code(&text).is_some_and(|c| c != 0));
        match denial.as_deref() {
            Some("denied" | "rejected") => self.rec.denials += 1,
            None if is_error => self.rec.tool_errors += 1,
            _ => {}
        }
        let mut v = json!({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": id, "is_error": is_error, "content": text}]}});
        if let Some(k) = denial {
            v["toolDenialKind"] = json!(k);
        }
        self.push(agent, "user", v, ts, e);
        if let Some(f) = feedback.filter(|f| !f.is_empty()) {
            self.human_turn(ts);
            *self.rec.user_messages.entry("human".into()).or_insert(0) += 1;
            let v = json!({"origin": {"kind": "human"}, "message": {"role": "user", "content": f}});
            self.push(agent, "user", v, ts, e);
        }
    }
}

/// Projects one session's `events.jsonl` (malformed lines are skipped).
pub fn convert<R: BufRead>(session_id: &str, client: &str, r: R) -> Converted {
    let mut b = Builder {
        sid: session_id.to_string(), cwd: String::new(), branch: String::new(), version: String::new(),
        main: vec![], subs: BTreeMap::new(), briefs: HashMap::new(), tools: HashMap::new(),
        rec: SessionRecord { session_id: session_id.to_string(), client: client.to_string(), ..Default::default() },
    };
    let mut last = String::new();
    for line in r.lines() {
        let Ok(line) = line else { continue };
        if let Some(t) = head_type(&line) {
            if !NEEDED.contains(&t) || (t == "hook.end" && !line.contains("\"success\":false")) {
                continue;
            }
        }
        let Ok(e) = serde_json::from_str::<Value>(&line) else { continue };
        let ty = field(&e, "type");
        if !NEEDED.contains(&ty.as_str()) {
            continue;
        }
        let ts = field(&e, "timestamp");
        if ts > last {
            last = ts;
        }
        b.event(&ty, &e);
    }
    let mut rec = b.rec;
    rec.project = project_of(&rec.cwd);
    rec.end = last;
    rec.subagent_runs = b.subs.len();
    Converted { record: rec, main: b.main, subagents: b.subs }
}

// ------------------------------------------------------------------ driver

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileState {
    pub mtime: u64,
    pub size: u64,
    /// Transcripts written for this session, relative to `out`.
    pub outputs: Vec<String>,
    pub record: SessionRecord,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub sessions: usize,
    pub processed: usize,
    pub skipped_unchanged: usize,
    pub skipped_since: usize,
    /// Sessions whose source disappeared; their transcripts were removed.
    pub removed: usize,
    pub main_transcripts: usize,
    pub subagent_transcripts: usize,
    pub tool_calls: usize,
    pub human_turns: usize,
    pub seconds: f64,
}

pub fn root_default() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".copilot/session-state")
}

/// `client_name` from a session's flat `workspace.yaml`; empty when absent.
fn client_of(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("workspace.yaml")).unwrap_or_default().lines()
        .find_map(|l| l.strip_prefix("client_name:")).map(|v| v.trim().to_string()).unwrap_or_default()
}

/// A tool call id as a file name component.
fn safe(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Writes `lines` to `out/rel` atomically, with the source's mtime.
fn write_lines(out: &Path, rel: &str, lines: &[String], mtime: u64) -> Result<()> {
    let path = out.join(rel);
    std::fs::create_dir_all(path.parent().context("transcript path has a parent")?)?;
    let tmp = path.with_extension("jsonl.part");
    let mut text = String::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
    for l in lines {
        text.push_str(l);
        text.push('\n');
    }
    std::fs::write(&tmp, text)?;
    std::fs::File::options().write(true).open(&tmp)?.set_modified(UNIX_EPOCH + Duration::from_secs(mtime))?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Every `<root>/<session>/events.jsonl`, by session id.
fn sources(root: &Path) -> Result<Vec<String>> {
    let mut out = vec![];
    for ent in std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let p = ent?.path();
        if p.join("events.jsonl").is_file() {
            out.push(p.file_name().unwrap_or_default().to_string_lossy().into_owned());
        }
    }
    out.sort();
    Ok(out)
}

fn process(root: &Path, out: &Path, sid: &str, prev: Option<&FileState>) -> Result<(FileState, bool)> {
    let src = root.join(sid).join("events.jsonl");
    let (mtime, size) = mtime_size(&src)?;
    if let Some(p) = prev {
        if p.mtime == mtime && p.size == size && p.outputs.iter().all(|o| out.join(o).is_file()) {
            return Ok((p.clone(), false));
        }
    }
    let conv = convert(sid, &client_of(&root.join(sid)), BufReader::new(std::fs::File::open(&src)?));
    let project = &conv.record.project;
    let mut outputs = vec![format!("{project}/{sid}.jsonl")];
    write_lines(out, &outputs[0], &conv.main, mtime)?;
    for (agent, lines) in &conv.subagents {
        let rel = format!("{project}/{sid}/subagents/agent-{}.jsonl", safe(agent));
        write_lines(out, &rel, lines, mtime)?;
        outputs.push(rel);
    }
    for o in prev.map(|p| p.outputs.as_slice()).unwrap_or_default() {
        if !outputs.contains(o) {
            let _ = std::fs::remove_file(out.join(o));
        }
    }
    Ok((FileState { mtime, size, outputs, record: conv.record }, true))
}

/// Converts every session under `root` into `out` (see the module docs).
/// `since` (an ISO date, or empty) skips sessions last written before it;
/// their earlier transcripts stay.
pub fn convert_root(root: &Path, out: &Path, since: &str, jobs: usize) -> Result<Summary> {
    if let Some(tree) = crate::enclosing_work_tree(out) {
        bail!("refusing to write Copilot transcripts to {} inside the git work tree {}: \
               they quote private sessions", out.display(), tree.display());
    }
    let t0 = Instant::now();
    std::fs::create_dir_all(out)?;
    let state_path = out.join(STATE_FILE);
    let state: BTreeMap<String, FileState> = match std::fs::read_to_string(&state_path) {
        Ok(t) => serde_json::from_str(&t).context(STATE_FILE)?,
        Err(_) => BTreeMap::new(),
    };
    let all = sources(root)?;
    let since_secs = since_epoch(since);
    let mut sum = Summary::default();
    let mut results: BTreeMap<String, FileState> = BTreeMap::new();
    let mut todo = vec![];
    for sid in &all {
        let fresh = since_secs.is_none_or(|s| mtime_size(&root.join(sid).join("events.jsonl")).map(|(m, _)| m >= s).unwrap_or(true));
        match (fresh, state.get(sid)) {
            (true, _) => todo.push(sid.clone()),
            (false, Some(prev)) => {
                sum.skipped_since += 1;
                results.insert(sid.clone(), prev.clone());
            }
            (false, None) => sum.skipped_since += 1,
        }
    }
    let next = AtomicUsize::new(0);
    let processed = AtomicUsize::new(0);
    let done: Mutex<BTreeMap<String, FileState>> = Mutex::new(BTreeMap::new());
    let errors: Mutex<Vec<String>> = Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| {
                while let Some(sid) = todo.get(next.fetch_add(1, Ordering::SeqCst)) {
                    match process(root, out, sid, state.get(sid)) {
                        Ok((st, did)) => {
                            if did {
                                processed.fetch_add(1, Ordering::SeqCst);
                            }
                            done.lock().unwrap().insert(sid.clone(), st);
                        }
                        Err(e) => errors.lock().unwrap().push(format!("{sid}: {e:#}")),
                    }
                }
            });
        }
    });
    for e in errors.lock().unwrap().iter() {
        eprintln!("[copilot-transcripts] skipped {e}");
    }
    results.extend(done.into_inner().unwrap());
    for (sid, prev) in &state {
        if !all.contains(sid) {
            sum.removed += 1;
            for o in &prev.outputs {
                let _ = std::fs::remove_file(out.join(o));
            }
            let _ = std::fs::remove_dir_all(out.join(&prev.record.project).join(sid));
        }
    }
    sum.processed = processed.load(Ordering::SeqCst);
    sum.skipped_unchanged = todo.len() - sum.processed - errors.lock().unwrap().len();
    sum.sessions = results.len();
    let mut index = String::new();
    for st in results.values() {
        sum.main_transcripts += 1;
        sum.subagent_transcripts += st.outputs.len() - 1;
        sum.tool_calls += st.record.tool_calls_by_utc_hour.values().sum::<usize>();
        sum.human_turns += st.record.human_turns_by_utc_hour.values().sum::<usize>();
        index.push_str(&serde_json::to_string(&st.record)?);
        index.push('\n');
    }
    std::fs::write(out.join(SESSIONS_FILE), index)?;
    std::fs::write(&state_path, serde_json::to_string(&results)?)?;
    sum.seconds = t0.elapsed().as_secs_f64();
    std::fs::write(out.join(SUMMARY_FILE), serde_json::to_string_pretty(&sum)?)?;
    Ok(sum)
}

// ------------------------------------------------------------- plugin

/// The converter as a plugin: `rrsi-mine copilot-transcripts --out DIR
/// [--root R] [--since DATE] [--jobs N]`.
pub struct CopilotTranscripts;

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
}

fn default_jobs() -> usize {
    8
}

impl crate::miner::Miner for CopilotTranscripts {
    fn name(&self) -> &'static str {
        "copilot-transcripts"
    }
    fn about(&self) -> &'static str {
        "GitHub Copilot CLI sessions as Claude Code-shaped transcripts, for traces/handoffs/pr-gap and measure (no LLM)"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("out", "output directory (outside every work tree); the miners' --root"),
          ("root", "Copilot session state, default ~/.copilot/session-state"),
          ("since", "skip sessions last written before this ISO date"), ("jobs", "converter threads, default 8")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("<project>/<session>.jsonl", "the main agent as a Claude Code transcript"),
          ("<project>/<session>/subagents/agent-<id>.jsonl", "one subagent run, brief first"),
          (SESSIONS_FILE, "per session: repository, cwd, client, version, tool calls and human turns per UTC hour"),
          (STATE_FILE, "per source: mtime/size/outputs for incremental runs"),
          (SUMMARY_FILE, "counts")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = crate::miner::parse_args(self.name(), args)?;
        let root = a.root.unwrap_or_else(root_default);
        Ok(serde_json::to_value(convert_root(&root, &a.out, &a.since, a.jobs)?)?)
    }
}

#[cfg(test)]
mod tests;
