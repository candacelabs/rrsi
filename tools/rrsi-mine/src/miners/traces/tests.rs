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

//! Every fixture here is synthetic: invented text in the transcript schema,
//! never an excerpt of a real session.

use super::*;
use serde_json::json;

fn ts(n: usize) -> String {
    format!("2026-01-01T{:02}:{:02}:{:02}.000Z", n / 3600, (n / 60) % 60, n % 60)
}

/// Builds a synthetic transcript, one JSON line per call, one second apart.
#[derive(Default)]
struct T {
    lines: Vec<String>,
    sidechain: bool,
    /// Seconds added to every later line's timestamp (see [`T::skip`]).
    clock: usize,
}

impl T {
    fn sub() -> T {
        T { sidechain: true, ..T::default() }
    }
    /// Lets the clock jump ahead by `secs`.
    fn skip(&mut self, secs: usize) -> &mut Self {
        self.clock += secs;
        self
    }
    fn push(&mut self, mut v: Value) -> &mut Self {
        let n = self.lines.len() + self.clock;
        v["timestamp"] = json!(ts(n));
        v["sessionId"] = json!("ses-1");
        v["isSidechain"] = json!(self.sidechain);
        if self.sidechain {
            v["agentId"] = json!("agent-1");
        }
        self.lines.push(v.to_string());
        self
    }
    fn human(&mut self, text: &str) -> &mut Self {
        self.push(json!({"type": "user", "origin": {"kind": "human"}, "message": {"role": "user", "content": text}}))
    }
    fn plain_user(&mut self, text: &str) -> &mut Self {
        self.push(json!({"type": "user", "message": {"role": "user", "content": text}}))
    }
    fn say(&mut self, text: &str) -> &mut Self {
        self.push(json!({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}))
    }
    fn call(&mut self, id: &str, name: &str, input: Value) -> &mut Self {
        self.push(json!({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": id, "name": name, "input": input}]}}))
    }
    fn result(&mut self, id: &str, is_error: bool, text: &str) -> &mut Self {
        self.push(json!({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": id, "is_error": is_error, "content": text}]}}))
    }
    fn denied(&mut self, id: &str, kind: &str, text: &str) -> &mut Self {
        self.push(json!({"type": "user", "toolDenialKind": kind, "message": {"content": [
            {"type": "tool_result", "tool_use_id": id, "is_error": true, "content": text}]}}))
    }
    fn attach(&mut self, a: Value) -> &mut Self {
        self.push(json!({"type": "attachment", "attachment": a}))
    }
    fn session(&self) -> Session {
        parse_session(BufReader::new(self.lines.join("\n").as_bytes()))
    }
    fn hits(&self, s: Signal) -> Vec<usize> {
        s.detect(&self.session().events)
    }
}

#[test]
fn parse_classifies_humans_notifications_and_meta() {
    let mut t = T::default();
    t.human("please add a widget")
        .push(json!({"type": "user", "origin": {"kind": "task-notification"}, "message": {"content": "<task-notification>x</task-notification>"}}))
        .push(json!({"type": "user", "isMeta": true, "message": {"content": "meta text"}}))
        .plain_user("legacy prompt without origin")
        .plain_user("<command-name>/clear</command-name>")
        .human("<system-reminder>ignore me</system-reminder>  real words ")
        .attach(json!({"type": "queued_command", "prompt": "mid-turn note", "origin": {"kind": "human"}, "commandMode": "prompt"}))
        .attach(json!({"type": "queued_command", "prompt": "<task-notification>", "origin": {"kind": "task-notification"}}));
    let humans: Vec<String> = t.session().events.into_iter().filter_map(|e| match e.kind {
        EvKind::Human { text } => Some(text), _ => None }).collect();
    assert_eq!(humans, ["please add a widget", "legacy prompt without origin", "real words", "mid-turn note"]);
}

#[test]
fn a_subagent_brief_is_not_a_human_turn() {
    let mut t = T::sub();
    t.plain_user("You are a subagent. Do the thing.").say("ok");
    let s = t.session();
    assert_eq!(s.agent_id.as_deref(), Some("agent-1"));
    assert!(!s.events.iter().any(|e| matches!(e.kind, EvKind::Human { .. })));
}

#[test]
fn tool_error_excludes_hooks_and_denials() {
    let mut t = T::default();
    t.call("a", "Bash", json!({"command": "false"})).result("a", true, "Exit code 1")
        .call("b", "Write", json!({"file_path": "/x"}))
        .result("b", true, "PreToolUse hook did not respond before its timeout (synthetic).")
        .call("c", "Bash", json!({"command": "rm x"})).denied("c", "user-rejected", "The user doesn't want to proceed with this tool use.")
        .call("d", "Bash", json!({"command": "ls"})).result("d", false, "ok");
    assert_eq!(t.hits(Signal::ToolError), vec![1]);
    assert_eq!(t.hits(Signal::HookTimeout), vec![3]);
    assert_eq!(t.hits(Signal::PermissionDenial), vec![5]);
}

#[test]
fn a_session_cut_off_mid_call_is_neither_error_nor_denial() {
    let mut t = T::default();
    t.call("a", "Bash", json!({})).denied("a", "interrupted", "[Tool call interrupted: synthetic]");
    assert!(t.hits(Signal::ToolError).is_empty());
    assert!(t.hits(Signal::PermissionDenial).is_empty());
}

#[test]
fn hook_no_response_attachment_counts_as_hook_timeout() {
    let mut t = T::default();
    t.say("done").attach(json!({"type": "hook_system_message", "content": "Host didn't respond, so this turn ended without it."}))
        .attach(json!({"type": "hook_system_message", "content": "an unrelated note"}));
    assert_eq!(t.hits(Signal::HookTimeout), vec![1]);
}

#[test]
fn auto_mode_classifier_denial_without_kind_is_a_denial() {
    let mut t = T::default();
    t.call("a", "Bash", json!({})).result("a", true, "Permission for this action was denied by the Claude Code auto mode classifier. Reason: synthetic");
    assert_eq!(t.hits(Signal::PermissionDenial), vec![1]);
}

#[test]
fn interrupts_and_silences() {
    let mut t = T::default();
    t.human("go").call("a", "Bash", json!({})).plain_user("[Request interrupted by user for tool use]")
        .attach(json!({"type": "silent_turn_reminder", "text": "synthetic"}));
    assert_eq!(t.hits(Signal::UserInterrupt), vec![2]);
    assert_eq!(t.hits(Signal::Silence), vec![3]);
}

#[test]
fn correction_words_phrases_and_shouting() {
    for yes in ["no, the other file", "wtf", "why is this here", "stop", "again??", "bruh", "I said the blue one",
                "that's not what i asked", "USE THE BRANCH", "this is NEVER EVER acceptable"] {
        assert!(is_correction(yes), "{yes}");
    }
    for no in ["add a README section for JSON output", "looks good, ship it", "nothing else", "notably fine",
               &"no ".repeat(200)] {
        assert!(!is_correction(no), "{no}");
    }
}

#[test]
fn a_correction_must_follow_an_agent_action() {
    let mut t = T::default();
    t.human("no tests yet; write the parser first")
        .call("a", "Edit", json!({})).result("a", false, "ok").human("no, wrong file");
    assert_eq!(t.hits(Signal::UserCorrection), vec![3]);
}

#[test]
fn reask_needs_similar_words_and_an_answer_in_between() {
    let mut t = T::default();
    t.human("how many widgets render on the dashboard page")
        .human("how many widgets render on the dashboard page today") // no answer yet
        .say("Twelve.")
        .human("so how many widgets render on the dashboard page")
        .human("deploy the staging cluster now please");
    assert_eq!(t.hits(Signal::Reask), vec![3]);
}

#[test]
fn retry_is_a_near_identical_call_after_a_failure() {
    let mut t = T::default();
    t.call("a", "Bash", json!({"command": "cargo test --manifest-path tools/x/Cargo.toml"})).result("a", true, "Exit code 101")
        .call("b", "Bash", json!({"command": "cargo test --manifest-path tools/x/Cargo.toml "})).result("b", false, "ok")
        .call("c", "Bash", json!({"command": "cargo test --manifest-path tools/x/Cargo.toml"})).result("c", false, "ok")
        .call("d", "Read", json!({"command": "cargo test --manifest-path tools/x/Cargo.toml"})).result("d", false, "ok");
    // c repeats b, but b succeeded; d is a different tool.
    assert_eq!(t.hits(Signal::Retry), vec![2]);
}

#[test]
fn test_failure_markers() {
    for yes in ["--- FAIL: TestWidget (0.01s)", "ok  \tx\nFAIL\tgithub.com/x/y\t0.2s", "test result: FAILED. 3 passed; 1 failed",
                "===== 2 failed, 10 passed in 1.2s =====", "FAILED tests/test_x.py::test_y - AssertionError",
                "//pkg:test    FAILED in 2.1s", "build\tfail\t1m2s\thttps://example.invalid/run/1",
                "##[error]Process completed with exit code 1."] {
        assert!(is_test_failure(yes), "{yes}");
    }
    for no in ["PASS\nok  \tx 0.1s", "all tests passed", "no failures", "FAILED_ATTEMPTS = 3"] {
        assert!(!is_test_failure(no), "{no}");
    }
}

#[test]
fn hits_close_together_merge_into_one_episode_with_context() {
    let mut t = T::default();
    t.human("fix the build");
    t.call("a", "Bash", json!({"command": "make"})).result("a", true, "Exit code 2");
    t.call("b", "Bash", json!({"command": "make"})).result("b", true, "Exit code 2");
    for k in 0..10 {
        t.say(&format!("step {k}"));
    }
    t.human("wtf");
    let ses = t.session();
    let eps = episodes(&ses, "proj", "proj/ses-1.jsonl");
    assert_eq!(eps.len(), 2, "{eps:#?}");
    let e = &eps[0];
    assert_eq!((e.start_event, e.end_event), (2, 4));
    assert_eq!(e.signals.get(&Signal::ToolError), Some(&2));
    assert_eq!(e.signals.get(&Signal::Retry), Some(&1));
    assert_eq!(e.user_turn, "fix the build");
    assert_eq!(e.context.first().unwrap().i, 0);
    assert_eq!(e.context[2].tool.as_deref(), Some("Bash"));
    assert_eq!(e.counts.tool_calls, 1);
    assert_eq!(e.counts.tool_errors, 2);
    assert_eq!(eps[1].signals.keys().copied().collect::<Vec<_>>(), vec![Signal::UserCorrection]);
    assert_ne!(eps[0].id, eps[1].id);
}

#[test]
fn long_windows_keep_head_and_tail_and_text_is_truncated() {
    let mut t = T::default();
    for k in 0..40 {
        let id = format!("t{k}");
        t.call(&id, "Bash", json!({"n": k})).result(&id, true, &"x".repeat(2000));
    }
    let eps = episodes(&t.session(), "p", "p/s.jsonl");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].context.len(), CONTEXT_MAX);
    assert!(eps[0].context.iter().all(|c| c.text.chars().count() < TEXT_MAX + 40));
}

#[test]
fn strip_reminders_handles_unclosed_blocks() {
    assert_eq!(strip_reminders("a <system-reminder>x</system-reminder> b"), "a  b");
    assert_eq!(strip_reminders("a <system-reminder>x"), "a");
}

#[test]
fn facts_count_tool_calls_per_utc_hour_and_desktop_host_markers() {
    let mut t = T::default();
    t.human("go").call("a", "Bash", json!({"command": "ls"})).result("a", false, "ok")
        .call("b", "Bash", json!({"command": "cd x && make"}))
        .result("b", true, "This agent is isolated in the worktree /w (synthetic); the command was rejected.")
        .skip(3600)
        .call("c", "Write", json!({"file_path": "/x"}))
        .result("c", true, "PreToolUse hook did not respond before its timeout (synthetic).")
        .call("d", "Edit", json!({})).denied("d", "interrupted", "[Tool call interrupted: synthetic]");
    let f = facts(&t.session().events);
    assert_eq!(f.version, FACTS_VERSION);
    assert_eq!(f.tool_calls_by_utc_hour, [("2026-01-01T00".to_string(), 2), ("2026-01-01T01".to_string(), 2)].into());
    assert_eq!((f.hook_timeouts, f.guard_rejections), (1, 1));
    // A transcript with no tool calls or markers contributes nothing but its version.
    let mut quiet = T::default();
    quiet.human("hi").say("hello");
    assert_eq!(facts(&quiet.session().events), Facts { version: FACTS_VERSION, ..Facts::default() });
}

#[test]
fn a_state_written_under_an_older_facts_version_is_reprocessed_once() {
    let root = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    write_corpus(root.path());
    mine_traces(root.path(), out.path(), "", 2, &[]).unwrap();
    let state_path = out.path().join("traces-state.json");
    let mut state: BTreeMap<String, FileState> = serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    assert!(state.values().all(|s| s.facts.version == FACTS_VERSION));
    assert_eq!(state["proj/ses-1.jsonl"].facts.tool_calls_by_utc_hour.values().sum::<usize>(), 1);
    // Older state: the facts field is missing altogether (serde default, version 0).
    let mut old: serde_json::Value = serde_json::to_value(&state).unwrap();
    old["proj/ses-1.jsonl"].as_object_mut().unwrap().remove("facts");
    std::fs::write(&state_path, serde_json::to_string(&old).unwrap()).unwrap();
    let s = mine_traces(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((s.processed, s.skipped_unchanged), (1, 1));
    state = serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    assert_eq!(state["proj/ses-1.jsonl"].facts.version, FACTS_VERSION);
    let s = mine_traces(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((s.processed, s.skipped_unchanged), (0, 2));
}

#[test]
fn since_epoch_is_utc_midnight() {
    assert_eq!(since_epoch("1970-01-02"), Some(86400));
    assert_eq!(since_epoch("2026-10-01"), Some(1790812800));
    assert_eq!(since_epoch("bad"), None);
}

fn write_corpus(root: &Path) {
    let mut a = T::default();
    a.human("do it").call("a", "Bash", json!({})).result("a", true, "Exit code 1");
    let mut sub = T::sub();
    sub.plain_user("brief").call("b", "Bash", json!({})).result("b", true, "--- FAIL: TestX");
    std::fs::create_dir_all(root.join("proj/ses-1/subagents")).unwrap();
    std::fs::write(root.join("proj/ses-1.jsonl"), a.lines.join("\n")).unwrap();
    std::fs::write(root.join("proj/ses-1/subagents/agent-1.jsonl"), sub.lines.join("\n")).unwrap();
}

#[test]
fn mine_traces_is_incremental_and_counts_subagents_with_their_session() {
    let root = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    write_corpus(root.path());
    let s1 = mine_traces(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((s1.transcripts, s1.processed, s1.skipped_unchanged), (2, 2, 0));
    assert_eq!((s1.sessions, s1.projects, s1.episodes), (1, 1, 2));
    assert_eq!(s1.episodes_per_signal.get("test_failure"), Some(&1));
    assert_eq!(std::fs::read_to_string(out.path().join("episodes.jsonl")).unwrap().lines().count(), 2);
    let s2 = mine_traces(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((s2.processed, s2.skipped_unchanged, s2.episodes), (0, 2, 2));
    // Excluding a transcript drops its episodes.
    let s3 = mine_traces(root.path(), out.path(), "", 2, &["subagents".into()]).unwrap();
    assert_eq!((s3.transcripts, s3.episodes), (1, 1));
    // A --since after every event yields no episodes.
    let s4 = mine_traces(root.path(), out.path(), "2026-01-02", 1, &[]).unwrap();
    assert_eq!(s4.episodes, 0);
}

#[test]
fn output_inside_a_work_tree_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let repo = tempfile::tempdir().unwrap();
    std::fs::create_dir(repo.path().join(".git")).unwrap();
    let inside = repo.path().join("private/harness");
    let err = mine_traces(root.path(), &inside, "", 1, &[]).unwrap_err();
    assert!(format!("{err}").contains("inside the git work tree"));
    assert!(!inside.exists());
}
