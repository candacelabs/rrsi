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

//! Every fixture here is synthetic: invented text in the Copilot CLI event
//! schema, never an excerpt of a real session.

use super::*;
use crate::miners::{handoffs, pr_gap, traces};
use serde_json::json;
use std::io::BufReader;

/// Builds a synthetic Copilot `events.jsonl`, one event per call, one second apart.
#[derive(Default)]
struct C {
    lines: Vec<String>,
}

impl C {
    fn ev(&mut self, ty: &str, data: Value) -> &mut Self {
        let n = self.lines.len();
        let ts = format!("2026-01-01T{:02}:{:02}:{:02}.000Z", n / 3600, (n / 60) % 60, n % 60);
        self.lines.push(json!({"type": ty, "data": data, "id": format!("e{n}"), "timestamp": ts,
                               "parentId": null}).to_string());
        self
    }
    fn start(&mut self, cwd: &str) -> &mut Self {
        self.ev("session.start", json!({"sessionId": "s-1", "copilotVersion": "9.9.9", "producer": "copilot-agent",
            "context": {"cwd": cwd, "branch": "feature-x", "repository": "acme/widgets"}}))
    }
    fn user(&mut self, source: Option<&str>, text: &str) -> &mut Self {
        let mut d = json!({"content": text, "transformedContent": format!("<ctx>now</ctx>{text}")});
        if let Some(s) = source {
            d["source"] = json!(s);
        }
        self.ev("user.message", d)
    }
    fn say(&mut self, parent: Option<&str>, text: &str) -> &mut Self {
        let mut d = json!({"content": text, "model": "model-a", "toolRequests": []});
        if let Some(p) = parent {
            d["parentToolCallId"] = json!(p);
        }
        self.ev("assistant.message", d)
    }
    fn call(&mut self, parent: Option<&str>, id: &str, tool: &str, args: Value) -> &mut Self {
        let mut d = json!({"toolCallId": id, "toolName": tool, "arguments": args});
        if let Some(p) = parent {
            d["parentToolCallId"] = json!(p);
        }
        self.ev("tool.execution_start", d)
    }
    fn ok(&mut self, parent: Option<&str>, id: &str, text: &str) -> &mut Self {
        let mut d = json!({"toolCallId": id, "success": true, "result": {"content": text, "detailedContent": text}});
        if let Some(p) = parent {
            d["parentToolCallId"] = json!(p);
        }
        self.ev("tool.execution_complete", d)
    }
    fn fail(&mut self, parent: Option<&str>, id: &str, code: &str, msg: &str) -> &mut Self {
        let mut d = json!({"toolCallId": id, "success": false, "error": {"code": code, "message": msg}});
        if let Some(p) = parent {
            d["parentToolCallId"] = json!(p);
        }
        self.ev("tool.execution_complete", d)
    }
    fn text(&self) -> String {
        self.lines.join("\n") + "\n"
    }
    fn convert(&self) -> Converted {
        convert("s-1", "github/cli", BufReader::new(self.text().as_bytes()))
    }
}

fn parse(lines: &[String]) -> Vec<Value> {
    lines.iter().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn joined(lines: &[String]) -> String {
    lines.join("\n") + "\n"
}

#[test]
fn project_is_the_claude_code_cwd_slug_behind_a_copilot_prefix() {
    assert_eq!(project_of("/home/u_x/work.d/plat_sim"), "copilot-home-u-x-work-d-plat-sim");
    assert_eq!(project_of(""), "copilot-unknown");
}

#[test]
fn exit_code_reads_both_marker_forms_only_at_the_end() {
    assert_eq!(exit_code("out\n<shellId: 7 completed with exit code 101>"), Some(101));
    assert_eq!(exit_code("out\n<exited with exit code 0>\n"), Some(0));
    assert_eq!(exit_code("<exited with exit code 2>\nmore output after"), None);
    assert_eq!(exit_code("plain output"), None);
    assert_eq!(exit_code(""), None);
}

#[test]
fn tools_map_to_the_claude_tools_and_fields_the_miners_read() {
    let patch = json!("*** Begin Patch\n*** Update File: /w/c.rs\n@@\n-a\n+b\n*** Add File: /w/d.rs\n+x\n*** End Patch");
    let cases: Vec<(&str, Value, &str, &str, Value)> = vec![
        ("bash", json!({"command": "ls", "description": "d"}), "Bash", "command", json!("ls")),
        ("view", json!({"path": "/w/a.rs"}), "Read", "file_path", json!("/w/a.rs")),
        ("edit", json!({"path": "/w/a.rs", "old_str": "x", "new_str": "y"}), "Edit", "file_path", json!("/w/a.rs")),
        ("edit", json!({"path": "/w/a.rs", "old_str": "x", "new_str": "y"}), "Edit", "old_string", json!("x")),
        ("edit", json!({"path": "/w/a.rs", "old_str": "x", "new_str": "y"}), "Edit", "new_string", json!("y")),
        ("create", json!({"path": "/w/b.rs", "file_text": "fn b() {}"}), "Write", "file_path", json!("/w/b.rs")),
        ("create", json!({"path": "/w/b.rs", "file_text": "fn b() {}"}), "Write", "content", json!("fn b() {}")),
        ("task", json!({"agent_type": "explore", "prompt": "find it"}), "Agent", "subagent_type", json!("explore")),
        ("task", json!({"agent_type": "explore", "prompt": "find it"}), "Agent", "prompt", json!("find it")),
        ("write_agent", json!({"agent_id": "w-1", "message": "hi"}), "SendMessage", "to", json!("w-1")),
        ("write_agent", json!({"agent_ids": ["w-1", "w-2"], "message": "hi"}), "SendMessage", "to", json!(["w-1", "w-2"])),
        ("apply_patch", patch.clone(), "Edit", "file_path", json!("/w/c.rs")),
        ("apply_patch", patch, "Edit", "files", json!(["/w/c.rs", "/w/d.rs"])),
        ("rg", json!({"pattern": "x"}), "rg", "pattern", json!("x")),
        ("odd_tool", json!("raw string"), "odd_tool", "input", json!("raw string")),
    ];
    for (tool, args, want_name, key, want) in cases {
        let (name, input) = map_tool(tool, &args);
        assert_eq!(name, want_name, "{tool}");
        assert_eq!(input.get(key), Some(&want), "{tool}.{key}");
    }
}

#[test]
fn user_sources_become_origins_and_only_typed_turns_are_human() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "typed with a source")
        .user(None, "typed by an older cli")
        .user(None, "[INLINE REVIEW] programmatic")
        .user(Some("agent-w-1"), "status from a peer")
        .user(Some("schedule-2"), "scheduled tick")
        .user(Some("skill-foo"), "skill text");
    let conv = c.convert();
    let lines = parse(&conv.main);
    let origin = |i: usize| lines[i].get("origin").cloned().unwrap_or(Value::Null);
    assert_eq!(lines.len(), 6);
    assert_eq!(origin(0)["kind"], "human");
    assert_eq!(origin(1), Value::Null);
    assert_eq!(origin(3)["kind"], "peer");
    assert_eq!(origin(3)["fromSession"], "w-1");
    assert_eq!(origin(4)["kind"], "scheduled");
    assert_eq!(origin(5)["kind"], "system");
    // The raw typed content, not the harness-transformed one.
    assert_eq!(lines[0]["message"]["content"], "typed with a source");
    let ses = traces::parse_session(BufReader::new(joined(&conv.main).as_bytes()));
    let humans: Vec<_> = ses.events.iter().filter_map(|e| match &e.kind {
        traces::EvKind::Human { text } => Some(text.as_str()),
        _ => None,
    }).collect();
    assert_eq!(humans, ["typed with a source", "typed by an older cli"]);
    assert_eq!(conv.record.human_turns_by_utc_hour.values().sum::<usize>(), 2);
    assert_eq!(conv.record.user_messages.get("peer"), Some(&1));
    let hs = handoffs::parse_session(BufReader::new(joined(&conv.main).as_bytes()));
    assert!(hs.events.iter().any(|e| matches!(&e.kind, handoffs::Kind::PeerIn { from_session, .. } if from_session == "w-1")));
}

#[test]
fn failures_and_nonzero_bash_exits_are_errors_like_claude_code() {
    let mut c = C::default();
    c.start("/w/repo")
        .call(None, "t1", "bash", json!({"command": "make"}))
        .ok(None, "t1", "boom\n<shellId: 1 completed with exit code 2>")
        .call(None, "t2", "bash", json!({"command": "true"}))
        .ok(None, "t2", "fine\n<exited with exit code 0>")
        .call(None, "t3", "view", json!({"path": "/w/nope"}))
        .fail(None, "t3", "failure", "Path does not exist")
        .call(None, "t4", "read_bash", json!({"shellId": "1"}))
        .ok(None, "t4", "tail\n<shellId: 1 completed with exit code 1>");
    let lines = parse(&c.convert().main);
    let results: Vec<(bool, String)> = lines.iter().filter_map(|l| {
        let b = l["message"]["content"].as_array()?.first()?.clone();
        (b["type"] == "tool_result").then(|| (b["is_error"].as_bool().unwrap(), b["content"].as_str().unwrap().to_string()))
    }).collect();
    assert_eq!(results.iter().map(|r| r.0).collect::<Vec<_>>(), [true, false, true, false]);
    assert_eq!(results[2].1, "Path does not exist");
    let uses: Vec<String> = lines.iter().filter_map(|l| {
        let b = l["message"]["content"].as_array()?.first()?.clone();
        (b["type"] == "tool_use").then(|| b["name"].as_str().unwrap().to_string())
    }).collect();
    assert_eq!(uses, ["Bash", "Bash", "Read", "read_bash"]);
    assert_eq!(lines[0]["copilotTool"], "bash");
}

#[test]
fn denials_rejections_cancels_and_aborts() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "deploy it")
        .call(None, "t1", "bash", json!({"command": "rm -rf build"}))
        .fail(None, "t1", "denied", "Permission denied and could not request permission from user")
        .call(None, "t2", "bash", json!({"command": "git push --force"}))
        .fail(None, "t2", "denied", "The user rejected this tool call. User feedback: no, wrong branch")
        .call(None, "t3", "bash", json!({"command": "sleep 100"}))
        .fail(None, "t3", "failure", "Cancelled")
        .ev("abort", json!({"reason": "user_initiated"}))
        .ev("abort", json!({"reason": "subagent_cancelled"}));
    let conv = c.convert();
    let lines = parse(&conv.main);
    let kinds: Vec<Option<&str>> = lines.iter().map(|l| l.get("toolDenialKind").and_then(Value::as_str)).collect();
    assert!(kinds.contains(&Some("denied")));
    assert!(kinds.contains(&Some("interrupted")));
    let ses = traces::parse_session(BufReader::new(joined(&conv.main).as_bytes()));
    assert_eq!(traces::detect_permission_denial(&ses.events).len(), 2);
    assert_eq!(traces::detect_tool_error(&ses.events).len(), 0, "denials and cancels are not tool errors");
    assert_eq!(traces::detect_user_interrupt(&ses.events).len(), 1, "only the user's abort");
    let corrections = traces::detect_user_correction(&ses.events);
    assert_eq!(corrections.len(), 1);
    assert_eq!(ses.events[corrections[0]].kind, traces::EvKind::Human { text: "no, wrong branch".into() });
    assert_eq!(conv.record.denials, 2);
    assert_eq!(conv.record.aborts, 1);
}

#[test]
fn subagent_events_become_a_sidechain_transcript_with_its_brief() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "investigate")
        .call(None, "task-1", "task", json!({"agent_type": "explore", "prompt": "Find the parser. Do not open a PR."}))
        .say(Some("task-1"), "looking")
        .call(Some("task-1"), "s1", "rg", json!({"pattern": "parse"}))
        .ok(Some("task-1"), "s1", "src/a.rs:1: parse")
        .ok(None, "task-1", "found it");
    let conv = c.convert();
    assert_eq!(conv.subagents.len(), 1);
    let sub = parse(&conv.subagents["task-1"]);
    assert_eq!(sub[0]["message"]["content"], "Find the parser. Do not open a PR.");
    assert!(sub.iter().all(|l| l["isSidechain"] == true && l["agentId"] == "task-1"));
    assert_eq!(sub.len(), 4, "brief, text, tool_use, tool_result");
    let main = parse(&conv.main);
    assert!(main.iter().all(|l| l["isSidechain"] == false));
    assert_eq!(main.len(), 3, "human, Agent tool_use, its result");
    // The brief is the subagent's first prompt, never a human turn.
    let ses = traces::parse_session(BufReader::new(joined(&conv.subagents["task-1"]).as_bytes()));
    assert!(!ses.events.iter().any(|e| matches!(e.kind, traces::EvKind::Human { .. })));
    let (run, _) = pr_gap::measure(joined(&conv.subagents["task-1"]).as_bytes(), "p", "p/s-1/subagents/agent-task-1.jsonl");
    assert_eq!(run.brief, pr_gap::Brief::Defers);
    assert_eq!(conv.record.subagent_runs, 1);
    assert_eq!(conv.record.tool_calls_by_utc_hour.values().sum::<usize>(), 2);
}

#[test]
fn cwd_and_branch_follow_the_session_context() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "one")
        .ev("session.resume", json!({"context": {"cwd": "/w/other", "branch": "b2", "repository": "acme/widgets"}}))
        .user(Some("user"), "two");
    let conv = c.convert();
    let lines = parse(&conv.main);
    assert_eq!((lines[0]["cwd"].as_str(), lines[0]["gitBranch"].as_str()), (Some("/w/repo"), Some("feature-x")));
    assert_eq!((lines[1]["cwd"].as_str(), lines[1]["gitBranch"].as_str()), (Some("/w/other"), Some("b2")));
    assert_eq!(conv.record.project, "copilot-w-repo", "the project is fixed by the first context");
    assert_eq!(conv.record.repository, "acme/widgets");
    assert_eq!(conv.record.client, "github/cli");
    assert_eq!(conv.record.copilot_version, "9.9.9");
}

#[test]
fn pr_gap_and_handoffs_read_converted_sessions() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "ship it")
        .call(None, "t1", "bash", json!({"command": "git commit -m 'feat: x'"}))
        .ok(None, "t1", "[feature-x abc123] feat: x\n<shellId: 1 completed with exit code 0>")
        .call(None, "t2", "bash", json!({"command": "git push origin feature-x"}))
        .ok(None, "t2", "To github.com:acme/widgets.git\n * [new branch] feature-x -> feature-x\n<shellId: 2 completed with exit code 0>")
        .call(None, "t3", "write_agent", json!({"agent_id": "w-1", "message": "pushed"}))
        .ok(None, "t3", "sent");
    let conv = c.convert();
    let (run, _) = pr_gap::measure(joined(&conv.main).as_bytes(), "p", "p/s-1.jsonl");
    assert_eq!((run.commits, run.pushes, run.prs_created), (1, 1, 0));
    assert!(run.outcome.is_gap(), "{:?}", run.outcome);
    let hs = handoffs::parse_session(BufReader::new(joined(&conv.main).as_bytes()));
    let sigs: Vec<handoffs::Signal> = handoffs::detect_all(&hs.events).into_values().flatten().collect();
    assert!(sigs.contains(&handoffs::Signal::MessageOut), "{sigs:?}");
}

#[test]
fn sidecar_record_holds_counts_never_text() {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "a very distinctive operator sentence")
        .call(None, "t1", "bash", json!({"command": "echo distinctive-argument"}))
        .ok(None, "t1", "distinctive-output\n<exited with exit code 0>")
        .ev("hook.end", json!({"hookType": "userPromptSubmitted", "success": false, "error": {"message": "node missing"}}));
    let rec = serde_json::to_string(&c.convert().record).unwrap();
    assert!(!rec.contains("distinctive"), "{rec}");
    let v: Value = serde_json::from_str(&rec).unwrap();
    assert_eq!(v["hook_failures"], 1);
    assert_eq!(v["tool_calls_by_utc_hour"]["2026-01-01T00"], 1);
}

fn write_source(root: &Path, sid: &str, c: &C) {
    std::fs::create_dir_all(root.join(sid)).unwrap();
    std::fs::write(root.join(sid).join("events.jsonl"), c.text()).unwrap();
    std::fs::write(root.join(sid).join("workspace.yaml"), format!("id: {sid}\nclient_name: github/cli\n")).unwrap();
}

/// A session that struggles: a failing build retried, a test failure in a
/// subagent, a denial and a correction.
fn struggling() -> C {
    let mut c = C::default();
    c.start("/w/repo")
        .user(Some("user"), "fix the build")
        .call(None, "t1", "bash", json!({"command": "cargo test --all"}))
        .ok(None, "t1", "test result: FAILED. 1 passed; 1 failed\n<shellId: 1 completed with exit code 101>")
        .call(None, "t2", "bash", json!({"command": "cargo test --all"}))
        .ok(None, "t2", "test result: FAILED. 1 passed; 1 failed\n<shellId: 2 completed with exit code 101>")
        .user(Some("user"), "no, that's wrong")
        .call(None, "task-1", "task", json!({"agent_type": "task", "prompt": "run the suite"}))
        .call(Some("task-1"), "s1", "bash", json!({"command": "pytest -q"}))
        .ok(Some("task-1"), "s1", "FAILED tests/test_a.py::test_x - assert 1 == 2\n<shellId: 3 completed with exit code 1>")
        .ok(None, "task-1", "the suite fails")
        .call(None, "t3", "bash", json!({"command": "rm -rf target"}))
        .fail(None, "t3", "denied", "Permission denied and could not request permission from user");
    c
}

#[test]
fn traces_mines_converted_sessions_end_to_end() {
    let d = tempfile::tempdir().unwrap();
    let (root, conv, mined) = (d.path().join("state"), d.path().join("conv"), d.path().join("mined"));
    write_source(&root, "s-1", &struggling());
    let mut quiet = C::default();
    quiet.start("/w/other").user(Some("user"), "hello").say(None, "hi");
    write_source(&root, "s-2", &quiet);
    let sum = convert_root(&root, &conv, "", 2).unwrap();
    assert_eq!((sum.sessions, sum.processed, sum.subagent_transcripts), (2, 2, 1));
    assert!(conv.join("copilot-w-repo/s-1.jsonl").is_file());
    assert!(conv.join("copilot-w-repo/s-1/subagents/agent-task-1.jsonl").is_file());
    let t = traces::mine_traces(&conv, &mined, "", 2, &[]).unwrap();
    assert_eq!(t.sessions, 2, "the subagent transcript belongs to its parent session");
    for sig in ["tool_error", "retry", "user_correction", "test_failure", "permission_denial"] {
        assert!(t.episodes_per_signal.contains_key(sig), "{sig}: {:?}", t.episodes_per_signal);
    }
    let eps: Vec<traces::Episode> = std::fs::read_to_string(mined.join("episodes.jsonl")).unwrap()
        .lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(eps.iter().any(|e| e.subagent && e.session_id == "s-1"));
    assert!(eps.iter().all(|e| e.project == "copilot-w-repo" || e.project == "copilot-w-other"));
    // The measurement's denominator: every tool call, subagents included, in its UTC hour.
    let state: BTreeMap<String, traces::FileState> =
        serde_json::from_str(&std::fs::read_to_string(mined.join("traces-state.json")).unwrap()).unwrap();
    let calls: usize = state.values().flat_map(|s| s.facts.tool_calls_by_utc_hour.values()).sum();
    assert_eq!(calls, 5);
    // No sidecar is mistaken for a transcript.
    assert_eq!(t.transcripts, 3);
}

#[test]
fn reruns_skip_unchanged_reconvert_changed_and_drop_vanished() {
    let d = tempfile::tempdir().unwrap();
    let (root, conv) = (d.path().join("state"), d.path().join("conv"));
    write_source(&root, "s-1", &struggling());
    let mut other = C::default();
    other.start("/w/other").user(Some("user"), "hello");
    write_source(&root, "s-2", &other);
    assert_eq!(convert_root(&root, &conv, "", 2).unwrap().processed, 2);
    let again = convert_root(&root, &conv, "", 2).unwrap();
    assert_eq!((again.processed, again.skipped_unchanged), (0, 2));
    // Output mtime follows the source, so traces' --since and its cache see the session's age.
    let src = mtime_size(&root.join("s-1/events.jsonl")).unwrap().0;
    assert_eq!(mtime_size(&conv.join("copilot-w-repo/s-1.jsonl")).unwrap().0, src);
    other.say(None, "and more");
    write_source(&root, "s-2", &other);
    assert_eq!(convert_root(&root, &conv, "", 2).unwrap().processed, 1);
    std::fs::remove_dir_all(root.join("s-1")).unwrap();
    let after = convert_root(&root, &conv, "", 2).unwrap();
    assert_eq!((after.sessions, after.removed), (1, 1));
    assert!(!conv.join("copilot-w-repo/s-1.jsonl").exists());
    assert!(!conv.join("copilot-w-repo/s-1/subagents/agent-task-1.jsonl").exists());
    let index = std::fs::read_to_string(conv.join(SESSIONS_FILE)).unwrap();
    assert_eq!(index.lines().count(), 1);
}

#[test]
fn a_directory_without_events_is_not_a_session() {
    let d = tempfile::tempdir().unwrap();
    let (root, conv) = (d.path().join("state"), d.path().join("conv"));
    std::fs::create_dir_all(root.join("empty/files")).unwrap();
    std::fs::write(root.join("empty/files/notes.jsonl"), "{\"type\":\"user\"}\n").unwrap();
    assert_eq!(convert_root(&root, &conv, "", 1).unwrap().sessions, 0);
}
