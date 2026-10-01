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

//! Synthetic fixtures only: invented text in the transcript schema.

use super::*;
use serde_json::json;

#[derive(Default)]
struct T {
    lines: Vec<String>,
    session: String,
    cwd: String,
    branch: String,
    minute: usize,
    sidechain: bool,
}

impl T {
    fn new(session: &str, cwd: &str, branch: &str) -> T {
        T { session: session.into(), cwd: cwd.into(), branch: branch.into(), ..T::default() }
    }
    fn push(&mut self, mut v: Value) -> &mut Self {
        let n = self.minute;
        self.minute += 1;
        v["timestamp"] = json!(format!("2026-01-01T{:02}:{:02}:00.000Z", n / 60, n % 60));
        v["sessionId"] = json!(self.session);
        v["cwd"] = json!(self.cwd);
        v["gitBranch"] = json!(self.branch);
        v["isSidechain"] = json!(self.sidechain);
        self.lines.push(v.to_string());
        self
    }
    fn human(&mut self, t: &str) -> &mut Self {
        self.push(json!({"type": "user", "origin": {"kind": "human"}, "message": {"content": t}}))
    }
    fn say(&mut self, t: &str) -> &mut Self {
        self.push(json!({"type": "assistant", "message": {"content": [{"type": "text", "text": t}]}}))
    }
    fn call(&mut self, id: &str, name: &str, input: Value) -> &mut Self {
        self.push(json!({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}}))
    }
    fn result(&mut self, id: &str, err: bool, t: &str) -> &mut Self {
        self.push(json!({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "is_error": err, "content": t}]}}))
    }
    fn peer(&mut self, from: &str, name: &str, body: &str) -> &mut Self {
        let text = format!("Another Claude session sent a message:\n<cross-session-message from=\"uds:/x.sock\" from-session=\"{from}\" from-name=\"{name}\">\n{body}\n</cross-session-message>");
        self.push(json!({"type": "user", "isMeta": true, "origin": {"kind": "peer", "fromSession": from, "name": name, "body": body},
                         "message": {"content": text}}))
    }
    fn session(&self) -> Session {
        parse_session(BufReader::new(self.lines.join("\n").as_bytes()))
    }
    fn hits(&self, s: Signal) -> Vec<usize> {
        s.detect(&self.session().events)
    }
}

#[test]
fn parse_peer_reads_attributes_and_body() {
    let (s, n, b) = parse_peer("x <cross-session-message from=\"u\" from-session=\"local_1\" from-name=\"Builder\">\nhello\n</cross-session-message>").unwrap();
    assert_eq!((s.as_str(), n.as_str(), b.as_str()), ("local_1", "Builder", "hello"));
    assert!(parse_peer("no block").is_none());
}

#[test]
fn peer_messages_from_origin_and_queued_attachments_dedupe() {
    let mut t = T::new("s1", "/r", "feat/a");
    t.peer("local_9", "Builder", "please rebase onto main");
    t.push(json!({"type": "attachment", "attachment": {"type": "queued_command",
        "prompt": "<cross-session-message from=\"u\" from-session=\"local_9\" from-name=\"Builder\">\nplease rebase onto main\n</cross-session-message>"}}));
    t.push(json!({"type": "user", "isMeta": true, "origin": {"kind": "coordinator"}, "message": {"content": "The coordinator sent a message while you were working:\nstop"}}));
    let evs = t.session().events;
    assert_eq!(evs.len(), 2, "{evs:#?}");
    assert!(matches!(&evs[0].kind, Kind::PeerIn { from_name, .. } if from_name == "Builder"));
    assert!(matches!(&evs[1].kind, Kind::CoordinatorIn { .. }));
    assert_eq!(evs[0].branch, "feat/a");
}

#[test]
fn user_relay_phrases() {
    let mut t = T::new("s", "/r", "");
    t.human("tell the other session to stop editing the parser").human("ship it")
        .human("oops I said this in the wrong session");
    assert_eq!(t.hits(Signal::UserRelay), vec![0, 2]);
}

#[test]
fn retractions_and_churn_count_per_sender() {
    let mut t = T::new("s", "/r", "");
    t.peer("a", "A", "Rename: the package is now gamma.")
        .peer("b", "B", "Withdrawn: ignore my last note.")
        .peer("a", "A", "One more rename, it is delta. This one is final.")
        .peer("a", "A", "All good, carry on.");
    assert_eq!(t.hits(Signal::PeerRetraction), vec![0, 1, 2]);
    assert_eq!(t.hits(Signal::PeerChurn), vec![2]);
}

#[test]
fn outcomes_score_acting_and_replying() {
    let mut t = T::new("s", "/r", "");
    t.peer("a", "A", "please run the tests")
        .call("1", "Bash", json!({"command": "cargo test"})).result("1", false, "ok")
        .call("2", "SendMessage", json!({"to": "A", "message": "done"}))
        .peer("b", "B", "fyi only").human("next thing");
    let o = outcomes(&t.session().events);
    assert_eq!(o.len(), 2);
    assert!(o[0].acted && o[0].replied && !o[0].coordinator);
    assert!(!o[1].acted && !o[1].replied);
}

#[test]
fn message_out_conflicts_and_worktree_collisions() {
    let mut t = T::new("s", "/r", "");
    t.call("1", "mcp__x__send_message", json!({"session_id": "q", "message": "hi"}))
        .call("2", "Bash", json!({"command": "git rebase origin/main"}))
        .result("2", true, "CONFLICT (content): Merge conflict in a.go")
        .call("3", "Write", json!({"file_path": "/x"}))
        .result("3", true, "hook error: /x belongs to a different worktree. Do not write to other worktrees.")
        .result("2", false, "Everything up-to-date");
    assert_eq!(t.hits(Signal::MessageOut), vec![0]);
    assert_eq!(t.hits(Signal::VcsConflict), vec![2]);
    assert_eq!(t.hits(Signal::WorktreeCollision), vec![4]);
}

#[test]
fn already_done_blocked_and_ownership_phrases() {
    let mut t = T::new("s", "/r", "");
    t.say("This was already implemented in another branch.")
        .say("I'm waiting for the other session to merge its PR.")
        .say("Done.")
        .human("who owns the parser rewrite?");
    assert_eq!(t.hits(Signal::AlreadyDone), vec![0]);
    assert_eq!(t.hits(Signal::BlockedOnOther), vec![1]);
    assert_eq!(t.hits(Signal::OwnershipQuestion), vec![3]);
}

#[test]
fn polling_needs_repeats_without_a_message_between() {
    let mut t = T::new("s", "/r", "");
    for k in 0..3 {
        t.call(&format!("p{k}"), "Bash", json!({"command": "gh pr view 12 --json state"}));
    }
    t.call("m", "SendMessage", json!({"to": "x"}));
    t.call("p3", "Bash", json!({"command": "gh pr view 12 --json state"}));
    t.call("p4", "Bash", json!({"command": "ls -la"}));
    assert_eq!(t.hits(Signal::Polling), vec![2]);
}

#[test]
fn claims_are_issue_comments() {
    let mut t = T::new("s", "/r", "");
    t.call("1", "Bash", json!({"command": "gh issue comment 7 --body 'Claimed by session \"X\"'"}))
        .call("2", "Bash", json!({"command": "gh issue view 7"}));
    assert_eq!(t.hits(Signal::Claim), vec![0]);
}

#[test]
fn repo_keys_fold_worktrees() {
    assert_eq!(repo_key("/h/repo/.claude/worktrees/wt-a/src/x.rs"), "/h/repo//src/x.rs");
    assert_eq!(repo_key("/h/repo/.claude/worktrees/wt-b/src/x.rs"), "/h/repo//src/x.rs");
    assert_eq!(repo_of("/h/repo/.claude/worktrees/wt-a"), "/h/repo");
    assert_eq!(repo_key("/tmp/x"), "/tmp/x");
}

#[test]
fn title_arg_forms() {
    assert_eq!(title_arg("gh pr create --title \"Fix the parser\" --body x").as_deref(), Some("Fix the parser"));
    assert_eq!(title_arg("gh issue create -t 'Add a gate' -b y").as_deref(), Some("Add a gate"));
    assert_eq!(title_arg("gh pr create --title=Short").as_deref(), Some("Short"));
    assert_eq!(title_arg("gh pr create --fill"), None);
}

#[test]
fn iso_and_back() {
    let s = iso_secs("2026-01-01T01:02:03.000Z").unwrap();
    assert_eq!(ts_of(s), "2026-01-01T01:02:03.000Z");
}

fn two_sessions() -> (T, T) {
    let mut a = T::new("A", "/h/repo/.claude/worktrees/wa", "feat/x");
    a.call("1", "Edit", json!({"file_path": "/h/repo/.claude/worktrees/wa/src/lib.rs"}))
        .call("2", "Bash", json!({"command": "gh pr create --title \"Add the widget parser module\" --body b"}));
    let mut b = T::new("B", "/h/repo/.claude/worktrees/wb", "feat/x");
    b.call("1", "Write", json!({"file_path": "/h/repo/.claude/worktrees/wb/src/lib.rs"}))
        .call("2", "Bash", json!({"command": "gh pr create --title \"Widget parser module: add\" --body b"}));
    (a, b)
}

#[test]
fn cross_session_overlaps_and_duplicates() {
    let (a, b) = two_sessions();
    let fa = facts(&a.session(), "p", "p/A.jsonl", 0);
    let fb = facts(&b.session(), "p", "p/B.jsonl", 0);
    let eps = cross_episodes(&[fa.clone(), fb]);
    let kinds: Vec<Signal> = eps.iter().flat_map(|e| e.signals.keys().copied()).collect();
    assert_eq!(kinds, vec![Signal::FileOverlap, Signal::BranchOverlap, Signal::DuplicateWork]);
    assert_eq!(eps[0].extra["other_session"], "B");
    assert!(eps[0].context[0].text.contains("src/lib.rs"));
    // One session alone overlaps with nobody; its own subagent is the same party.
    let mut sub = fa.clone();
    sub.file = "p/A/subagents/agent-1.jsonl".into();
    assert!(cross_episodes(&[fa, sub]).is_empty());
}

#[test]
fn far_apart_edits_do_not_overlap() {
    let (a, _) = two_sessions();
    let fa = facts(&a.session(), "p", "p/A.jsonl", 0);
    let mut fb = fa.clone();
    fb.session_id = "B".into();
    fb.edits = vec![(fa.edits[0].0.clone(), fa.edits[0].1 + OVERLAP_SLACK_SECS + 120)];
    fb.branches.clear();
    fb.titles.clear();
    assert!(cross_episodes(&[fa, fb]).is_empty());
}

#[test]
fn mine_handoffs_end_to_end_and_incremental() {
    let root = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let (mut a, b) = two_sessions();
    a.peer("local_B", "B", "Withdrawn: void my earlier ruling.").say("ok");
    std::fs::create_dir_all(root.path().join("p")).unwrap();
    std::fs::write(root.path().join("p/A.jsonl"), a.lines.join("\n")).unwrap();
    std::fs::write(root.path().join("p/B.jsonl"), b.lines.join("\n")).unwrap();
    let s = mine_handoffs(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((s.transcripts, s.processed, s.sessions), (2, 2, 2));
    assert_eq!(s.peer.messages, 1);
    assert_eq!(s.peer.retractions, 1);
    for k in ["file_overlap", "branch_overlap", "duplicate_work", "peer_message", "peer_retraction"] {
        assert_eq!(s.episodes_per_signal.get(k), Some(&1), "{k}: {:?}", s.episodes_per_signal);
    }
    let again = mine_handoffs(root.path(), out.path(), "", 2, &[]).unwrap();
    assert_eq!((again.processed, again.skipped_unchanged, again.episodes), (0, 2, s.episodes));
    assert_eq!(std::fs::read_to_string(out.path().join("episodes.jsonl")).unwrap().lines().count(), s.episodes);
}

#[test]
fn output_inside_a_work_tree_is_refused() {
    let repo = tempfile::tempdir().unwrap();
    std::fs::create_dir(repo.path().join(".git")).unwrap();
    let err = mine_handoffs(repo.path(), &repo.path().join("o"), "", 1, &[]).unwrap_err();
    assert!(format!("{err}").contains("inside the git work tree"));
}
