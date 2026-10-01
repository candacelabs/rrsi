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
    minute: usize,
    sidechain: bool,
}

impl T {
    fn new(session: &str) -> T {
        T { session: session.into(), ..T::default() }
    }
    fn sub(session: &str, brief: &str) -> T {
        let mut t = T { session: session.into(), sidechain: true, ..T::default() };
        t.push(json!({"type": "user", "agentId": "a1", "message": {"role": "user", "content": brief}}));
        t
    }
    fn at(&mut self, minute: usize) -> &mut Self {
        self.minute = minute;
        self
    }
    fn push(&mut self, mut v: Value) -> &mut Self {
        let n = self.minute;
        self.minute += 1;
        v["timestamp"] = json!(format!("2026-01-01T{:02}:{:02}:00.000Z", n / 60, n % 60));
        v["sessionId"] = json!(self.session);
        v["cwd"] = json!("/w/repo");
        v["gitBranch"] = json!("feat/x");
        v["isSidechain"] = json!(self.sidechain);
        self.lines.push(v.to_string());
        self
    }
    fn human(&mut self, t: &str) -> &mut Self {
        self.push(json!({"type": "user", "origin": {"kind": "human"}, "message": {"content": t}}))
    }
    fn call(&mut self, id: &str, name: &str, input: Value) -> &mut Self {
        self.push(json!({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}}))
    }
    fn bash(&mut self, id: &str, cmd: &str, err: bool, out: &str) -> &mut Self {
        self.call(id, "Bash", json!({"command": cmd}));
        self.push(json!({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "is_error": err, "content": out}]}}))
    }
    fn bytes(&self) -> Vec<u8> {
        self.lines.join("\n").into_bytes()
    }
    fn measure(&self, file: &str) -> (Run, Vec<Moment>) {
        measure(&self.bytes(), "proj", file)
    }
}

const NEW_BRANCH: &str = "remote:\nremote: Create a pull request for 'feat/x' on GitHub by visiting:\nremote:      https://github.com/acme/widgets/pull/new/feat/x\nremote:\nTo github.com:acme/widgets.git\n * [new branch]      feat/x -> feat/x";
const UPDATE: &str = "To https://github.com/acme/widgets.git\n   1a2b3c4..5d6e7f8  feat/x -> feat/x";

#[test]
fn messages_and_heredocs_are_not_operations() {
    assert_eq!(shell_ops("git add -A && git commit -qm \"wip: then git push and gh pr create\" && git push -u origin feat/x"), (1, 1, 0));
    assert_eq!(shell_ops("git commit -F- <<'EOF'\ngit push origin main\ngh pr create\nEOF\necho done"), (1, 0, 0));
    assert_eq!(shell_ops("echo 'remember to git push'"), (0, 0, 0));
    assert_eq!(shell_ops("cat notes.md | grep 'git commit'"), (0, 0, 0));
    assert_eq!(shell_ops("git -C /w/repo commit -m x; git -C /w/repo push"), (1, 1, 0));
}

#[test]
fn amend_dry_run_and_delete_are_not_new_work() {
    assert_eq!(shell_ops("git commit --amend --no-edit && git push --force-with-lease"), (0, 1, 0));
    assert_eq!(shell_ops("git push --dry-run origin feat/x; git push origin --delete old"), (0, 0, 0));
    assert_eq!(shell_ops("git commit --dry-run"), (0, 0, 0));
}

#[test]
fn pr_creation_forms() {
    assert_eq!(shell_ops("gh pr create --draft --title \"Add a thing\" --body-file b.md").2, 1);
    assert_eq!(shell_ops("gh api repos/acme/widgets/pulls -f title=x -f head=feat/x -f base=main").2, 1);
    assert_eq!(shell_ops("gh api -X POST \"repos/acme/widgets/pulls\" --input pr.json").2, 1);
    assert_eq!(shell_ops("gh api -X PATCH repos/acme/widgets/pulls/8 -F body=@b.md").2, 0, "an edit is not a create");
    assert_eq!(shell_ops("gh api repos/acme/widgets/pulls").2, 0, "a GET lists");
    assert_eq!(shell_ops("gh pr create --help").2, 0);
}

#[test]
fn push_output_names_repo_branch_and_whether_github_saw_no_pr() {
    assert_eq!(push_targets(NEW_BRANCH), vec![("acme/widgets".into(), "feat/x".into(), true)]);
    assert_eq!(push_targets(UPDATE), vec![("acme/widgets".into(), "feat/x".into(), false)]);
    assert!(push_targets("Everything up-to-date").is_empty());
    assert!(push_targets("To github.com:acme/widgets.git\n ! [rejected]        feat/x -> feat/x (fetch first)").is_empty());
}

#[test]
fn brief_classes() {
    assert_eq!(classify_brief("Implement it. Don't open a separate PR; push to the shared branch."), Brief::Defers);
    assert_eq!(classify_brief("When the slice is complete, open the PR against main."), Brief::Defers);
    assert_eq!(classify_brief("Do not push until the tests pass."), Brief::Defers);
    assert_eq!(classify_brief("Never push to main. Open a draft PR right after your first commit."), Brief::Early);
    assert_eq!(classify_brief("Fix the parser and report back."), Brief::Silent);
    assert_eq!(classify_brief(""), Brief::None);
    assert!(defer_phrases("don't push to main, do not push --force").is_empty());
}

#[test]
fn operator_corrections_are_not_requests() {
    assert!(is_correction("ACTIVELY WORKING AGENTS NEED TO ALWAYS HAVE A PR"));
    assert!(is_correction("why is there no PR for the parser branch?"));
    assert!(is_correction("three of them haven't even pushed"));
    assert!(!is_correction("open a PR for this when you are ready"));
    assert!(!is_correction("the previous approach works"));
}

#[test]
fn subagent_that_pushed_late_and_never_opened_a_pr() {
    let mut t = T::sub("s1", "Implement the widget. When complete, open a PR.");
    t.at(1).bash("1", "git add -A && git commit -qm 'first'", false, "")
        .at(3).bash("2", "git commit -qm 'second'", false, "")
        .at(25).bash("3", "git push -u origin feat/x", false, NEW_BRANCH)
        .at(30).bash("4", "git commit -qm 'third'", false, "")
        .at(40).bash("5", "git commit -qm 'nothing'", true, "nothing to commit, working tree clean");
    let (run, moments) = t.measure("proj/s1/subagents/agent-a1.jsonl");
    assert!(run.subagent && moments.is_empty());
    assert_eq!((run.commits, run.commits_before_first_push, run.unpushed_at_end, run.pushes, run.prs_created), (3, 2, 1, 1, 0));
    assert_eq!(run.commit_to_push_secs, Some(24 * 60));
    assert_eq!(run.brief, Brief::Defers);
    assert_eq!(run.pushed, vec![("acme/widgets".into(), "feat/x".into(), true)]);
    assert_eq!(run.outcome, Outcome::PushedNoPr);
    assert_eq!(run_signals(&run), vec![Signal::PushedNoPr, Signal::SlowPush, Signal::BriefDefersPr]);
}

#[test]
fn early_pr_run_is_healthy_and_a_cat_brief_counts() {
    let mut t = T::sub("s2", "Read your brief: cat /tmp/briefs/b1.md");
    t.call("0", "Bash", json!({"command": "cat /tmp/briefs/b1.md"}))
        .push(json!({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "0", "is_error": false,
            "content": "Open a draft PR right after your first commit."}]}}))
        .at(2).bash("1", "git commit -qm wip && git push -u origin feat/x", false, NEW_BRANCH)
        .at(4).bash("2", "gh pr create --draft --title 'Widget' --body-file b.md", false, "https://github.com/acme/widgets/pull/9");
    let (run, _) = t.measure("proj/s2/subagents/agent-a2.jsonl");
    assert_eq!(run.brief, Brief::Early);
    assert_eq!(run.outcome, Outcome::PrOpened);
    assert_eq!((run.commit_to_push_secs, run.push_to_pr_secs), (Some(0), Some(120)));
    assert_eq!(run.pr_urls, vec!["https://github.com/acme/widgets/pull/9".to_string()]);
    assert!(run_signals(&run).is_empty());
    assert!(episodes(&run, &[], "").is_empty());
}

#[test]
fn never_pushed_and_failed_calls() {
    let mut t = T::sub("s3", "Fix it.");
    t.bash("1", "git commit -qm a && git push", true, "error: failed to push some refs\n ! [rejected] feat/x -> feat/x")
        .bash("2", "gh pr create --title 'x y z'", true, "no commits between main and feat/x");
    let (run, _) = t.measure("proj/s3/subagents/agent-a3.jsonl");
    assert_eq!((run.commits, run.pushes, run.prs_created), (0, 0, 0), "an errored chain proves nothing");
    let mut t = T::sub("s3", "Fix it.");
    t.bash("1", "git commit -qm a", false, "");
    let (run, _) = t.measure("proj/s3/subagents/agent-a3.jsonl");
    assert_eq!(run.outcome, Outcome::NeverPushed);
    assert_eq!(run.unpushed_at_end, 1);
    let eps = episodes(&run, &[], "Fix it.");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].signals, [(Signal::NeverPushed, 1)].into());
    assert_eq!(eps[0].user_turn, "Fix it.");
}

#[test]
fn pushes_to_main_need_no_pr() {
    let mut t = T::new("s4");
    t.bash("1", "git commit -qm a && git push", false, "To github.com:acme/notes.git\n   1a2b3c4..5d6e7f8  main -> main");
    assert_eq!(t.measure("proj/s4.jsonl").0.outcome, Outcome::PushedDefault);
}

#[test]
fn orchestrator_briefs_and_corrections() {
    let mut t = T::new("s5");
    t.human("no PR yet, that's fine: we have not started")
        .call("a", "Agent", json!({"prompt": "Implement slice 3. Don't open a separate PR.", "description": "slice 3"}))
        .call("b", "Agent", json!({"prompt": "Implement slice 4. Open a draft PR right after your first commit."}))
        .human("ACTIVELY WORKING AGENTS NEED TO ALWAYS HAVE A PR");
    let (run, moments) = t.measure("proj/s5.jsonl");
    assert_eq!(run.outcome, Outcome::NoCommits);
    let sigs: Vec<_> = moments.iter().map(|m| m.signal).collect();
    assert_eq!(sigs, vec![Some(Signal::BriefDefersPr), Some(Signal::OperatorPrCorrection)], "the first turn precedes any work");
    let eps = episodes(&run, &moments, "");
    assert_eq!(eps.len(), 2);
    assert_ne!(eps[0].id, eps[1].id);
    assert_eq!(eps[1].user_turn, "ACTIVELY WORKING AGENTS NEED TO ALWAYS HAVE A PR");
}

fn run_with(c2p: Option<u64>, p2r: Option<u64>, c2end: u64, outcome: Outcome) -> Run {
    Run { commits: 1, commit_to_push_secs: c2p, push_to_pr_secs: p2r, commit_to_end_secs: Some(c2end),
          push_to_end_secs: c2p.map(|p| c2end - p), outcome, ..Run::default() }
}

#[test]
fn rules_fire_on_gaps_still_active_after_n_minutes() {
    let quick = run_with(Some(60), Some(60), 3600, Outcome::PrOpened);
    let late_pr = run_with(Some(60), Some(20 * 60), 3600, Outcome::PrOpened);
    let never = run_with(None, None, 40 * 60, Outcome::NeverPushed);
    let short_gap = run_with(None, None, 3 * 60, Outcome::NeverPushed);
    let no_pr = run_with(Some(0), None, 90 * 60, Outcome::PushedNoPr);
    for (r, at5, at30) in [(&quick, false, false), (&late_pr, true, false), (&never, true, true), (&short_gap, false, false), (&no_pr, true, true)] {
        assert_eq!((fires(r, 300), fires(r, 1800)), (at5, at30));
    }
    let scores = score_rules(&[&quick, &late_pr, &never, &short_gap, &no_pr]);
    let five = scores.iter().find(|s| s.minutes == 5).unwrap();
    assert_eq!((five.fires, five.gaps_caught, five.nags, five.gaps_missed, five.score), (3, 2, 1, 1, 1));
    let thirty = scores.iter().find(|s| s.minutes == 30).unwrap();
    assert_eq!((thirty.gaps_caught, thirty.nags, thirty.score), (2, 0, 2));
    let top = top_rule(&scores).unwrap();
    assert_eq!((top.minutes, top.score, top.gaps_caught), (2, 2, 3), "equal score: the rule that catches more gaps wins");
    assert!(top.rule.contains("2 minutes"));
    let tie = [RuleScore { minutes: 5, score: 1, gaps_caught: 1, ..RuleScore::default() },
               RuleScore { minutes: 15, score: 1, gaps_caught: 1, ..RuleScore::default() }];
    assert_eq!(top_rule(&tie).unwrap().minutes, 15, "a full tie goes to the least eager rule");
    assert!(!fires(&run_with(Some(0), None, 9999, Outcome::PushedDefault), 60));
}

struct FakeGithub {
    prs: BTreeMap<String, Vec<GithubPr>>,
    asked: Mutex<Vec<String>>,
}

impl Github for FakeGithub {
    fn prs_for_head(&self, repo: &str, branch: &str) -> Result<Vec<GithubPr>> {
        self.asked.lock().unwrap().push(format!("{repo}#{branch}"));
        if repo == "acme/broken" {
            anyhow::bail!("HTTP 404");
        }
        Ok(self.prs.get(&format!("{repo}#{branch}")).cloned().unwrap_or_default())
    }
}

#[test]
fn github_join_confirms_gaps_and_finds_prs_opened_by_others() {
    let pr = GithubPr { url: "https://github.com/acme/widgets/pull/3".into(), created_at: "2026-01-01T00:10:00Z".into() };
    let gh = FakeGithub { prs: [("acme/widgets#feat/x".to_string(), vec![pr.clone()])].into(), asked: Mutex::new(vec![]) };
    let base = |repo: &str, end: &str| Run { commits: 1, pushes: 1, end: end.into(), outcome: Outcome::PushedNoPr,
                                            pushed: vec![(repo.into(), "feat/x".into(), false)], ..Run::default() };
    let mut runs = vec![base("acme/widgets", "2026-01-01T00:30:00Z"), base("acme/widgets", "2026-01-01T00:05:00Z"),
                        base("acme/other", "2026-01-01T00:30:00Z"), base("acme/broken", "2026-01-01T00:30:00Z")];
    let mut cache = BTreeMap::new();
    let (asked, failed) = join_github(&mut runs, &gh, &mut cache);
    assert_eq!((asked, failed), (3, 1), "a found PR is cached; empty and failed lookups are asked again");
    let outs: Vec<_> = runs.iter().map(|r| r.outcome).collect();
    assert_eq!(outs, [Outcome::PrExisting, Outcome::PushedNoPr, Outcome::PushedNoPr, Outcome::PushedNoPr],
               "a PR created after the run ended does not excuse it");
    assert_eq!(runs[0].github, vec![pr]);
    join_github(&mut runs, &gh, &mut cache);
    assert_eq!(gh.asked.lock().unwrap().iter().filter(|k| *k == "acme/widgets#feat/x").count(), 1);
}

#[test]
fn end_to_end_over_a_synthetic_corpus_and_incremental_rerun() {
    let root = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let w = |rel: &str, t: &T| {
        let p = root.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, t.bytes()).unwrap();
    };
    let mut main = T::new("s1");
    main.call("a", "Agent", json!({"prompt": "Do slice 1. Do not open a PR."}))
        .human("they don't even have a PR");
    w("p/s1.jsonl", &main);
    let mut a = T::sub("s1", "Do slice 1. Do not open a PR.");
    a.bash("1", "git commit -qm a", false, "").at(20).bash("2", "git push -u origin feat/x", false, NEW_BRANCH);
    w("p/s1/subagents/agent-a.jsonl", &a);
    let mut b = T::sub("s1", "Do slice 2. Open a draft PR right after your first commit.");
    b.bash("1", "git commit -qm a && git push -u origin feat/y", false, NEW_BRANCH)
        .bash("2", "gh pr create --draft --title 'Slice two'", false, "https://github.com/acme/widgets/pull/5");
    w("p/s1/subagents/agent-b.jsonl", &b);
    let sum = mine_pr_gap(root.path(), out.path(), "", 2, &[], None).unwrap();
    assert_eq!((sum.transcripts, sum.processed, sum.sessions, sum.projects), (3, 3, 1, 1));
    assert_eq!((sum.subagents.active_runs, sum.subagents.gap_runs, sum.main_sessions.active_runs), (2, 1, 0));
    assert_eq!(sum.episodes_per_signal["pushed_no_pr"], 1);
    assert_eq!(sum.episodes_per_signal["slow_push"], 1);
    assert_eq!(sum.episodes_per_signal["brief_defers_pr"], 2, "the Agent prompt and the subagent's own gap run");
    assert_eq!(sum.episodes_per_signal["operator_pr_correction"], 1);
    assert_eq!(sum.episodes_per_signal["never_pushed"], 0);
    assert_eq!(sum.subagent_gap_by_brief["defers"], (1, 1));
    assert_eq!(sum.subagent_gap_by_brief["early"], (1, 0));
    assert_eq!(sum.github, json!({"joined": false}));
    assert!(sum.top_rule.is_some());
    for f in ["episodes.jsonl", "runs.jsonl", "pr-gap-summary.json", "pr-gap-task.json", "pr-gap-exam-candidates.jsonl", "pr-gap-state.json"] {
        assert!(out.path().join(f).exists(), "{f}");
    }
    let task: Value = serde_json::from_str(&std::fs::read_to_string(out.path().join("pr-gap-task.json")).unwrap()).unwrap();
    assert_eq!(task["evidence"]["gap_runs"], 1);
    assert!(task["trigger_rule"]["rule"].as_str().unwrap().starts_with("when an agent's first commit is"));
    assert_eq!(std::fs::read_to_string(out.path().join("pr-gap-exam-candidates.jsonl")).unwrap().lines().count(), 1);
    let again = mine_pr_gap(root.path(), out.path(), "", 2, &[], None).unwrap();
    assert_eq!((again.processed, again.skipped_unchanged, again.episodes), (0, 3, sum.episodes));
    let later = mine_pr_gap(root.path(), out.path(), "2026-02-01", 2, &[], None).unwrap();
    assert_eq!(later.episodes, 0, "runs that ended before --since are dropped");
}

#[test]
fn misuse_is_refused() {
    use crate::miner::run;
    let repo = tempfile::tempdir().unwrap();
    std::fs::create_dir(repo.path().join(".git")).unwrap();
    let inside = repo.path().join("o");
    assert!(format!("{}", run("pr-gap", json!({"out": inside.to_str().unwrap()})).unwrap_err()).contains("inside the git work tree"));
    assert!(!inside.exists());
    let err = run("pr-gap", json!({"out": "/nonexistent-rrsi-test", "githb": true})).unwrap_err();
    assert!(format!("{err}").contains("githb"), "{err}");
}
