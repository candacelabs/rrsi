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

//! Synthetic fixtures only: invented PRs, repositories and scorer output.

use super::*;
use serde_json::json;

fn pr(number: u64, title: &str, body: &str) -> Pr {
    Pr { number, title: title.into(), body: body.into(), ..Pr::default() }
}

fn id_of(title: &str, body: &str) -> Option<(String, Vec<String>)> {
    classify(&pr(7, title, body)).map(|t| (t.id, t.evidence))
}

#[test]
fn slice_titles_name_their_slice() {
    let cases = [
        ("Slice W1: event intake to the owning agent", "W1", "title_slice"),
        ("slice: config capability", "pr-7", "title_slice"),
        ("gotth-live: the UI joins every goroutine (slice G1)", "G1", "title_slice_tag"),
        ("runner: drop orphans (slice: zero leaked runner registrations)", "zero-leaked-runner-registrations",
         "title_slice_tag"),
        ("db: one pool owner (S4)", "S4", "title_id"),
        ("harness: operating rules (HX1)", "HX1", "title_id"),
        ("N1: retire the old name", "N1", "title_id"),
        ("runtime: lazily started services (L1, #283)", "L1", "title_id"),
        ("Slice H: one host app per machine", "H", "title_slice"),
        ("Slice add the thing", "pr-7", "title_slice"),
    ];
    for (title, id, rule) in cases {
        let (got, ev) = id_of(title, "").unwrap_or_else(|| panic!("{title} is a slice"));
        assert_eq!(got, id, "{title}");
        assert!(ev.iter().any(|e| e == rule), "{title}: {ev:?}");
    }
}

#[test]
fn slice_bodies_name_their_slice() {
    let (id, ev) = id_of("Exports: staging is private", "## Summary\n\nThis PR is slice P: the manifest.\n").unwrap();
    assert_eq!((id.as_str(), ev), ("P", vec!["body_slice_statement".to_string()]));
    let (id, ev) = id_of("x: y", "## Slice\n\nK2 — stub the controller\n\n## Proof\n\nok\n").unwrap();
    assert_eq!(id, "K2");
    assert_eq!(ev, ["body_slice_section"]);
    let (id, _) = id_of("x: y", "## Slice: B3\n").unwrap();
    assert_eq!(id, "B3");
    let (id, _) = id_of("x: y", "## Slice\n\nZero leaked registrations. More text.\n").unwrap();
    assert_eq!(id, "zero-leaked-registrations");
    let (id, ev) = id_of("x: y", "Delivers one slice of the plan.\n\n### Proof\n\n`go test` ok\n").unwrap();
    assert_eq!((id.as_str(), ev), ("pr-7", vec!["body_proof_section".to_string()]));
    // A body slice id only refines a PR already classified as a slice.
    let (id, ev) = id_of("Harness: vertical slices (slice: harness rule)", "This is slice S0 of the plan.").unwrap();
    assert_eq!(id, "S0", "a short code beats a slug");
    assert!(ev.contains(&"body_slice_id".to_string()));
}

#[test]
fn ordinary_prs_are_not_slices() {
    for (title, body) in [
        ("runner: raise max_workers to 8", "## Summary\n\n## Verification\n\n| a | b |\n"),
        ("Fix the build (#261)", ""),
        ("CSF ontology alignment", "three of five ontology slices are complete\n\n## Verification\n"),
        ("docs: proof reading", "## Proof\n\nno slice word anywhere? none.\n".replace("slice", "part").as_str()),
        ("ci: bump", "| Slice specs (per slice report) | ok |"),
    ] {
        assert!(classify(&pr(1, title, body)).is_none(), "{title}");
    }
}

#[test]
fn proof_section_is_recorded() {
    let t = classify(&pr(3, "Slice A1: x", "## Summary\n## Verification\n")).unwrap();
    assert!(t.proof_section);
    let t = classify(&pr(3, "Slice A1: x", "## Summary\n")).unwrap();
    assert!(!t.proof_section);
}

#[test]
fn slugs_are_lowercase_and_hyphenated() {
    assert_eq!(slug("  Zero leaked: runner/registrations! "), "zero-leaked-runner-registrations");
    assert_eq!(slug("--"), "");
}

#[test]
fn gh_and_snake_case_pr_lists_parse() {
    let gh = json!([{"number": 5, "title": "t", "body": "b", "mergeCommit": {"oid": "abc"},
                     "mergedAt": "2026-10-01T00:00:00Z", "url": "https://example.invalid/5"},
                    {"number": 6, "title": "u", "mergeCommit": null}]);
    let prs = prs_from_json(&gh).unwrap();
    assert_eq!(prs[0], Pr { number: 5, title: "t".into(), body: "b".into(), merge_commit: Some("abc".into()),
                            merged_at: Some("2026-10-01T00:00:00Z".into()), url: Some("https://example.invalid/5".into()) });
    assert_eq!(prs[1].merge_commit, None);
    assert_eq!(prs_from_json(&json!([{"number": 1, "title": "x", "merge_commit": "def"}])).unwrap()[0].merge_commit,
               Some("def".into()));
    assert!(prs_from_json(&json!([{"title": "no number"}])).is_err());
    assert!(prs_from_json(&json!({"not": "a list"})).is_err());
}

#[test]
fn scorer_signals_and_moves() {
    let nested = json!({"score": 4.5, "weights": {"CS-15": 1}, "signals": {"CS-15": 3, "CS-16": {"count": 2}, "note": "x"}});
    assert_eq!(signals_of(&nested), BTreeMap::from([("CS-15".into(), 3.0), ("CS-16".into(), 2.0), ("score".into(), 4.5)]));
    // The shape `candace ontology score` prints: a list, null = not measured.
    let listed = json!({"score": 0.25, "penalty": 10, "complete": false, "signals": [
        {"id": "CS-15", "status": "measured", "count": 4, "weight": 2},
        {"id": "unlinked-terms", "status": "not_measured", "count": null, "reason": "TODO"}]});
    assert_eq!(signals_of(&listed),
               BTreeMap::from([("CS-15".into(), 4.0), ("penalty".into(), 10.0), ("score".into(), 0.25)]));
    assert_eq!(unmeasured_of(&listed), ["unlinked-terms"]);
    assert!(unmeasured_of(&nested).is_empty());
    let flat = json!({"CS-17": 1, "label": "x"});
    assert_eq!(signals_of(&flat), BTreeMap::from([("CS-17".into(), 1.0)]));
    let before = BTreeMap::from([("a".to_string(), 3.0), ("b".to_string(), 1.0), ("gone".to_string(), 2.0)]);
    let after = BTreeMap::from([("a".to_string(), 1.0), ("b".to_string(), 1.0), ("new".to_string(), 4.0)]);
    assert_eq!(moved(&before, &after),
               BTreeMap::from([("a".into(), -2.0), ("gone".into(), -2.0), ("new".into(), 4.0)]));
}

#[test]
fn the_scorer_runs_in_the_tree_and_its_failures_are_reported() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("o.json"), r#"{"signals": {"CS-16": 2}, "score": 7}"#).unwrap();
    let got = score_tree("echo scoring >&2; echo 'log line'; cat {tree}/o.json | tr -d '\\n'; echo", d.path()).unwrap();
    assert_eq!(signals_of(&got), BTreeMap::from([("CS-16".into(), 2.0), ("score".into(), 7.0)]));
    let err = score_tree("echo boom >&2; exit 3", d.path()).unwrap_err();
    assert!(format!("{err}").contains("exit 3: boom"), "{err}");
    assert!(score_tree("echo not json", d.path()).is_err());
}

/// A git repository built commit by commit with chosen subjects.
struct Repo {
    dir: tempfile::TempDir,
}

impl Repo {
    fn new() -> Repo {
        let r = Repo { dir: tempfile::tempdir().unwrap() };
        r.git(&["init", "-q", "-b", "main"]);
        r
    }
    fn git(&self, args: &[&str]) -> String {
        let o = std::process::Command::new("git").args(args).current_dir(self.dir.path())
            .env("GIT_AUTHOR_NAME", "t").env("GIT_AUTHOR_EMAIL", "t@example.invalid")
            .env("GIT_COMMITTER_NAME", "t").env("GIT_COMMITTER_EMAIL", "t@example.invalid")
            .env("GIT_AUTHOR_DATE", "2020-01-01T00:00:00Z").env("GIT_COMMITTER_DATE", "2020-01-01T00:00:00Z")
            .output().unwrap();
        assert!(o.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).trim().to_string()
    }
    fn commit(&self, subject: &str, files: &[(&str, &str)]) -> String {
        for (p, text) in files {
            let path = self.dir.path().join(p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        self.git(&["add", "-A"]);
        self.git(&["commit", "-qm", subject]);
        self.git(&["rev-parse", "HEAD"])
    }
    fn path(&self) -> &Path {
        self.dir.path()
    }
}

const GO_MOD: &str = "module example.invalid/m\n\ngo 1.26\n";

/// main: base, a squash slice PR (#12, a candidate), an ordinary PR (#13,
/// a candidate but not a slice), a slice PR without a test change (#15),
/// and a merge-commit slice PR (#14) whose branch holds one candidate.
fn fixture() -> (Repo, BTreeMap<&'static str, String>) {
    let r = Repo::new();
    let mut s = BTreeMap::new();
    s.insert("base", r.commit("base", &[("go.mod", GO_MOD), ("a/a.go", "package a\n"), ("ontology.json", r#"{"signals": {"CS-16": 3}, "score": 9}"#)]));
    s.insert("s12", r.commit("a: add Foo (slice S9) (#12)", &[("a/a.go", "package a\n\nfunc Foo() int { return 1 }\n"),
        ("a/a_test.go", "package a\n"), ("ontology.json", r#"{"signals": {"CS-16": 1}, "score": 5}"#)]));
    s.insert("o13", r.commit("a: add Bar (#13)", &[("a/b.go", "package a\n\nfunc Bar() {}\n"), ("a/b_test.go", "package a\n")]));
    s.insert("d15", r.commit("Slice D1: docs (#15)", &[("README.md", "docs\n")]));
    r.git(&["checkout", "-q", "-b", "feature"]);
    s.insert("b14", r.commit("b: add Baz", &[("b/b.go", "package b\n\nfunc Baz() {}\n"), ("b/b_test.go", "package b\n")]));
    r.git(&["checkout", "-q", "main"]);
    r.git(&["merge", "-q", "--no-ff", "-m", "Merge pull request #14 from x/feature", "feature"]);
    s.insert("m14", r.git(&["rev-parse", "HEAD"]));
    (r, s)
}

fn prs(s: &BTreeMap<&str, String>) -> Vec<Pr> {
    vec![
        pr(14, "b: add Baz", "## Slice\n\nM2\n\n## Proof\n\nok"),
        pr(15, "Slice D1: docs", ""),
        pr(13, "a: add Bar", "## Summary"),
        Pr { merge_commit: Some(s["s12"].clone()), ..pr(12, "a: add Foo (slice S9)", "") },
        pr(99, "Slice Z1: merged elsewhere", ""),
    ]
}

#[test]
fn slice_prs_select_their_own_candidates_only() {
    let (r, s) = fixture();
    let scanned = crate::scan_rev(r.path(), "main", "2000-01-01", &crate::csf::MineCsf::default()).unwrap();
    assert_eq!(scanned.0.len(), 3, "the scan sees every candidate: {:?}", scanned.0.iter().map(|c| &c.subject).collect::<Vec<_>>());
    let sel = select(r.path(), "main", "2000-01-01", &prs(&s), scanned).unwrap();
    assert_eq!(sel.prs_seen, 4, "#99 has no commit on main");
    let by_pr: BTreeMap<u64, &SliceRecord> = sel.slices.iter().map(|x| (x.tag.pr, x)).collect();
    assert_eq!(by_pr.keys().copied().collect::<Vec<_>>(), [12, 14, 15], "#13 is not a slice");
    assert_eq!(by_pr[&12].tag.id, "S9");
    assert_eq!(by_pr[&12].candidates, [s["s12"].clone()]);
    assert_eq!(by_pr[&14].tag.id, "M2");
    assert_eq!(by_pr[&14].commits, [s["b14"].clone()], "a merge PR contributes its branch commits");
    assert_eq!(by_pr[&14].candidates, [s["b14"].clone()]);
    assert_eq!(by_pr[&15].no_test_change, [s["d15"].clone()]);
    assert!(by_pr[&15].candidates.is_empty());
    let picked: Vec<(&str, &str)> = sel.candidates.iter().map(|(c, t)| (c.sha.as_str(), t.id.as_str())).collect();
    assert_eq!(picked, [(s["b14"].as_str(), "M2"), (s["s12"].as_str(), "S9")]);
}

#[test]
fn the_window_drops_older_slice_commits() {
    let (r, s) = fixture();
    let scanned = crate::scan_rev(r.path(), "main", "2025-01-01", &crate::csf::MineCsf::default()).unwrap();
    let sel = select(r.path(), "main", "2025-01-01", &prs(&s), scanned).unwrap();
    assert_eq!((sel.prs_seen, sel.slices.len(), sel.candidates.len()), (0, 0, 0));
}

#[test]
fn tagging_adds_slice_and_the_ontology_signals_the_fix_moved() {
    let (r, s) = fixture();
    let scanned = crate::scan_rev(r.path(), "main", "2000-01-01", &crate::csf::MineCsf::default()).unwrap();
    let sel = select(r.path(), "main", "2000-01-01", &prs(&s), scanned).unwrap();
    let (cand, tag) = sel.candidates.iter().find(|(c, _)| c.sha == s["s12"]).unwrap();
    let out = tempfile::tempdir().unwrap();
    assert_eq!(tag_task(out.path(), r.path(), cand, tag, None).unwrap(), None, "no record yet");
    let dir = out.path().join(&cand.sha[..12]);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("task.json"), json!({"sha": cand.sha, "valid": true, "reason": "FAIL_TO_PASS"}).to_string()).unwrap();
    // The scorer sees a git checkout of each side, as `candace ontology score` needs.
    let scorer = "test \"$(git rev-parse HEAD)\" = \"$(git log -1 --format=%H)\" && cat {tree}/ontology.json";
    assert_eq!(tag_task(out.path(), r.path(), cand, tag, Some(scorer)).unwrap(), Some(true));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("task.json")).unwrap()).unwrap();
    assert_eq!(v["reason"], "FAIL_TO_PASS", "other fields are kept");
    assert_eq!(v["slice"]["id"], "S9");
    assert_eq!(v["slice"]["pr"], 12);
    assert_eq!(v["slice"]["ontology"]["status"], "measured");
    assert_eq!(v["slice"]["ontology"]["moved"], json!({"CS-16": -2.0, "score": -4.0}));
    assert_eq!(v["slice"]["ontology"]["after"], json!({"CS-16": 1.0, "score": 5.0}));
    let head = std::process::Command::new("sh").arg("-c").arg("git rev-parse HEAD").current_dir(
        checkout(r.path(), &cand.sha).unwrap().path().join("t")).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&head.stdout).trim(), cand.sha);
    // A measured ontology survives a re-run without a scorer.
    tag_task(out.path(), r.path(), cand, tag, None).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("task.json")).unwrap()).unwrap();
    assert_eq!(v["slice"]["ontology"]["status"], "measured");
}

#[test]
fn ontology_says_why_it_was_not_measured() {
    let (r, s) = fixture();
    let scanned = crate::scan_rev(r.path(), "main", "2000-01-01", &crate::csf::MineCsf::default()).unwrap();
    let sel = select(r.path(), "main", "2000-01-01", &prs(&s), scanned).unwrap();
    let (cand, _) = &sel.candidates[0];
    assert_eq!(ontology(None, r.path(), cand, true)["status"], "unavailable");
    assert_eq!(ontology(Some("true"), r.path(), cand, false)["status"], "skipped");
    let e = ontology(Some("exit 4"), r.path(), cand, true);
    assert_eq!(e["status"], "error");
    assert!(e["reason"].as_str().unwrap().contains("parent tree"), "{e}");
    // The b14 branch commit has no ontology.json: the scorer fails, honestly.
    assert_eq!(ontology(Some("cat ontology.jsn"), r.path(), cand, true)["status"], "error");
}

#[test]
fn a_dry_run_writes_the_slice_list_through_the_registry() {
    let (r, s) = fixture();
    let out = tempfile::tempdir().unwrap();
    let prs_file = out.path().join("prs.json");
    let gh: Vec<Value> = prs(&s).iter().map(|p| json!({"number": p.number, "title": p.title, "body": p.body,
        "mergeCommit": p.merge_commit.as_ref().map(|m| json!({"oid": m}))})).collect();
    std::fs::write(&prs_file, Value::Array(gh).to_string()).unwrap();
    let dest = out.path().join("mined");
    let summary = crate::miner::run("slices", json!({"repo": r.path(), "out": dest, "rev": "main",
        "since": "2000-01-01", "prs": prs_file, "dry_run": true})).unwrap();
    assert_eq!(summary["prs_seen"], 4);
    assert_eq!(summary["slice_prs"], 3);
    assert_eq!(summary["candidates"], 2);
    assert_eq!(summary["valid"], 0);
    let lines = std::fs::read_to_string(dest.join("slices.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 3);
    assert!(dest.join("slices-summary.json").is_file() && dest.join("rejected.jsonl").is_file());
    assert!(!dest.join(&s["s12"][..12]).exists(), "a dry run validates nothing");
}

#[test]
fn misuse_is_refused() {
    let (r, _) = fixture();
    let inside = r.path().join("out");
    let err = crate::miner::run("slices", json!({"repo": r.path(), "out": inside, "prs": "/nonexistent"})).unwrap_err();
    assert!(format!("{err}").contains("inside the git work tree"), "{err}");
    assert!(!inside.exists());
    let err = crate::miner::run("slices", json!({"repo": r.path(), "out": "/nonexistent-rrsi-t", "sincee": "x"})).unwrap_err();
    assert!(format!("{err}").contains("sincee"), "{err}");
    let out = tempfile::tempdir().unwrap();
    let err = crate::miner::run("slices", json!({"repo": r.path(), "out": out.path().join("o"),
                                                  "prs": "/nonexistent-prs.json"})).unwrap_err();
    assert!(format!("{err:#}").contains("prs"), "{err:#}");
}
