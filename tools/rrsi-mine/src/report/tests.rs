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

//! Regression tests over synthetic task directories (made-up shas and
//! subjects only).

use super::guard::enclosing_work_tree;
use super::load::{area, load_tasks, Split};
use super::render::{embed_json, iso_utc, render};
use super::split::{assign, examples, summarize};
use serde_json::{json, Value};
use std::path::Path;

fn sha(n: u32) -> String {
    format!("{n:012x}{}", "e".repeat(28))
}

/// Write a task dir; `fairness` maps stage -> pass.
fn write_task(root: &Path, n: u32, valid: bool, reason: &str, churn: u64,
              fairness: &[(&str, bool)], retry: bool) {
    let s = sha(n);
    let dir = root.join(&s[..12]);
    std::fs::create_dir_all(&dir).unwrap();
    let meta = json!({
        "sha": s, "parent": sha(n + 10_000), "subject": format!("task {n} </script><b>x</b>"),
        "body": "", "module_root": "mod", "packages": ["./services/alpha/inner"],
        "src_files": ["mod/services/alpha/inner/a.go"], "test_files": ["mod/services/alpha/inner/a_test.go"],
        "src_churn": churn, "valid": valid, "reason": reason,
        "parent_outcome": if valid { "BuildFail" } else { "Pass" }, "commit_outcome": "Pass", "seconds": 1.0,
    });
    let name = if retry { "retry.json" } else { "task.json" };
    std::fs::write(dir.join(name), meta.to_string()).unwrap();
    std::fs::write(dir.join("src.patch"), "+func Fix() {} // </script><script>alert(1)</script>\n").unwrap();
    std::fs::write(dir.join("tests.patch"), "+func TestFix(t *testing.T) {}\n").unwrap();
    std::fs::write(dir.join("parent.log"), "undefined: Fix\n").unwrap();
    std::fs::write(dir.join("commit.log"), "ok\n").unwrap();
    if !fairness.is_empty() {
        std::fs::create_dir_all(dir.join("fairness")).unwrap();
        for (stage, pass) in fairness {
            std::fs::write(dir.join("fairness").join(format!("{stage}.json")),
                           json!({"stage": stage, "pass": pass, "reason": format!("{stage} said {pass}")}).to_string())
                .unwrap();
        }
    }
}

fn dated(tasks: &mut [super::load::Task], ts_of: impl Fn(&str) -> Option<i64>) {
    for t in tasks {
        t.ts = ts_of(&t.sha);
        t.date = t.ts.map(iso_utc);
    }
}

#[test]
fn newest_n_are_held_out_ties_by_sha_and_not_ready_are_excluded() {
    let d = tempfile::tempdir().unwrap();
    for n in 1..=6 {
        write_task(d.path(), n, true, "ok", 10 * n as u64, &[], false);
    }
    write_task(d.path(), 7, false, "tests already pass on parent", 5, &[], false);
    write_task(d.path(), 8, false, "infra: a run could not assemble its build inputs", 5, &[], true);
    let mut tasks = load_tasks(d.path()).unwrap();
    // Task 6 is newest; 4 and 5 tie; 1 has no date (counts as oldest).
    dated(&mut tasks, |s| match u32::from_str_radix(&s[..12], 16).unwrap() {
        1 => None,
        4 | 5 => Some(4_000),
        n => Some(n as i64 * 1_000),
    });
    let sp = assign(&mut tasks, 2);
    // Newest is 6, then the 4/5 tie broken by sha ascending: 4.
    assert_eq!(sp.heldout, vec![sha(4)[..12].to_string(), sha(6)[..12].to_string()]);
    assert_eq!(sp.evolve, vec![sha(1)[..12].to_string(), sha(2)[..12].to_string(),
                               sha(3)[..12].to_string(), sha(5)[..12].to_string()]);
    assert_eq!(sp.excluded.get(&sha(7)[..12]).map(String::as_str), Some("tests already pass on parent"));
    assert!(sp.excluded[&sha(8)[..12]].starts_with("pending retry: infra"));
    // Deterministic: same input, same split.
    let mut again = load_tasks(d.path()).unwrap();
    dated(&mut again, |s| match u32::from_str_radix(&s[..12], 16).unwrap() {
        1 => None, 4 | 5 => Some(4_000), n => Some(n as i64 * 1_000) });
    assert_eq!(assign(&mut again, 2), sp);
}

#[test]
fn heldout_larger_than_ready_holds_everything_out() {
    let d = tempfile::tempdir().unwrap();
    write_task(d.path(), 1, true, "ok", 1, &[], false);
    let mut tasks = load_tasks(d.path()).unwrap();
    let sp = assign(&mut tasks, 10);
    assert_eq!((sp.evolve.len(), sp.heldout.len()), (0, 1));
}

#[test]
fn fairness_gate_decides_readiness_and_funnel_adds_up() {
    let d = tempfile::tempdir().unwrap();
    write_task(d.path(), 1, true, "ok", 10, &[("flake", true), ("gate", true)], false);
    write_task(d.path(), 2, true, "ok", 20, &[("flake", true), ("api", false), ("gate", false)], false);
    write_task(d.path(), 3, true, "ok", 30, &[("gate", false)], false);
    write_task(d.path(), 4, true, "ok", 40, &[], false); // fairness not run: valid is enough
    write_task(d.path(), 5, false, "tests fail on the commit", 50, &[], false);
    write_task(d.path(), 6, false, "infra: x", 60, &[], true);
    let mut tasks = load_tasks(d.path()).unwrap();
    dated(&mut tasks, |s| Some(u32::from_str_radix(&s[..12], 16).unwrap() as i64));
    let sp = assign(&mut tasks, 1);
    assert_eq!(sp.excluded[&sha(2)[..12]], "fairness/api: api said false");
    assert_eq!(sp.excluded[&sha(3)[..12]], "fairness/gate: gate said false");
    assert_eq!(sp.heldout, vec![sha(4)[..12].to_string()]);
    assert_eq!(sp.evolve, vec![sha(1)[..12].to_string()]);

    let s = summarize(&tasks);
    assert_eq!((s.candidates, s.valid, s.exam_ready, s.evolve, s.heldout), (6, 4, 2, 1, 1));
    assert_eq!((s.pending_retry, s.fairness_ran), (1, 3));
    assert_eq!(s.candidates, s.valid + s.excluded_by_mine);
    assert_eq!(s.valid, s.exam_ready + s.excluded_by_fairness);
    assert_eq!(s.exam_ready, s.evolve + s.heldout);
    assert_eq!(s.excluded_by_mine + s.excluded_by_fairness, sp.excluded.len());
    assert_eq!(s.cutoff, Some(iso_utc(4)));
    let count = |l: &str| s.rejections.iter().find(|r| r.0 == l).map(|r| r.2);
    assert_eq!(count("fairness/api"), Some(1));
    assert_eq!(count("fairness/gate"), Some(2));
    assert_eq!(count("tests fail on the commit"), Some(1));
    assert_eq!(count("pending retry: infra: x"), Some(1));
}

#[test]
fn missing_fairness_files_render_as_not_run() {
    let d = tempfile::tempdir().unwrap();
    write_task(d.path(), 1, true, "ok", 10, &[], false);
    write_task(d.path(), 2, true, "ok", 30, &[], false);
    write_task(d.path(), 3, true, "ok", 20, &[], false);
    let mut tasks = load_tasks(d.path()).unwrap();
    assert!(tasks.iter().all(|t| t.fairness.is_empty() && t.instruction.is_none()));
    let sp = assign(&mut tasks, 1); // undated: sha order
    assert_eq!(sp.heldout, vec![sha(1)[..12].to_string()]);
    let ex = examples(&tasks);
    assert_eq!(ex["evolve"], vec![sha(3)[..12].to_string(), sha(2)[..12].to_string()]);
    let html = render(&tasks, &sp, 1, None, "2026-01-01T00:00:00Z").unwrap();
    let data = embedded(&html);
    assert_eq!(data["summary"]["fairness_ran"], 0);
    assert_eq!(data["exam_lines"], Value::Null);
    assert_eq!(data["has_dates"], false);
}

/// The JSON between the data script tags, parsed.
fn embedded(html: &str) -> Value {
    let start = html.find("<script id=\"rrsi-data\" type=\"application/json\">").unwrap();
    let rest = &html[start..];
    let body = &rest[rest.find('>').unwrap() + 1..rest.find("</script>").unwrap()];
    serde_json::from_str(body).unwrap()
}

#[test]
fn embedded_json_cannot_close_its_script_element() {
    let v = json!({"s": "a</script><script>alert(1)</script><!-- & \u{2028}"});
    let e = embed_json(&v);
    assert!(!e.contains('<') && !e.contains('>') && !e.contains('&'));
    assert!(!e.contains('\u{2028}'));
    assert_eq!(serde_json::from_str::<Value>(&e).unwrap(), v, "escapes round-trip");

    let d = tempfile::tempdir().unwrap();
    write_task(d.path(), 1, true, "ok", 10, &[], false);
    let mut tasks = load_tasks(d.path()).unwrap();
    let sp = assign(&mut tasks, 0);
    let html = render(&tasks, &sp, 0, Some(3), "2026-01-01T00:00:00Z").unwrap();
    // Exactly the template's own closing tags: data never adds one.
    let template = include_str!("../../report/report.html");
    assert_eq!(html.matches("</script>").count(), template.matches("</script>").count());
    let data = embedded(&html);
    assert_eq!(data["tasks"][0]["subject"], "task 1 </script><b>x</b>");
    assert!(data["tasks"][0]["src_patch"].as_str().unwrap().contains("</script><script>"));
    assert_eq!(data["exam_lines"], 3);
    assert_eq!(data["tasks"][0]["split"], "evolve");
}

#[test]
fn refuses_output_inside_a_work_tree() {
    let d = tempfile::tempdir().unwrap();
    let repo = d.path().join("checkout");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let hit = enclosing_work_tree(&repo.join("sub/not-yet/report.html")).unwrap();
    assert_eq!(hit, Some(repo.canonicalize().unwrap()));
    // A worktree or submodule has a .git file, not a directory.
    let wt = d.path().join("wt");
    std::fs::create_dir_all(&wt).unwrap();
    std::fs::write(wt.join(".git"), "gitdir: elsewhere\n").unwrap();
    assert!(enclosing_work_tree(&wt.join("r.html")).unwrap().is_some());
    let outside = d.path().join("plain");
    std::fs::create_dir_all(&outside).unwrap();
    if enclosing_work_tree(d.path()).unwrap().is_none() {
        assert_eq!(enclosing_work_tree(&outside.join("r.html")).unwrap(), None);
    }
}

#[test]
fn area_takes_three_repository_segments() {
    assert_eq!(area("mod", &["./services/alpha/inner".into()]), "mod/services/alpha");
    assert_eq!(area("tool/scaler", &["./".into()]), "tool/scaler");
    assert_eq!(area("", &[".".into()]), "(root)");
    assert_eq!(area("a/b/c/d", &["./x".into()]), "a/b/c");
}

#[test]
fn split_values_serialize_lowercase_and_dates_format() {
    assert_eq!(serde_json::to_value(Split::Heldout).unwrap(), "heldout");
    assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
    assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00Z");
    assert_eq!(iso_utc(1_790_000_000), "2026-09-21T14:13:20Z");
}
