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
    let html = render_page(&tasks, &sp, d.path(), 1, None);
    let data = embedded(&html);
    assert_eq!(data["summary"]["fairness_ran"], 0);
    assert_eq!(data["exam_lines"], Value::Null);
    assert_eq!(data["has_dates"], false);
}

fn render_page(tasks: &[super::load::Task], sp: &super::split::Splits, dir: &Path, heldout: usize,
               exam_lines: Option<usize>) -> String {
    let h = super::health::health(tasks);
    render(&super::render::Inputs { tasks, splits: sp, health: &h, start: &start_of(tasks, dir), page: &page_of(tasks),
                                    heldout, exam_lines }, "2026-01-01T00:00:00Z").unwrap()
}

fn page_of(tasks: &[super::load::Task]) -> super::figures::Page {
    let prov = super::figures::Provenance { since: "2026-01-01", go_image: "golang:example", heldout: 1 };
    super::figures::build(tasks, &summarize(tasks), &super::health::health(tasks), &prov)
}

fn start_of(tasks: &[super::load::Task], dir: &Path) -> super::explain::Start {
    super::explain::start(tasks, dir, &summarize(tasks), &super::health::health(tasks))
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
    let html = render_page(&tasks, &sp, d.path(), 0, Some(3));
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

mod health {
    use super::super::health::{component, health, module_tree, quantile, Status};
    use super::super::load::{area, Split, Task};
    use super::super::render::iso_utc;
    use std::collections::BTreeMap;

    /// A task built in memory: (split, module_root, package, churn, parent outcome, day).
    fn t(split: Split, root: &str, pkg: &str, churn: u64, parent: &str, day: i64) -> Task {
        let n = churn * 1000 + day as u64;
        Task {
            sha12: format!("{n:012x}"), sha: format!("{n:040x}"), subject: String::new(), body: String::new(),
            module_root: root.into(), packages: vec![pkg.into()], src_files: vec![], test_files: vec![],
            src_churn: churn, valid: true, reason: "ok".into(), pending_retry: false,
            parent_outcome: Some(parent.into()), commit_outcome: Some("Pass".into()),
            fairness: BTreeMap::new(), instruction: None, src_patch: String::new(), tests_patch: String::new(),
            parent_log_tail: String::new(), commit_log_tail: String::new(),
            area: area(root, &[pkg.to_string()]), date: Some(iso_utc(day * 86_400)), ts: Some(day * 86_400),
            split, excluded_reason: None, seconds: 1.0, written_at: None,
        }
    }

    fn status(h: &super::super::health::Health, name: &str) -> (Status, String) {
        let c = h.checks.iter().find(|c| c.id == name).unwrap();
        (match c.status { Status::Pass => Status::Pass, Status::Warn => Status::Warn, Status::NotApplicable => Status::NotApplicable },
         c.detail.clone())
    }

    /// Balanced: same trees, areas spread, similar churn and outcomes, wide dates.
    fn balanced() -> Vec<Task> {
        let mut v = Vec::new();
        for (i, (root, pkg)) in [("go", "./services/a/x"), ("go", "./services/b/x"), ("lib", "./pkg/c"),
                                 ("go", "./services/d/x")].iter().enumerate() {
            let i = i as i64;
            v.push(t(Split::Evolve, root, pkg, 40 + 10 * i as u64, if i % 2 == 0 { "BuildFail" } else { "TestFail" }, i));
            v.push(t(Split::Heldout, root, pkg, 45 + 10 * i as u64, if i % 2 == 0 { "BuildFail" } else { "TestFail" }, 30 + 5 * i));
        }
        v.push(t(Split::Excluded, "zzz", "./q", 9999, "Pass", 99));
        v
    }

    #[test]
    fn quantiles_interpolate_like_numpy() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[3.0], 0.5), Some(3.0));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), Some(2.5));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.25), Some(2.0));
        assert_eq!(quantile(&[10.0, 20.0, 30.0, 40.0], 0.75), Some(32.5));
        assert_eq!(module_tree("go/services/x"), "go");
        assert_eq!(component("svc", &["./services/warden/election".into()]), "warden");
        assert_eq!(component("go", &["./app/warden/cmd".into()]), "warden");
        assert_eq!(component("tool/scaler", &["./".into()]), "scaler");
        assert_eq!(component("mod", &["./internal".into()]), "mod");
        assert_eq!(module_tree(""), "(root)");
    }

    #[test]
    fn a_balanced_split_passes_every_check() {
        let h = health(&balanced());
        assert_eq!((h.evolve.n, h.heldout.n), (4, 4), "excluded tasks are not measured");
        assert_eq!(h.evolve.churn_median, Some(55.0));
        assert_eq!(h.heldout.churn_median, Some(60.0));
        assert_eq!((h.evolve.churn_q1, h.evolve.churn_q3), (Some(47.5), Some(62.5)));
        assert_eq!(h.evolve.module_tree["go"], 0.75);
        assert_eq!(h.heldout.module_tree["lib"], 0.25);
        assert_eq!(h.heldout.outcome["BuildFail\u{2192}Pass"], 0.5);
        assert_eq!(h.heldout.span_days, Some(15.0));
        assert!(h.checks.iter().all(|c| c.status == Status::Pass), "{:#?}", h.checks);
        assert_eq!(h.warnings, 0);
        assert!(h.verdict.starts_with("6 of 6 checks pass: the final exam (held-out) looks like the practice set"), "{}", h.verdict);
        for c in &h.checks {
            assert!(!c.name.is_empty() && !c.meaning.is_empty() && !c.why.is_empty() && !c.action.is_empty(), "{c:?}");
        }
        let area_sum: f64 = h.heldout.area_top.values().sum();
        assert!((area_sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn each_threshold_fires_on_its_own_imbalance() {
        // Held-out churn doubled: only the churn check warns.
        let mut v = balanced();
        v.iter_mut().filter(|x| x.split == Split::Heldout).for_each(|x| x.src_churn *= 2);
        let h = health(&v);
        let (s, d) = status(&h, "churn");
        assert_eq!(s, Status::Warn);
        assert_eq!(d, "evolve 55, held-out 120 \u{2192} 2.18x, outside 0.67\u{2013}1.5x");
        assert_eq!(h.warnings, 1);
        let c = h.checks.iter().find(|c| c.id == "churn").unwrap();
        assert_eq!(c.name, "Task size (median lines changed)");
        assert_eq!(c.meaning, "The median final-exam fix changed 120 lines (95% CI 90\u{2013}150; n=4 tasks); \
                               the median practice fix changed 55 lines (95% CI 40\u{2013}70; n=4 tasks) \u{2014} \
                               final-exam fixes are 2.2\u{d7} bigger.");
        assert_eq!(h.verdict, "1 of 6 checks warn: the final exam (held-out) is harder than the practice set (evolve), \
                               so its score would be hard to interpret. Suggested fix: hold out more tasks, or mine \
                               further back so both sets draw from the same code.");

        // Three of four held-out in one area and one tree: concentration and tree mix warn.
        let mut v = balanced();
        for x in v.iter_mut().filter(|x| x.split == Split::Heldout).take(3) {
            x.module_root = "newtree".into();
            x.area = "newtree/services/w".into();
        }
        let h = health(&v);
        let (s, d) = status(&h, "area");
        assert_eq!(s, Status::Warn);
        assert_eq!(d, "largest held-out folder newtree/services/w: 3 of 4 (75%), above max 40%");
        assert_eq!(h.top_areas[0], "newtree/services/w", "a held-out-only area ranks by its held-out share");
        assert!(h.verdict.starts_with("2 of 6 checks warn: the final exam (held-out) is narrower than the practice set"), "{}", h.verdict);
        let (s, d) = status(&h, "tree");
        assert_eq!(s, Status::Warn);
        assert_eq!(d, "newtree: evolve 0%, held-out 75% \u{2192} 75 points, above max 30");
        assert_eq!(status(&h, "churn").0, Status::Pass);

        // Same component under different areas and trees: only the component check sees it.
        let mut v = balanced();
        let pkgs = ["./services/w/a", "./app/w/cmd", "./pkg/w"];
        for (x, p) in v.iter_mut().filter(|x| x.split == Split::Heldout).take(3).zip(pkgs) {
            x.packages = vec![p.to_string()];
            x.area = area(&x.module_root, &x.packages);
        }
        let h = health(&v);
        assert_eq!(status(&h, "area").0, Status::Pass);
        let (s, d) = status(&h, "component");
        assert_eq!(s, Status::Warn);
        assert_eq!(d, "largest held-out subsystem w: 3 of 4 (75%), above max 40%");

        // All held-out TestFail: outcome mix warns (50 points).
        let mut v = balanced();
        v.iter_mut().filter(|x| x.split == Split::Heldout).for_each(|x| x.parent_outcome = Some("TestFail".into()));
        assert_eq!(status(&health(&v), "outcome").0, Status::Warn);

        // Held-out within three days: span warns.
        let mut v = balanced();
        for (i, x) in v.iter_mut().filter(|x| x.split == Split::Heldout).enumerate() {
            x.ts = Some(i as i64 * 86_400);
            x.date = Some(iso_utc(i as i64 * 86_400));
        }
        let (s, d) = status(&health(&v), "span");
        assert_eq!(s, Status::Warn);
        assert_eq!(d, "1970-01-01 .. 1970-01-04 = 3.0 days, below min 7");
        let h = health(&v);
        let c = h.checks.iter().find(|c| c.id == "span").unwrap();
        assert!(c.meaning.starts_with("All 4 final-exam commits were made within 3 days (1970-01-01 to 1970-01-04, UTC)"), "{}", c.meaning);
        assert!(h.verdict.contains("covers only a very short stretch of work") && h.verdict.contains("newest ~2 weeks"), "{}", h.verdict);
    }

    #[test]
    fn plain_names_and_old_new_layout() {
        let mut v = balanced();
        // "lib" tasks are the oldest by median date, "go" the newest.
        for x in v.iter_mut() {
            if x.module_root == "lib" { x.ts = Some(-1_000_000); }
        }
        let h = health(&v);
        let names: Vec<&str> = h.checks.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Task size (median lines changed)", "Biggest folder share of the final exam",
                           "Biggest subsystem share of the final exam", "Old lib/ vs new go/ code layout",
                           "How tests failed before the fix (build error vs failing test)",
                           "How many days of work the final exam covers"]);
        assert_eq!(super::super::health::span_words(0.1), "2.4 hours");
        assert_eq!(super::super::health::span_words(1.0), "1 day");
    }

    #[test]
    fn empty_or_undated_splits_are_not_applicable() {
        let mut v = balanced();
        v.retain(|x| x.split != Split::Heldout);
        let h = health(&v);
        assert!(h.checks.iter().all(|c| c.status == Status::NotApplicable));
        let mut v = balanced();
        v.iter_mut().for_each(|x| { x.ts = None; x.date = None; });
        assert_eq!(status(&health(&v), "span").0, Status::NotApplicable);
    }
}

mod start_here {
    use super::super::explain::{failure_excerpt, ok_line, strip_ansi, test_excerpt};
    use super::super::render::render;
    use super::super::split::assign;
    use super::{embedded, load_tasks, start_of, write_task};

    fn page(dir: &std::path::Path) -> (String, serde_json::Value) {
        let mut tasks = load_tasks(dir).unwrap();
        let sp = assign(&mut tasks, 1);
        let html = render(&super::super::render::Inputs { tasks: &tasks, splits: &sp, health: &super::super::health::health(&tasks), start: &start_of(&tasks, dir), page: &super::page_of(&tasks), heldout: 1, exam_lines: None }, "2026-01-01T00:00:00Z").unwrap();
        let data = embedded(&html);
        (html, data)
    }

    /// The worked example's spoiler is a closed <details> and no script opens it.
    fn fix_stays_collapsed(html: &str) {
        assert!(html.contains(r#"<details class="spoiler hidden" id="we-fix"><summary>SPOILER: the reference fix</summary>"#));
        assert!(!html.contains("<details open") && !html.contains(" open>") && !html.contains(".open ="));
        assert!(!html.contains("setAttribute(\"open\"") && !html.contains("toggleAttribute(\"open\""));
    }

    #[test]
    fn renders_without_fairness_files() {
        let d = tempfile::tempdir().unwrap();
        write_task(d.path(), 1, true, "ok", 20, &[], false);
        write_task(d.path(), 2, true, "ok", 30, &[], false);
        write_task(d.path(), 3, false, "tests already pass on parent", 5, &[], false);
        let (html, data) = page(d.path());
        let start = html.find(r#"id="start-here""#).unwrap();
        assert!(start < html.find(r#"id="tiles""#).unwrap() && start < html.find(r#"data-fig="funnel""#).unwrap(),
                "Start here comes before the counts and the funnel");
        let st = &data["start"];
        assert!(st["worked"]["instruction"].is_null());
        assert_eq!(st["worked"]["parent_failure"], "undefined: Fix");
        assert_eq!(st["already_passes"]["subject"], "task 3 </script><b>x</b>");
        assert_eq!(st["missing_name"]["detail"], "undefined: Fix");
        assert!(st["leak"].is_null() && st["flaky"].is_null());
        let stage = |i: usize| (st["pipeline"][i]["name"].as_str().unwrap().to_string(),
                                st["pipeline"][i]["status"].as_str().unwrap().to_string());
        assert_eq!(stage(2), ("fairness".into(), "next".into()));
        assert_eq!(stage(4).1, "not started");
        fix_stays_collapsed(&html);
    }

    #[test]
    fn renders_with_fairness_files() {
        let d = tempfile::tempdir().unwrap();
        let all = [("flake", true), ("api", true), ("describe", true), ("probe", true), ("specificity", true), ("gate", true)];
        write_task(d.path(), 1, true, "ok", 20, &all, false);
        write_task(d.path(), 2, true, "ok", 30, &[("flake", false), ("gate", false)], false);
        write_task(d.path(), 3, true, "ok", 40, &[("describe", false), ("gate", false)], false);
        let t1 = d.path().join(format!("{:012x}", 1));
        std::fs::write(t1.join("instruction.md"), "Make Fix exist.\n").unwrap();
        std::fs::write(t1.join("fairness/api.json"), serde_json::json!({"stage": "api", "pass": true, "reason": "ok",
            "required": [{"symbol": "Fix", "kind": "func", "signature": "func Fix()"}]}).to_string()).unwrap();
        let (html, data) = page(d.path());
        let st = &data["start"];
        assert_eq!(st["worked"]["instruction"], "Make Fix exist.\n");
        assert_eq!(st["missing_name"]["detail"], "required by the hidden tests: func Fix()");
        assert_eq!(st["flaky"]["detail"], "flake said false");
        assert_eq!(st["leak"]["detail"], "describe said false");
        assert_eq!(st["pipeline"][2]["status"], "done");
        assert_eq!(st["pipeline"][4]["status"], "next");
        fix_stays_collapsed(&html);
    }

    #[test]
    fn log_and_patch_excerpts() {
        assert_eq!(strip_ansi("\u{1b}[38;5;9m[FAILED]\u{1b}[0m x"), "[FAILED] x");
        let ginkgo = "\u{1b}[1m[It] names the product\u{1b}[0m\n  [FAILED] Expected\n  <string>: A\nto equal\n  <string>: B\n------\nFAIL";
        assert_eq!(failure_excerpt(ginkgo), "  [FAILED] Expected\n  <string>: A\nto equal\n  <string>: B");
        assert_eq!(failure_excerpt("x\n--- FAIL: TestA (0s)\n    a_test.go:9: got 1\nFAIL"),
                   "--- FAIL: TestA (0s)\n    a_test.go:9: got 1\nFAIL");
        assert_eq!(ok_line("$ go test\nexit=0\nok  \tmod/pkg\t0.1s\n"), "ok  \tmod/pkg\t0.1s");
        let patch = "+++ b/p/a_test.go\n+package p\n+func TestOther(t *testing.T) {}\n+var _ = It(\"names the product\", func() {\n+  Expect(x).To(Equal(1))\n+})\n";
        let (f, ex) = test_excerpt(patch, "[It] names the product\n");
        assert_eq!(f, "p/a_test.go");
        assert!(ex.starts_with("var _ = It(\"names the product\""), "{ex}");
        let (_, ex) = test_excerpt(patch, "--- FAIL: TestOther (0s)\n");
        assert!(ex.starts_with("func TestOther("));
    }
}

#[test]
fn health_table_renders_five_columns_and_a_verdict() {
    let d = tempfile::tempdir().unwrap();
    for n in 1..=4 {
        write_task(d.path(), n, true, "ok", 10 * n as u64, &[], false);
    }
    let mut tasks = load_tasks(d.path()).unwrap();
    dated(&mut tasks, |s| Some(u32::from_str_radix(&s[..12], 16).unwrap() as i64 * 86_400 * 5));
    let sp = assign(&mut tasks, 2);
    let html = render_page(&tasks, &sp, d.path(), 2, None);
    assert!(html.contains("<th>Check</th><th>Result</th><th>What it means</th><th>Why it matters</th><th>What to do if it warns</th>"));
    for field in ["c.name", "mark", "c.meaning", "c.why", "c.action", "HL.verdict"] {
        assert!(html.contains(field), "the health table script renders {field}");
    }
    assert!(html.contains("Is the final exam (held-out) the same kind of exam as the practice set (evolve)?"));
    let data = embedded(&html);
    let checks = data["health"]["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 6);
    for c in checks {
        for k in ["name", "status", "meaning", "why", "action"] {
            assert!(c[k].as_str().is_some_and(|v| !v.is_empty()), "check {} has {k}", c["id"]);
        }
    }
    assert!(data["health"]["verdict"].as_str().unwrap().contains("checks"));
    let hl = &data["start"]["headline"];
    assert_eq!(hl["kind"], "exam");
    assert!(hl["before"].as_str().unwrap().starts_with("to learn whether a change to an AI coding agent actually helps"));
    let now = hl["now"].as_str().unwrap();
    assert!(now.starts_with("one command turns your repository's git history into a verified exam \u{2014} 4 recent commits"), "{now}");
    assert!(now.contains("4 of them proven questions"), "{now}");
    let tries = hl["tries"].as_array().unwrap();
    assert_eq!(tries.len(), 7);
    assert_eq!(tries[1]["filter"], "tests already pass on parent");
    assert_eq!(tries[6]["future"], true, "the not-yet-possible item is marked");
    assert!(tries.iter().take(6).all(|t| t["future"] == false));
    assert_eq!(hl["not_shown"], "Not shown yet: whether RRSI then improves an agent on that exam.");
    assert!(html.find(r#"id="headline""#).unwrap() < html.find(r#"id="start-here""#).unwrap());
}

mod stats {
    use super::super::health::{frac, frac_ci};
    use super::super::stats::{bootstrap_median_ci, wilson};

    fn close(a: f64, b: f64) -> bool { (a - b).abs() < 5e-4 }

    #[test]
    fn wilson_matches_reference_values() {
        // Reference: Wilson 95% for 10/10 is 0.7225..1, for 5/10 0.2366..0.7634, for 0/10 0..0.2775.
        let (lo, hi) = wilson(10, 10).unwrap();
        assert!(close(lo, 0.7225) && close(hi, 1.0), "{lo} {hi}");
        let (lo, hi) = wilson(5, 10).unwrap();
        assert!(close(lo, 0.2366) && close(hi, 0.7634), "{lo} {hi}");
        let (lo, hi) = wilson(0, 10).unwrap();
        assert!(close(lo, 0.0) && close(hi, 0.2775), "{lo} {hi}");
        assert_eq!(wilson(0, 0), None);
    }

    #[test]
    fn bootstrap_median_interval_is_deterministic_and_brackets_the_median() {
        let v: Vec<f64> = (1..=21).map(f64::from).collect();
        let (lo, hi) = bootstrap_median_ci(&v).unwrap();
        assert!(lo <= 11.0 && hi >= 11.0 && lo >= 1.0 && hi <= 21.0, "{lo} {hi}");
        assert!(hi - lo < 10.0, "a 21-point sample gives a reasonably tight interval: {lo} {hi}");
        assert_eq!(bootstrap_median_ci(&v), Some((lo, hi)), "seeded: same answer every run");
        assert_eq!(bootstrap_median_ci(&[3.0]), None);
        let same = bootstrap_median_ci(&[7.0; 5]).unwrap();
        assert_eq!(same, (7.0, 7.0));
    }

    #[test]
    fn fractions_carry_their_denominators() {
        assert_eq!(frac(14, 60, "practice tasks"), "23% (14 of 60 practice tasks)");
        assert_eq!(frac_ci(10, 10, "final-exam tasks"), "100% (10 of 10 final-exam tasks; 95% CI 72\u{2013}100%)");
    }
}

mod paper_bar {
    use super::super::figures::lint;
    use super::super::split::assign;
    use super::{dated, load_tasks, page_of, write_task};
    use serde_json::{json, Value};

    fn page() -> super::super::figures::Page {
        let d = tempfile::tempdir().unwrap();
        for n in 1..=6 {
            write_task(d.path(), n, true, "ok", 10 * n as u64, &[], false);
        }
        write_task(d.path(), 7, false, "tests already pass on parent", 5, &[], false);
        let mut tasks = load_tasks(d.path()).unwrap();
        dated(&mut tasks, |s| Some(u32::from_str_radix(&s[..12], 16).unwrap() as i64 * 86_400));
        assign(&mut tasks, 2);
        page_of(&tasks)
    }

    fn spec_mut<'a>(p: &'a mut super::super::figures::Page, id: &str) -> &'a mut Value {
        p.figures.iter_mut().find(|f| f.caption.id == id).unwrap().spec.as_mut().unwrap()
    }

    #[test]
    fn the_generated_page_passes_the_lint() {
        let p = page();
        assert_eq!(lint(&p), Vec::<String>::new());
        assert!(p.figures.iter().filter(|f| f.caption.kind == "Figure").count() >= 8);
        assert!(p.figures.iter().any(|f| f.caption.kind == "Table"));
    }

    #[test]
    fn every_figure_in_the_template_has_a_numbered_caption_in_page_order() {
        let p = page();
        let html = include_str!("../../report/report.html");
        let mut positions: Vec<(usize, &str)> = Vec::new();
        for f in &p.figures {
            let tag = format!("data-fig=\"{}\"", f.caption.id);
            let at = html.find(&tag).unwrap_or_else(|| panic!("the template has no element for {}", f.caption.id));
            let close = html[at..].find("</figure>").unwrap();
            assert!(html[at..at + close].contains("<figcaption>"), "{} has a figcaption", f.caption.id);
            positions.push((at, f.caption.id));
        }
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(sorted, positions, "figures and tables are numbered in the order they appear");
        let markup = &html[..html.find("<script id=\"rrsi-data\"").unwrap()];
        assert_eq!(markup.matches("data-fig=\"").count(), p.figures.len(), "no template figure lacks a caption");
        let mut markers: Vec<&str> = html.lines().filter(|l| l.starts_with("// ---- ")).collect();
        let n = markers.len();
        markers.sort();
        markers.dedup();
        assert_eq!(markers.len(), n, "no script section is duplicated");
    }

    #[test]
    fn the_lint_catches_each_kind_of_regression() {
        let mut p = page();
        spec_mut(&mut p, "funnel")["encoding"]["x"].as_object_mut().unwrap().remove("title");
        assert!(lint(&p).iter().any(|e| e.contains("(funnel): channel x has no title")), "{:?}", lint(&p));

        let mut p = page();
        spec_mut(&mut p, "coverage")["encoding"]["x"]["title"] = json!("Count");
        assert!(lint(&p).iter().any(|e| e.contains("(coverage): axis x title \"Count\" names no unit")), "{:?}", lint(&p));

        let mut p = page();
        spec_mut(&mut p, "tree")["layer"][0]["encoding"]["color"]["title"] = json!("");
        assert!(lint(&p).iter().any(|e| e.contains("(tree): channel color has no title")), "{:?}", lint(&p));

        let mut p = page();
        p.figures[3].caption.takeaway.clear();
        assert!(lint(&p).iter().any(|e| e.contains("caption lacks takeaway")));

        let mut p = page();
        p.figures.swap(2, 5);
        assert!(lint(&p).iter().any(|e| e.contains("out of order")));

        let mut p = page();
        p.tiles[0].unit.clear();
        assert!(lint(&p).iter().any(|e| e.contains("tile \"candidate commits\"")));
    }

    #[test]
    fn tiles_carry_units_and_definitions() {
        let p = page();
        let labels: Vec<&str> = p.tiles.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["candidate commits", "valid questions", "waiting for a retry", "fairness checked",
                            "exam-ready", "practice set (evolve)", "final exam (held-out)"]);
        assert_eq!(p.tiles[0].unit, "commits");
        assert!(p.tiles[0].definition.contains("\u{2264} 400 lines"), "{}", p.tiles[0].definition);
        assert!(p.tiles[1].definition.starts_with("86%") || p.tiles[1].definition.starts_with("86% of 7"),
                "{}", p.tiles[1].definition);
        assert!(p.tiles[3].definition.contains("stages not run yet"));
    }
}
