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

//! Every figure, table caption, number tile and the methods box of the
//! report, built from the data at render time.
//!
//! The report is held to the bar of a figure in an empirical paper: every
//! figure and table is numbered and captioned (what is plotted, unit, n,
//! how it was computed, one data-derived takeaway), every axis and legend
//! is titled with its unit, and every number tile has a unit and a
//! definition. [`lint`] checks this structurally and [`super::render`]
//! refuses to write a page that fails it.

use super::health::{frac, median_ci, Health, SplitStats, TOP_AREAS};
use super::load::{Split, Task};
use super::render::iso_utc;
use super::split::Summary;
use super::stats::BOOTSTRAP_RESAMPLES;
use rrsi_mine::{GENERATED, MAX_SRC_CHURN, MAX_SRC_PACKAGES};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const PRACTICE: &str = "practice set (evolve)";
pub const FINAL: &str = "final exam (held-out)";
pub const UNUSED: &str = "not used (excluded)";
pub const FIX_SIZE_UNIT: &str = "lines changed in the reference fix (added + removed, non-test, non-generated Go files)";

pub fn set_label(s: Split) -> &'static str {
    match s {
        Split::Evolve => PRACTICE,
        Split::Heldout => FINAL,
        Split::Excluded => UNUSED,
    }
}

/// How the tasks were mined, for the methods box (the miner does not
/// record its own flags in the task files).
pub struct Provenance<'a> {
    pub since: &'a str,
    pub go_image: &'a str,
    pub heldout: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Caption {
    /// "Figure" or "Table".
    pub kind: &'static str,
    pub number: usize,
    /// The page element carrying it: `data-fig="<id>"`.
    pub id: &'static str,
    pub title: String,
    pub what: String,
    pub unit: String,
    pub n: String,
    pub how: String,
    pub takeaway: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Figure {
    #[serde(flatten)]
    pub caption: Caption,
    /// Vega-Lite spec; colours are `token:<css variable>` resolved by the page.
    pub spec: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct Tile {
    pub value: String,
    pub unit: String,
    pub label: String,
    pub definition: String,
}

#[derive(Debug, Serialize)]
pub struct Page {
    pub tiles: Vec<Tile>,
    pub figures: Vec<Figure>,
    pub methods: Vec<(String, String)>,
}

fn pct(k: usize, n: usize) -> String {
    if n == 0 { "0%".into() } else { format!("{:.0}%", 100.0 * k as f64 / n as f64) }
}

fn set_scale() -> Value {
    json!({"domain": [PRACTICE, FINAL, UNUSED], "range": ["token:evolve", "token:heldout", "token:excluded"]})
}

fn base(values: Value) -> Value {
    json!({"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "data": {"values": values}})
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Some(a), Value::Object(b)) = (a.as_object_mut(), b) {
        a.extend(b);
    }
    a
}

/// Rows {set, key, k, n, share, label} for a 100%-stacked share chart,
/// folding keys outside `keep` into "other".
fn share_rows(e: &SplitStats, h: &SplitStats, pick: fn(&SplitStats) -> &BTreeMap<String, usize>,
              keep: Option<&[String]>) -> (Vec<Value>, Vec<String>) {
    let mut rows = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for (label, st) in [(PRACTICE, e), (FINAL, h)] {
        let mut folded: BTreeMap<String, usize> = BTreeMap::new();
        for (k, c) in pick(st) {
            let key = match keep {
                Some(keep) if !keep.contains(k) => "other".to_string(),
                _ => k.clone(),
            };
            *folded.entry(key).or_default() += c;
        }
        for (k, c) in folded {
            if !keys.contains(&k) {
                keys.push(k.clone());
            }
            let share = if st.n == 0 { 0.0 } else { c as f64 / st.n as f64 };
            rows.push(json!({"set": label, "key": k, "k": c, "n": st.n, "share": share,
                             "label": format!("{c}/{}", st.n), "pct": format!("{:.0}%", share * 100.0)}));
        }
    }
    keys.sort();
    if let Some(i) = keys.iter().position(|k| k == "other") {
        let o = keys.remove(i);
        keys.push(o);
    }
    (rows, keys)
}

fn share_spec(rows: Vec<Value>, keys: &[String], legend_title: &str) -> Value {
    let palette = ["token:cat1", "token:cat2", "token:cat3", "token:cat4", "token:cat5", "token:cat6"];
    let range: Vec<&str> = keys.iter().enumerate()
        .map(|(i, k)| if k == "other" { "token:excluded" } else { palette[i % palette.len()] }).collect();
    merge(base(json!(rows)), json!({
        "height": {"step": 38},
        "encoding": {
            "y": {"field": "set", "type": "nominal", "sort": [PRACTICE, FINAL], "title": "Set",
                  "axis": {"labelLimit": 180}},
            "x": {"field": "share", "type": "quantitative", "stack": "normalize",
                  "title": "Share of tasks in the set (%)", "axis": {"format": "%"}},
            "order": {"field": "key", "sort": "ascending"}
        },
        "layer": [
            {"mark": {"type": "bar", "height": {"band": 0.72}, "stroke": "token:panel", "strokeWidth": 2},
             "encoding": {
                "color": {"field": "key", "type": "nominal", "title": legend_title,
                          "scale": {"domain": keys, "range": range},
                          "legend": {"orient": "bottom", "columns": 2, "labelLimit": 240}},
                "tooltip": [{"field": "set", "title": "Set"}, {"field": "key", "title": legend_title},
                            {"field": "pct", "title": "Share of the set"},
                            {"field": "label", "title": "Tasks (count/set size)"}]}},
            {"transform": [{"filter": "datum.share >= 0.12"}],
             "mark": {"type": "text", "color": "#ffffff", "fontSize": 11, "fontWeight": 600},
             "encoding": {
                "x": {"field": "share", "type": "quantitative", "stack": "normalize", "bandPosition": 0.5,
                      "title": "Share of tasks in the set (%)"},
                "text": {"field": "label"}, "detail": {"field": "key"}}}
        ]
    }))
}

/// Keys of `m` ranked by their larger share in either split, top `k`.
fn top_keys(e: &SplitStats, h: &SplitStats, pick: fn(&SplitStats) -> &BTreeMap<String, usize>, k: usize) -> Vec<String> {
    let share = |st: &SplitStats, key: &String| {
        if st.n == 0 { 0.0 } else { pick(st).get(key).copied().unwrap_or(0) as f64 / st.n as f64 }
    };
    let mut keys: Vec<String> = pick(e).keys().chain(pick(h).keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    keys.sort_by(|a, b| share(e, b).max(share(h, b)).total_cmp(&share(e, a).max(share(h, a))).then_with(|| a.cmp(b)));
    keys.truncate(k);
    keys
}

fn meaning(h: &Health, id: &str) -> String {
    h.checks.iter().find(|c| c.id == id).map(|c| c.meaning.clone()).unwrap_or_default()
}

fn date_range(tasks: &[Task]) -> Option<(String, String)> {
    let min = tasks.iter().filter_map(|t| t.ts).min()?;
    let max = tasks.iter().filter_map(|t| t.ts).max()?;
    Some((iso_utc(min)[..10].to_string(), iso_utc(max)[..10].to_string()))
}

/// (kind, id, title, what, unit, n, how, takeaway, spec), in page order.
type Row = (&'static str, &'static str, String, String, String, String, String, String, Option<Value>);

pub fn build(tasks: &[Task], s: &Summary, h: &Health, prov: &Provenance) -> Page {
    let (e, x) = (&h.evolve, &h.heldout);
    let cand = s.candidates;
    let dropped = cand - s.exam_ready;
    let range = date_range(tasks);
    let window = match &range {
        Some((a, b)) => format!("committed {a} to {b} (UTC)"),
        None => format!("committed since {}", prov.since),
    };
    let sizes = format!("practice set n={} tasks, final exam n={} tasks", e.n, x.n);

    let tiles = vec![
        Tile { value: cand.to_string(), unit: "commits".into(), label: "candidate commits".into(),
               definition: format!("changed Go code and its own package's tests, {window}; fix \u{2264} {MAX_SRC_CHURN} lines, \u{2264} {MAX_SRC_PACKAGES} packages") },
        Tile { value: s.valid.to_string(), unit: "commits".into(), label: "valid questions".into(),
               definition: format!("{} of {cand}: tests fail before the fix and pass after it, offline", pct(s.valid, cand)) },
        Tile { value: s.pending_retry.to_string(), unit: "commits".into(), label: "waiting for a retry".into(),
               definition: "a test run hit an infrastructure failure (e.g. a module download) and will be rerun".into() },
        Tile { value: s.fairness_ran.to_string(), unit: "tasks".into(), label: "fairness checked".into(),
               definition: if s.fairness_ran == 0 { "tasks with a fairness verdict (stages not run yet)".into() }
                           else { format!("tasks with a fairness verdict, of {} valid", s.valid) } },
        Tile { value: s.exam_ready.to_string(), unit: "tasks".into(), label: "exam-ready".into(),
               definition: "valid and, where fairness checks ran, passed all of them".into() },
        Tile { value: s.evolve.to_string(), unit: "tasks".into(), label: PRACTICE.into(),
               definition: format!("exam-ready tasks RRSI may practise on: all but the newest {}", prov.heldout) },
        Tile { value: s.heldout.to_string(), unit: "tasks".into(), label: FINAL.into(),
               definition: format!("the {} newest exam-ready tasks by commit date, never seen during RRSI", prov.heldout) },
    ];

    let top_reason = s.rejections.first();
    let mut figs: Vec<Row> = Vec::new();

    // Table: why each filter exists (Start here).
    let mine_dropped = s.excluded_by_mine;
    figs.push(("Table", "why", "Why each filter exists".into(),
        "Each row names a filter, what would go wrong without it, and a real case from this data.".into(),
        "commits (one task is one commit)".into(), format!("{cand} candidate commits"),
        "each example is the smallest matching task; log lines are quoted verbatim".into(),
        format!("{mine_dropped} of {cand} commits ({}) failed the test-run filters; {}.", pct(mine_dropped, cand),
                if s.fairness_ran == 0 { "the fairness filters have not run yet".to_string() }
                else { format!("{} more failed the fairness filters", s.excluded_by_fairness) }),
        None));
    figs.push(("Table", "outcomes", "How to read a future result (illustrative numbers, not data)".into(),
        "Hypothetical practice-set and final-exam pass rates before \u{2192} after RRSI, and what each pattern would mean.".into(),
        "pass rate, % of tasks solved".into(), "none: illustrative, not measured".into(),
        "not computed; invented to show the reading".into(),
        "No RRSI result exists yet; these rows are not data.".into(), None));

    // Figure: funnel.
    let steps = [("1. candidate commits", cand, "all"), ("2. valid questions", s.valid, "all"),
                 ("3. exam-ready", s.exam_ready, "all"), ("4a. practice set (evolve)", s.evolve, "evolve"),
                 ("4b. final exam (held-out)", s.heldout, "heldout")];
    let rows: Vec<Value> = steps.iter().map(|(st, n, set)| json!({"step": st, "n": n, "set": set,
        "label": format!("{n} ({} of {cand})", pct(*n, cand))})).collect();
    let funnel = merge(base(json!(rows)), json!({
        "height": 200,
        "encoding": {"y": {"field": "step", "type": "nominal", "sort": null, "title": "Filter step", "axis": {"labelLimit": 200}},
                     "x": {"field": "n", "type": "quantitative", "title": "Number of commits",
                           "axis": {"tickMinStep": 1, "format": "d"}, "scale": {"domainMax": cand as f64 * 1.18}}},
        "layer": [
            {"mark": {"type": "bar", "height": {"band": 0.7}},
             "encoding": {"color": {"field": "set", "type": "nominal", "title": "Set",
                                    "scale": {"domain": ["all", "evolve", "heldout"], "range": ["token:ink-2", "token:evolve", "token:heldout"]},
                                    "legend": null},
                          "tooltip": [{"field": "step", "title": "Filter step"}, {"field": "label", "title": "Commits"}]}},
            {"mark": {"type": "text", "align": "left", "dx": 4, "color": "token:ink"},
             "encoding": {"text": {"field": "label"}}}]
    }));
    figs.push(("Figure", "funnel", "Where commits were lost".into(),
        format!("Bars show how many commits remain after each filter, from {cand} candidates to the two exam sets; the practice set and the final exam partition the {} exam-ready tasks.", s.exam_ready),
        "commits (one task is one commit)".into(), format!("{cand} candidate commits"),
        format!("candidate: {window}, changing Go source and the tests of the same package, fix \u{2264} {MAX_SRC_CHURN} lines, \u{2264} {MAX_SRC_PACKAGES} packages; valid: tests fail before the fix and pass after it in an offline container; exam-ready: valid and, where fairness checks ran, passed all of them"),
        match top_reason {
            Some((r, _, k)) => format!("{dropped} of {cand} commits ({}) were dropped, most often because \u{201c}{r}\u{201d} ({k} commits).", pct(dropped, cand)),
            None => "No commit was dropped.".into(),
        },
        Some(funnel)));

    // Table: split health.
    figs.push(("Table", "health", "Is the final exam comparable to the practice set?".into(),
        "Six checks compare the final exam (held-out) with the practice set (evolve); each row gives the result, the numbers in words, why it matters, what to do, and its threshold.".into(),
        format!("fix size in {FIX_SIZE_UNIT}; shares in % of the set with count/denominator; days"),
        sizes.clone(),
        format!("medians with 95% percentile-bootstrap intervals ({BOOTSTRAP_RESAMPLES} resamples, fixed seed); proportions with 95% Wilson intervals; thresholds are heuristics chosen for this prototype, not standards. With only {} final-exam tasks its estimates are noisy: read the intervals, not just the point values", x.n),
        h.verdict.clone(), None));

    // Figure: fix size per set.
    let pts: Vec<Value> = tasks.iter().filter(|t| t.split != Split::Excluded).map(|t| json!({
        "set": set_label(t.split), "lines": t.src_churn.max(1), "subject": t.subject, "sha": t.sha12})).collect();
    let meds: Vec<Value> = [(PRACTICE, e), (FINAL, x)].iter().filter_map(|(l, st)| {
        let m = st.churn_median?;
        let (lo, hi) = st.churn_median_ci.unwrap_or((m, m));
        Some(json!({"set": l, "median": m.max(1.0), "lo": lo.max(1.0), "hi": hi.max(1.0),
                    "label": format!("median {}", median_ci(m, st.churn_median_ci, st.n))}))
    }).collect();
    let size = json!({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": {"step": 70},
        "encoding": {"y": {"field": "set", "type": "nominal", "sort": [PRACTICE, FINAL], "title": "Set", "axis": {"labelLimit": 180}}},
        "layer": [
            {"data": {"values": meds}, "mark": {"type": "rule", "strokeWidth": 10, "opacity": 0.25, "color": "token:ink-2"},
             "encoding": {"x": {"field": "lo", "type": "quantitative", "scale": {"type": "log"}, "title": "Fix size, lines changed (log scale)"},
                          "x2": {"field": "hi"}}},
            {"data": {"values": pts}, "mark": {"type": "tick", "thickness": 2, "size": 22, "opacity": 0.75},
             "encoding": {"x": {"field": "lines", "type": "quantitative", "scale": {"type": "log"}, "title": "Fix size, lines changed (log scale)"},
                          "color": {"field": "set", "type": "nominal", "scale": set_scale(), "title": "Set", "legend": null},
                          "tooltip": [{"field": "subject", "title": "Commit"}, {"field": "sha", "title": "sha"},
                                      {"field": "lines", "title": "Lines changed"}]}},
            {"data": {"values": meds}, "mark": {"type": "tick", "thickness": 3, "size": 36, "color": "token:ink"},
             "encoding": {"x": {"field": "median", "type": "quantitative", "title": "Fix size, lines changed (log scale)"},
                          "tooltip": [{"field": "label", "title": "Median"}]}},
            {"data": {"values": meds}, "mark": {"type": "text", "dy": -26, "fontSize": 11, "fontWeight": 600, "color": "token:ink"},
             "encoding": {"x": {"field": "median", "type": "quantitative", "title": "Fix size, lines changed (log scale)"},
                          "text": {"field": "label"}}}]
    });
    figs.push(("Figure", "size", "Fix size by set".into(),
        "Each tick is one task, placed by the size of its reference fix; the black mark and label give the median, the grey band its 95% interval.".into(),
        format!("{FIX_SIZE_UNIT}; log scale"), sizes.clone(),
        format!("median with a 95% percentile-bootstrap interval ({BOOTSTRAP_RESAMPLES} resamples, fixed seed)"),
        meaning(h, "churn"), Some(size)));

    // Figures: share charts.
    let (rows, keys) = share_rows(e, x, |st| &st.module_tree_n, None);
    figs.push(("Figure", "tree", "Code layout (module tree) by set".into(),
        "Each bar splits one set by the module tree its tasks change (the first folder of the Go module root); labels give task count/set size.".into(),
        "share of tasks in the set, %".into(), sizes.clone(),
        "tasks per module tree divided by the set size".into(),
        meaning(h, "tree"), Some(share_spec(rows, &keys, "Module tree"))));
    let top = top_keys(e, x, |st| &st.area_n, TOP_AREAS);
    let (rows, keys) = share_rows(e, x, |st| &st.area_n, Some(&top));
    figs.push(("Figure", "folders", "Folders by set".into(),
        format!("Each bar splits one set by folder (the first three path segments of the task's primary package); folders outside the {TOP_AREAS} with the largest share in either set are grouped as \u{201c}other\u{201d}."),
        "share of tasks in the set, %".into(), sizes.clone(),
        "tasks per folder divided by the set size".into(),
        meaning(h, "area"), Some(share_spec(rows, &keys, "Folder"))));
    let top = top_keys(e, x, |st| &st.component_n, TOP_AREAS);
    let (rows, keys) = share_rows(e, x, |st| &st.component_n, Some(&top));
    figs.push(("Figure", "subsystems", "Subsystems by set".into(),
        format!("Like Folders, but tasks in different folders of one subsystem (e.g. its service and its command) count together; subsystems outside the {TOP_AREAS} largest are \u{201c}other\u{201d}."),
        "share of tasks in the set, %".into(), sizes.clone(),
        "subsystem = first path segment after the module tree that is not a grouping folder (services, app, pkg, cmd, internal, ...)".into(),
        meaning(h, "component"), Some(share_spec(rows, &keys, "Subsystem"))));
    let (rows, keys) = share_rows(e, x, |st| &st.outcome_n, None);
    figs.push(("Figure", "failkind", "How the tests failed before the fix, by set".into(),
        "Each bar splits one set by the outcome of the hidden tests on the commit before the fix: BuildFail (the tests did not compile) or TestFail (they compiled and a check failed). Every task then passes with the fix.".into(),
        "share of tasks in the set, %".into(), sizes.clone(),
        "outcome classified from the go test exit code and log".into(),
        meaning(h, "outcome"), Some(share_spec(rows, &keys, "Before \u{2192} with the fix"))));

    // Figure: rejections.
    let rej: Vec<Value> = s.rejections.iter().map(|(r, k, n)| json!({"reason": r, "n": n,
        "source": if k == "mine" { "test runs" } else { "fairness checks" }})).collect();
    let total_rej: usize = s.rejections.iter().map(|r| r.2).sum();
    let reject = if rej.is_empty() { None } else { Some(merge(base(json!(rej)), json!({
        "height": {"step": 30},
        "encoding": {"y": {"field": "reason", "type": "nominal", "sort": "-x", "title": "Reason for dropping", "axis": {"labelLimit": 300}},
                     "x": {"field": "n", "type": "quantitative", "title": "Number of commits dropped", "axis": {"tickMinStep": 1, "format": "d"}}},
        "layer": [
            {"mark": {"type": "bar", "height": {"band": 0.7}},
             "encoding": {"color": {"field": "source", "type": "nominal", "title": "Where the reason comes from",
                                    "scale": {"domain": ["test runs", "fairness checks"], "range": ["token:mine", "token:fair"]}},
                          "tooltip": [{"field": "reason", "title": "Reason"}, {"field": "n", "title": "Commits"}]}},
            {"mark": {"type": "text", "align": "left", "dx": 4, "color": "token:ink"}, "encoding": {"text": {"field": "n"}}}]
    }))) };
    figs.push(("Figure", "reject", "Why commits were dropped".into(),
        "Bars count dropped commits per reason. Test-run reasons come from the fail-before / pass-after check; fairness reasons from the fairness stages (one commit can fail several stages).".into(),
        "commits".into(), format!("{dropped} of {cand} candidate commits dropped; {total_rej} reasons recorded"),
        "the reason the miner recorded, and each fairness stage with a failing verdict".into(),
        match top_reason {
            Some((r, _, k)) => format!("The most common reason is \u{201c}{r}\u{201d}: {k} of {cand} candidate commits ({}).", pct(*k, cand)),
            None => "Nothing was dropped.".into(),
        },
        reject));

    // Figure: timeline.
    let dated: Vec<Value> = tasks.iter().filter_map(|t| Some(json!({
        "date": iso_utc(t.ts?), "lines": t.src_churn.max(1), "set": set_label(t.split),
        "subject": t.subject, "sha": t.sha12, "why": t.excluded_reason.clone().unwrap_or_default()}))).collect();
    let n_dated = dated.len();
    let cutoff = tasks.iter().filter(|t| t.split == Split::Heldout).filter_map(|t| t.ts).min();
    let timeline = if dated.is_empty() { None } else {
        let mut layers = vec![json!({
            "mark": {"type": "point", "filled": true, "size": 70, "opacity": 0.85, "stroke": "token:panel", "strokeWidth": 1},
            "encoding": {
                "x": {"field": "date", "type": "temporal", "timeUnit": "utcyearmonthdatehoursminutes",
                      "title": "Commit date (UTC)", "axis": {"format": "%b %d", "labelOverlap": true}},
                "y": {"field": "lines", "type": "quantitative", "scale": {"type": "log"}, "title": "Fix size, lines changed (log scale)"},
                "color": {"field": "set", "type": "nominal", "scale": set_scale(), "title": "Set"},
                "tooltip": [{"field": "subject", "title": "Commit"}, {"field": "sha", "title": "sha"},
                            {"field": "date", "type": "temporal", "timeUnit": "utcyearmonthdatehoursminutes", "format": "%Y-%m-%d %H:%M UTC", "title": "Committed"},
                            {"field": "set", "title": "Set"}, {"field": "lines", "title": "Lines changed"},
                            {"field": "why", "title": "Not used because"}]}})];
        if let Some(c) = cutoff {
            layers.push(json!({"data": {"values": [{"cutoff": iso_utc(c), "label": "final exam starts"}]},
                "layer": [
                    {"mark": {"type": "rule", "strokeDash": [5, 4], "color": "token:heldout", "strokeWidth": 1.5},
                     "encoding": {"x": {"field": "cutoff", "type": "temporal", "timeUnit": "utcyearmonthdatehoursminutes", "title": "Commit date (UTC)"}}},
                    {"mark": {"type": "text", "align": "right", "dx": -4, "dy": 8, "baseline": "top", "color": "token:ink-2", "fontSize": 11},
                     "encoding": {"x": {"field": "cutoff", "type": "temporal", "timeUnit": "utcyearmonthdatehoursminutes", "title": "Commit date (UTC)"},
                                  "y": {"value": 0}, "text": {"field": "label"}}}]}));
        }
        Some(merge(base(json!(dated)), json!({"height": 280, "layer": layers})))
    };
    figs.push(("Figure", "timeline", "Commit dates and sets".into(),
        "Each dot is one task at its commit date; the dashed line marks the oldest final-exam commit.".into(),
        "date (UTC); fix size in lines changed, log scale".into(),
        format!("{n_dated} of {cand} tasks with a commit date"),
        "committer date from git; the final exam is the newest exam-ready tasks".into(),
        if timeline.is_some() { format!("{} {}", meaning(h, "span"), match &range {
            Some((a, b)) => format!("All {cand} candidates were committed between {a} and {b}."), None => String::new() }) }
        else { "No commit dates (run with --repo), so there is no timeline and the final exam is the last tasks by sha.".into() },
        timeline));

    // Figure: coverage by folder.
    let cov: Vec<Value> = tasks.iter().map(|t| json!({"folder": t.area, "set": set_label(t.split)})).collect();
    let mut ho_folders: BTreeMap<&str, usize> = BTreeMap::new();
    for t in tasks.iter().filter(|t| t.split == Split::Heldout) {
        *ho_folders.entry(t.area.as_str()).or_default() += 1;
    }
    let n_folders = tasks.iter().map(|t| t.area.as_str()).collect::<std::collections::BTreeSet<_>>().len();
    let biggest = ho_folders.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)));
    figs.push(("Figure", "coverage", "Tasks per folder and set".into(),
        "One bar per folder (the first three path segments of the task's primary package), split by set.".into(),
        "tasks (one task is one commit)".into(), format!("{cand} tasks in {n_folders} folders"),
        "count of tasks per folder and set".into(),
        match biggest {
            Some((f, k)) => format!("The final exam covers {} of {n_folders} folders; its largest, {f}, holds {}.",
                                    ho_folders.len(), frac(*k, x.n, "final-exam tasks")),
            None => "The final exam is empty.".into(),
        },
        Some(merge(base(json!(cov)), json!({
            "height": {"step": 20},
            "mark": {"type": "bar", "height": {"band": 0.75}, "stroke": "token:panel", "strokeWidth": 1},
            "encoding": {
                "y": {"field": "folder", "type": "nominal", "sort": "-x", "title": "Folder (first three path segments)", "axis": {"labelLimit": 260}},
                "x": {"aggregate": "count", "type": "quantitative", "title": "Number of tasks", "axis": {"tickMinStep": 1, "format": "d"}},
                "color": {"field": "set", "type": "nominal", "scale": set_scale(), "title": "Set"},
                "tooltip": [{"field": "folder", "title": "Folder"}, {"field": "set", "title": "Set"},
                            {"aggregate": "count", "title": "Tasks"}]}})))));

    figs.push(("Table", "browser", "All tasks".into(),
        "One row per candidate commit: its set, commit date, folder, fix size, test outcomes before \u{2192} with the fix, and fairness verdicts. Click a row for its card.".into(),
        "fix size in lines changed; dates in UTC".into(), format!("{cand} tasks"),
        "as recorded by the miner and the fairness stages".into(),
        format!("{} in the final exam, {} in the practice set, {} not used.", s.heldout, s.evolve, cand - s.exam_ready),
        None));

    let mut counters: BTreeMap<&str, usize> = BTreeMap::new();
    let figures = figs.into_iter().map(|(kind, id, title, what, unit, n, how, takeaway, spec)| {
        let c = counters.entry(kind).or_default();
        *c += 1;
        Figure { caption: Caption { kind, number: *c, id, title, what, unit, n, how, takeaway }, spec }
    }).collect();

    let methods = vec![
        ("Data".to_string(), format!("Non-merge commits of the repository given with --repo that touch a *_test.go file, {window}; the miner ran with --since {}.", prov.since)),
        ("Candidate filter".to_string(), format!("The commit changes Go source files and test files in the same package directory; fix size (added + removed lines in non-test Go files, excluding generated files matching {}) \u{2264} {MAX_SRC_CHURN}; at most {MAX_SRC_PACKAGES} source packages; all shared packages under one Go module.", GENERATED.join(", "))),
        ("Validation".to_string(), format!("The package tests run twice in a network-less container ({}, GOPROXY=off, GOFLAGS=-mod=mod, module and build caches filled beforehand, per-run timeout): on the parent commit with the new tests applied, and on the commit itself. Each run is classified Pass, TestFail (a test ran and failed), BuildFail (the package or its tests did not compile), Infra (the toolchain could not assemble its inputs, e.g. a module download) or Timeout. A task is valid only for TestFail or BuildFail \u{2192} Pass; Infra and Timeout are never evidence and are retried.", prov.go_image)),
        ("Split".to_string(), format!("Exam-ready tasks are ordered by committer date; the newest {} form the final exam (held-out), ties broken by commit hash; the rest form the practice set (evolve).", prov.heldout)),
        ("Fairness stages".to_string(), "flake (reruns the hidden tests to catch flaky ones), api (lists the names the tests need), describe (writes the instruction with a model and checks it does not leak the fix), probe (a second model judges whether the instruction alone is sufficient and unambiguous), specificity (flags tests that pin details no instruction could name), gate (all of the above).".into()),
        ("Statistics".to_string(), format!("Medians with 95% percentile-bootstrap intervals ({BOOTSTRAP_RESAMPLES} resamples, fixed seed); proportions with 95% Wilson score intervals. Split-health thresholds are heuristics chosen for this prototype.")),
        ("Not yet measured".to_string(), "Whether RRSI improves an agent on this exam: no baseline and no RRSI rounds have been run. Pass rates, the noise band between repeated runs, and any held-out result are therefore absent.".into()),
    ];
    Page { tiles, figures, methods }
}

/// Words that count as a unit in an axis title of a quantitative or
/// temporal channel.
pub const UNIT_WORDS: [&str; 7] = ["commits", "tasks", "lines", "%", "UTC", "days", "share"];

fn lint_spec(fig: &str, v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            if let Some(Value::Object(enc)) = m.get("encoding") {
                for (ch, def) in enc {
                    if !["x", "y", "color", "row", "column"].contains(&ch.as_str()) {
                        continue;
                    }
                    let Some(d) = def.as_object() else { continue };
                    if !(d.contains_key("field") || d.contains_key("aggregate")) {
                        continue;
                    }
                    let title = d.get("title").and_then(Value::as_str).unwrap_or("");
                    if title.trim().is_empty() {
                        out.push(format!("{fig}: channel {ch} has no title"));
                        continue;
                    }
                    let ty = d.get("type").and_then(Value::as_str).unwrap_or("");
                    if (ch == "x" || ch == "y") && (ty == "quantitative" || ty == "temporal")
                        && !UNIT_WORDS.iter().any(|u| title.to_lowercase().contains(&u.to_lowercase())) {
                        out.push(format!("{fig}: axis {ch} title {title:?} names no unit"));
                    }
                }
            }
            for (k, c) in m {
                if k != "data" {
                    lint_spec(fig, c, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|c| lint_spec(fig, c, out)),
        _ => {}
    }
}

/// Structural checks of the paper bar. Returns every problem found.
pub fn lint(page: &Page) -> Vec<String> {
    let mut out = Vec::new();
    for t in &page.tiles {
        if t.value.is_empty() || t.unit.trim().is_empty() || t.label.trim().is_empty() || t.definition.trim().is_empty() {
            out.push(format!("tile {:?} lacks a value, unit, label or definition", t.label));
        }
    }
    let mut expect: BTreeMap<&str, usize> = BTreeMap::new();
    for f in &page.figures {
        let c = &f.caption;
        let n = expect.entry(c.kind).or_default();
        *n += 1;
        if c.number != *n {
            out.push(format!("{} {} ({}) is out of order: expected {}", c.kind, c.number, c.id, n));
        }
        for (name, text) in [("title", &c.title), ("what", &c.what), ("unit", &c.unit), ("n", &c.n),
                             ("how", &c.how), ("takeaway", &c.takeaway)] {
            if text.trim().is_empty() {
                out.push(format!("{} {} ({}) caption lacks {name}", c.kind, c.number, c.id));
            }
        }
        if let Some(spec) = &f.spec {
            lint_spec(&format!("{} {} ({})", c.kind, c.number, c.id), spec, &mut out);
        }
    }
    out
}
