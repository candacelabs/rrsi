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

//! Split health: how comparable the held-out split is to evolve.
//!
//! Per split: task count, source-churn median and IQR, the share of tasks
//! per module tree (first segment of the module root), per area (see
//! [`super::load::area`]), per component (see [`component`]: `warden` for
//! both `svc/services/warden/x` and `svc/app/warden/cmd`) and per
//! parent→commit outcome transition, and the commit-date span, with 95%
//! intervals (see [`super::stats`]). The checks compare held-out against
//! evolve with the fixed thresholds below; each yields `pass`, `warn`, or
//! `n/a` when a split is empty or undated. The thresholds are heuristics
//! chosen for this prototype, not established standards; each check states
//! its rationale.

use super::load::{Split, Task};
use super::stats::{bootstrap_median_ci, wilson};
use serde::Serialize;
use std::collections::BTreeMap;

/// Held-out median churn must be within this factor range of evolve's.
pub const CHURN_RATIO_MIN: f64 = 0.67;
pub const CHURN_RATIO_MAX: f64 = 1.5;
/// No single area, and no single component, may hold more than this share
/// of held-out.
pub const MAX_HELDOUT_AREA_SHARE: f64 = 0.40;
/// Module-tree shares may differ between splits by at most this (0..1).
pub const MAX_TREE_SHARE_DIFF: f64 = 0.30;
/// Outcome-transition shares may differ between splits by at most this.
pub const MAX_OUTCOME_SHARE_DIFF: f64 = 0.30;
/// Held-out must cover at least this many days of commits.
pub const MIN_HELDOUT_SPAN_DAYS: f64 = 7.0;
/// Areas drawn individually in the area-share chart; the rest are "other".
pub const TOP_AREAS: usize = 6;

#[derive(Debug, Default, Serialize)]
pub struct SplitStats {
    pub n: usize,
    pub churn_median: Option<f64>,
    pub churn_q1: Option<f64>,
    pub churn_q3: Option<f64>,
    /// 95% percentile-bootstrap interval of the median.
    pub churn_median_ci: Option<(f64, f64)>,
    /// Module tree -> share of this split's tasks (0..1).
    pub module_tree: BTreeMap<String, f64>,
    pub area: BTreeMap<String, f64>,
    pub component: BTreeMap<String, f64>,
    /// `area` folded to the report's top areas plus "other".
    pub area_top: BTreeMap<String, f64>,
    /// "BuildFail→Pass" etc. -> share.
    pub outcome: BTreeMap<String, f64>,
    /// Task counts behind each share map (the numerators; `n` is the denominator).
    pub module_tree_n: BTreeMap<String, usize>,
    pub area_n: BTreeMap<String, usize>,
    pub component_n: BTreeMap<String, usize>,
    pub outcome_n: BTreeMap<String, usize>,
    pub date_min: Option<String>,
    pub date_max: Option<String>,
    pub span_days: Option<f64>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pass,
    Warn,
    #[serde(rename = "n/a")]
    NotApplicable,
}

#[derive(Debug, Serialize)]
pub struct Check {
    /// Stable identifier (`churn`, `area`, `component`, `tree`, `outcome`, `span`).
    pub id: &'static str,
    /// Plain-language name.
    pub name: String,
    pub status: Status,
    /// The actual numbers and the threshold, in one line.
    pub detail: String,
    /// One plain sentence using the actual numbers.
    pub meaning: String,
    /// What would go wrong if this is off.
    pub why: &'static str,
    /// The threshold, in words.
    pub threshold: String,
    /// Why this threshold (a heuristic chosen for this prototype).
    pub rationale: &'static str,
    /// What to do when it warns.
    pub action: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Thresholds {
    pub churn_ratio: (f64, f64),
    pub max_heldout_area_share: f64,
    pub max_tree_share_diff: f64,
    pub max_outcome_share_diff: f64,
    pub min_heldout_span_days: f64,
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub thresholds: Thresholds,
    pub top_areas: Vec<String>,
    pub evolve: SplitStats,
    pub heldout: SplitStats,
    pub checks: Vec<Check>,
    pub warnings: usize,
    /// Plain-language conclusion over all checks.
    pub verdict: String,
}

/// Linear-interpolated quantile (numpy's default) of sorted values.
pub fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64))
}

/// First segment of the module root: `go`, `candace`, ... (`(root)` if none).
pub fn module_tree(module_root: &str) -> String {
    module_root.split('/').find(|s| !s.is_empty() && *s != ".")
        .unwrap_or("(root)").to_string()
}

/// Path segments that only group code, never name what it is.
pub const CONTAINER_SEGMENTS: [&str; 11] =
    ["services", "service", "app", "apps", "pkg", "cmd", "internal", "lib", "src", "tools", "examples"];

/// What the task's primary package is part of, across module trees: the
/// first segment of its repository path, after the module tree, that is not
/// a grouping directory (`svc/services/warden/election` and
/// `go/app/warden/cmd` are both `warden`).
pub fn component(module_root: &str, packages: &[String]) -> String {
    let pkg = packages.first().map(String::as_str).unwrap_or(".");
    let segs: Vec<&str> = module_root.split('/').chain(pkg.split('/'))
        .filter(|s| !s.is_empty() && *s != ".").collect();
    segs.iter().skip(1).find(|s| !CONTAINER_SEGMENTS.contains(s))
        .or(segs.first()).map(|s| s.to_string()).unwrap_or_else(|| "(root)".into())
}

pub fn transition(t: &Task) -> String {
    format!("{}\u{2192}{}", t.parent_outcome.as_deref().unwrap_or("?"),
            t.commit_outcome.as_deref().unwrap_or("?"))
}

fn shares<'a>(keys: impl Iterator<Item = String> + 'a, n: usize) -> BTreeMap<String, f64> {
    let mut m: BTreeMap<String, f64> = BTreeMap::new();
    for k in keys {
        *m.entry(k).or_default() += 1.0;
    }
    if n > 0 {
        m.values_mut().for_each(|v| *v /= n as f64);
    }
    m
}

fn stats(tasks: &[&Task], top: &[String]) -> SplitStats {
    let n = tasks.len();
    let mut churn: Vec<f64> = tasks.iter().map(|t| t.src_churn as f64).collect();
    churn.sort_by(f64::total_cmp);
    let dated: Vec<(i64, &String)> = tasks.iter()
        .filter_map(|t| Some((t.ts?, t.date.as_ref()?))).collect();
    let min = dated.iter().min();
    let max = dated.iter().max();
    let fold = |a: &String| if top.contains(a) { a.clone() } else { "other".to_string() };
    SplitStats {
        n,
        churn_median: quantile(&churn, 0.5),
        churn_q1: quantile(&churn, 0.25),
        churn_q3: quantile(&churn, 0.75),
        churn_median_ci: bootstrap_median_ci(&churn),
        module_tree_n: counts(tasks.iter().map(|t| module_tree(&t.module_root))),
        area_n: counts(tasks.iter().map(|t| t.area.clone())),
        component_n: counts(tasks.iter().map(|t| component(&t.module_root, &t.packages))),
        outcome_n: counts(tasks.iter().map(|t| transition(t))),
        module_tree: shares(tasks.iter().map(|t| module_tree(&t.module_root)), n),
        area: shares(tasks.iter().map(|t| t.area.clone()), n),
        component: shares(tasks.iter().map(|t| component(&t.module_root, &t.packages)), n),
        area_top: shares(tasks.iter().map(|t| fold(&t.area)), n),
        outcome: shares(tasks.iter().map(|t| transition(t)), n),
        date_min: min.map(|d| super::render::iso_utc(d.0)),
        date_max: max.map(|d| super::render::iso_utc(d.0)),
        span_days: min.zip(max).map(|(a, b)| (b.0 - a.0) as f64 / 86_400.0),
    }
}

fn counts(keys: impl Iterator<Item = String>) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for k in keys {
        *m.entry(k).or_default() += 1;
    }
    m
}

/// "23% (14 of 60 practice tasks)".
pub fn frac(k: usize, n: usize, of: &str) -> String {
    let p = if n == 0 { 0.0 } else { k as f64 / n as f64 };
    format!("{} ({k} of {n} {of})", pct(p))
}

/// "100% (10 of 10 final-exam tasks; 95% CI 72\u{2013}100%)".
pub fn frac_ci(k: usize, n: usize, of: &str) -> String {
    let p = if n == 0 { 0.0 } else { k as f64 / n as f64 };
    match wilson(k, n) {
        Some((lo, hi)) => format!("{} ({k} of {n} {of}; 95% CI {:.0}\u{2013}{:.0}%)", pct(p), lo * 100.0, hi * 100.0),
        None => format!("{} ({k} of {n} {of})", pct(p)),
    }
}

/// "118 lines (95% CI 43\u{2013}190; n=10 tasks)".
pub fn median_ci(m: f64, ci: Option<(f64, f64)>, n: usize) -> String {
    match ci {
        Some((lo, hi)) => format!("{} lines (95% CI {}\u{2013}{}; n={n} tasks)", m.round(), lo.round(), hi.round()),
        None => format!("{} lines (n={n} tasks)", m.round()),
    }
}

fn pct(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

fn num(x: f64) -> String {
    let s = format!("{x:.1}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

/// The largest per-key share difference between two share maps.
fn max_diff(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> Option<(String, f64, f64)> {
    a.keys().chain(b.keys())
        .map(|k| (k.clone(), a.get(k).copied().unwrap_or(0.0), b.get(k).copied().unwrap_or(0.0)))
        .max_by(|x, y| (x.1 - x.2).abs().total_cmp(&(y.1 - y.2).abs()).then_with(|| y.0.cmp(&x.0)))
}

fn status(ok: Option<bool>) -> Status {
    match ok {
        Some(true) => Status::Pass,
        Some(false) => Status::Warn,
        None => Status::NotApplicable,
    }
}

/// A day count in words: "2.5 hours", "1 day", "12 days".
pub fn span_words(days: f64) -> String {
    if days < 1.0 {
        let h = days * 24.0;
        if h < 1.0 { "under an hour".into() } else { format!("{} hours", num((h * 10.0).round() / 10.0)) }
    } else {
        let d = (days * 10.0).round() / 10.0;
        format!("{} day{}", num(d), if d == 1.0 { "" } else { "s" })
    }
}

/// The module tree whose tasks are, by median commit time, oldest and newest.
fn old_new_trees(tasks: &[&Task]) -> Option<(String, String)> {
    let mut by: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for t in tasks {
        if let Some(ts) = t.ts {
            by.entry(module_tree(&t.module_root)).or_default().push(ts);
        }
    }
    let mut med: Vec<(i64, String)> = by.into_iter().map(|(k, mut v)| {
        v.sort_unstable();
        (v[v.len() / 2], k)
    }).collect();
    med.sort();
    if med.len() < 2 {
        return None;
    }
    Some((med[0].1.clone(), med[med.len() - 1].1.clone()))
}

pub fn health(tasks: &[Task]) -> Health {
    let ev: Vec<&Task> = tasks.iter().filter(|t| t.split == Split::Evolve).collect();
    let ho: Vec<&Task> = tasks.iter().filter(|t| t.split == Split::Heldout).collect();
    // Top areas by their larger share in either split, so a held-out-only
    // area is drawn even when evolve's many tasks outnumber it.
    let (ea, ha) = (stats(&ev, &[]).area, stats(&ho, &[]).area);
    let mut ranked: Vec<(&String, f64)> = ea.keys().chain(ha.keys())
        .map(|k| (k, ea.get(k).copied().unwrap_or(0.0).max(ha.get(k).copied().unwrap_or(0.0))))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked.dedup_by(|a, b| a.0 == b.0);
    let top: Vec<String> = ranked.iter().take(TOP_AREAS).map(|a| a.0.clone()).collect();
    let (e, h) = (stats(&ev, &top), stats(&ho, &top));
    let both = e.n > 0 && h.n > 0;
    let mut checks = Vec::new();
    let na = |id, name: String, why, action, threshold: String, rationale, what: &str| Check {
        id, name, status: Status::NotApplicable, detail: what.to_string(),
        meaning: format!("Cannot tell yet: {what}."), why, action, threshold, rationale,
    };
    const PRACTICE: &str = "practice tasks";
    const FINAL: &str = "final-exam tasks";

    // 1. Task size.
    let name = "Task size (median lines changed)".to_string();
    let why = "Bigger fixes usually mean harder tasks, so a lower final-exam score could not be told \
               apart from memorizing the practice set.";
    let action = "Choose final-exam tasks whose sizes match the practice set, or mine more history so \
                  there are more tasks to choose from.";
    let threshold = format!("final-exam median within {CHURN_RATIO_MIN}\u{2013}{CHURN_RATIO_MAX}\u{d7} of the practice median");
    let rationale = "a larger gap in fix size would plausibly change pass rates on its own";
    checks.push(match (e.churn_median, h.churn_median) {
        (Some(em), Some(hm)) if em > 0.0 => {
            let r = hm / em;
            let ok = (CHURN_RATIO_MIN..=CHURN_RATIO_MAX).contains(&r);
            let cmp = if r >= 1.0 { format!("{:.1}\u{d7} bigger", r) } else { format!("{:.1}\u{d7} smaller", 1.0 / r) };
            Check {
                id: "churn", name, status: status(Some(ok)), why, action, threshold, rationale,
                detail: format!("evolve {}, held-out {} \u{2192} {:.2}x, {} {}\u{2013}{}x",
                                num(em), num(hm), r, if ok { "within" } else { "outside" },
                                CHURN_RATIO_MIN, CHURN_RATIO_MAX),
                meaning: format!("The median final-exam fix changed {}; the median practice fix changed {} \
                                  \u{2014} final-exam fixes are {cmp}.",
                                 median_ci(hm, h.churn_median_ci, h.n), median_ci(em, e.churn_median_ci, e.n)),
            }
        }
        _ => na("churn", name, why, action, threshold, rationale, "a split is empty"),
    });

    // 2-3. Concentration by folder and by subsystem.
    for (id, name, what, m, em, why, action) in [
        ("area", "Biggest folder share of the final exam", "folder", &h.area_n, &e.area_n,
         "An exam mostly about one folder measures the agent on that folder, not in general.",
         "Cap how many final-exam tasks may come from one folder, or hold out more tasks."),
        ("component", "Biggest subsystem share of the final exam", "subsystem", &h.component_n, &e.component_n,
         "If the final exam is mostly one subsystem, its score reflects that subsystem more than the harness.",
         "Balance the final exam across subsystems, e.g. take the newest few tasks of each."),
    ] {
        let name = name.to_string();
        let threshold = format!("at most {} of the final exam in one {what}", pct(MAX_HELDOUT_AREA_SHARE));
        let rationale = "above this, one part of the code would decide most of the final score";
        checks.push(match m.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))) {
            Some((a, &k)) => {
                let s = k as f64 / h.n as f64;
                let ok = s <= MAX_HELDOUT_AREA_SHARE;
                Check {
                    id, name, status: status(Some(ok)), why, action, threshold, rationale,
                    detail: format!("largest held-out {what} {a}: {k} of {} ({}), {} max {}",
                                    h.n, pct(s), if ok { "within" } else { "above" }, pct(MAX_HELDOUT_AREA_SHARE)),
                    meaning: format!("The largest {what} of the final exam, {a}, holds {}; in the practice set it holds {}.",
                                     frac_ci(k, h.n, FINAL), frac(em.get(a).copied().unwrap_or(0), e.n, PRACTICE)),
                }
            }
            None => na(id, name, why, action, threshold, rationale, "the final exam is empty"),
        });
    }

    // 4. Module tree (code layout).
    let name = match old_new_trees(&ev.iter().chain(&ho).copied().collect::<Vec<_>>()) {
        Some((old, new)) if old != new => format!("Old {old}/ vs new {new}/ code layout"),
        _ => "Code layout (module tree) mix".to_string(),
    };
    let why = "Newer code is often laid out and tested differently; if only the final exam uses it, \
               the harness is judged on ground it never practised on.";
    let action = "Make sure both sets contain every code layout, e.g. mine further back or hold out \
                  the newest tasks per layout.";
    let threshold = format!("shares differ by at most {:.0} percentage points", MAX_TREE_SHARE_DIFF * 100.0);
    let rationale = "a larger difference means the two sets mostly test different code";
    checks.push(match max_diff(&e.module_tree, &h.module_tree).filter(|_| both) {
        Some((k, x, y)) => {
            let d = (x - y).abs();
            let ok = d <= MAX_TREE_SHARE_DIFF;
            let (ke, kh) = (e.module_tree_n.get(&k).copied().unwrap_or(0), h.module_tree_n.get(&k).copied().unwrap_or(0));
            Check {
                id: "tree", name, status: status(Some(ok)), why, action, threshold, rationale,
                detail: format!("{k}: evolve {}, held-out {} \u{2192} {:.0} points, {} max {:.0}",
                                pct(x), pct(y), d * 100.0, if ok { "within" } else { "above" }, MAX_TREE_SHARE_DIFF * 100.0),
                meaning: format!("Code under {k}/ is {} but {} \u{2014} {:.0} percentage points apart.",
                                 frac_ci(kh, h.n, FINAL), frac(ke, e.n, PRACTICE), d * 100.0),
            }
        }
        None => na("tree", name, why, action, threshold, rationale, "a split is empty"),
    });

    // 5. Outcome mix.
    let name = "How tests failed before the fix (build error vs failing test)".to_string();
    let why = "A build error usually means \u{201c}add the missing code\u{201d}; a failing test means \
               \u{201c}fix the behaviour\u{201d}. A different mix is a different kind of exam.";
    let action = "Pick final-exam tasks so the build-error / failing-test mix matches the practice set.";
    let threshold = format!("shares differ by at most {:.0} percentage points", MAX_OUTCOME_SHARE_DIFF * 100.0);
    let rationale = "a larger difference changes what kind of work the exam mostly asks for";
    checks.push(match max_diff(&e.outcome, &h.outcome).filter(|_| both) {
        Some((k, x, y)) => {
            let d = (x - y).abs();
            let ok = d <= MAX_OUTCOME_SHARE_DIFF;
            let key = "BuildFail\u{2192}Pass".to_string();
            let (bx, by) = (e.outcome_n.get(&key).copied().unwrap_or(0), h.outcome_n.get(&key).copied().unwrap_or(0));
            Check {
                id: "outcome", name, status: status(Some(ok)), why, action, threshold, rationale,
                detail: format!("{k}: evolve {}, held-out {} \u{2192} {:.0} points, {} max {:.0}",
                                pct(x), pct(y), d * 100.0, if ok { "within" } else { "above" }, MAX_OUTCOME_SHARE_DIFF * 100.0),
                meaning: format!("Before the fix, the tests did not compile for {} and for {} \u{2014} \
                                  {:.0} percentage points apart.",
                                 frac_ci(by, h.n, FINAL), frac(bx, e.n, PRACTICE), d * 100.0),
            }
        }
        None => na("outcome", name, why, action, threshold, rationale, "a split is empty"),
    });

    // 6. Date span.
    let name = "How many days of work the final exam covers".to_string();
    let why = "Tasks from one burst of work share context and often one feature, so one busy day \
               stands in for \u{201c}the future\u{201d}.";
    let action = "Hold out the newest ~2 weeks of work instead of a fixed number of tasks.";
    let threshold = format!("at least {} days between the oldest and newest final-exam commit", num(MIN_HELDOUT_SPAN_DAYS));
    let rationale = "less than a week of work is usually one feature or one burst of commits";
    checks.push(match (&h.span_days, &h.date_min, &h.date_max) {
        (Some(d), Some(a), Some(b)) => {
            let ok = *d >= MIN_HELDOUT_SPAN_DAYS;
            let (a, b) = (&a[..a.len().min(10)], &b[..b.len().min(10)]);
            Check {
                id: "span", name, status: status(Some(ok)), why, action, threshold, rationale,
                detail: format!("{a} .. {b} = {:.1} days, {} min {}", d, if ok { "at least" } else { "below" },
                                num(MIN_HELDOUT_SPAN_DAYS)),
                meaning: format!("All {} final-exam commits were made within {} ({a} to {b}, UTC); the {} \
                                  practice commits span {}.", h.n, span_words(*d), e.n,
                                 e.span_days.map(span_words).unwrap_or_else(|| "an unknown time".into())),
            }
        }
        _ => na("span", name, why, action, threshold, rationale, "the final exam has no commit dates (run with --repo)"),
    });

    let warnings = checks.iter().filter(|c| c.status == Status::Warn).count();
    let verdict = verdict(&checks, e.churn_median.zip(h.churn_median));
    Health {
        thresholds: Thresholds {
            churn_ratio: (CHURN_RATIO_MIN, CHURN_RATIO_MAX),
            max_heldout_area_share: MAX_HELDOUT_AREA_SHARE,
            max_tree_share_diff: MAX_TREE_SHARE_DIFF,
            max_outcome_share_diff: MAX_OUTCOME_SHARE_DIFF,
            min_heldout_span_days: MIN_HELDOUT_SPAN_DAYS,
        },
        top_areas: top,
        evolve: e,
        heldout: h,
        checks,
        warnings,
        verdict,
    }
}

/// The plain conclusion over the checks: how many warn, what that says
/// about the final exam, and a suggested fix.
pub fn verdict(checks: &[Check], medians: Option<(f64, f64)>) -> String {
    let warn = |id: &str| checks.iter().any(|c| c.id == id && c.status == Status::Warn);
    let n = checks.len();
    let w = checks.iter().filter(|c| c.status == Status::Warn).count();
    let na = checks.iter().filter(|c| c.status == Status::NotApplicable).count();
    if w == 0 {
        let tail = if na > 0 { format!(" ({na} could not be checked yet)") } else { String::new() };
        return format!("{} of {n} checks pass{tail}: the final exam (held-out) looks like the practice set \
                        (evolve), so a gap between their scores can be read as learning versus memorizing.", n - na);
    }
    let mut adj: Vec<&str> = Vec::new();
    if warn("area") || warn("component") || warn("tree") {
        adj.push("narrower");
    }
    if warn("churn") {
        adj.push(match medians { Some((e, h)) if h < e => "easier", _ => "harder" });
    }
    let mut what = if adj.is_empty() {
        "the final exam (held-out) differs from the practice set (evolve)".to_string()
    } else {
        format!("the final exam (held-out) is {} than the practice set (evolve)", adj.join(" and "))
    };
    if warn("outcome") {
        what.push_str(if adj.is_empty() { " in the kind of task" } else { " and asks for a different kind of task" });
    }
    if warn("span") {
        what.push_str(", and it covers only a very short stretch of work");
    }
    let mut fix: Vec<&str> = Vec::new();
    if warn("span") {
        fix.push("hold out the newest ~2 weeks of work");
    } else {
        fix.push("hold out more tasks");
    }
    if warn("area") || warn("component") || warn("outcome") {
        fix.push("balanced across subsystems and task kinds");
    }
    let mut suggestion = fix.join(", ");
    if warn("tree") || warn("churn") {
        suggestion.push_str(", or mine further back so both sets draw from the same code");
    }
    format!("{w} of {n} checks warn: {what}, so its score would be hard to interpret. Suggested fix: {suggestion}.")
}
