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
//! parent→commit outcome transition, and the commit-date span. The checks compare held-out against evolve with the
//! fixed thresholds below; each yields `pass`, `warn`, or `n/a` when a split
//! is empty or undated.

use super::load::{Split, Task};
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
    /// Module tree -> share of this split's tasks (0..1).
    pub module_tree: BTreeMap<String, f64>,
    pub area: BTreeMap<String, f64>,
    pub component: BTreeMap<String, f64>,
    /// `area` folded to the report's top areas plus "other".
    pub area_top: BTreeMap<String, f64>,
    /// "BuildFail→Pass" etc. -> share.
    pub outcome: BTreeMap<String, f64>,
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
    pub name: &'static str,
    pub status: Status,
    /// The actual numbers and the threshold, in one line.
    pub detail: String,
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
        module_tree: shares(tasks.iter().map(|t| module_tree(&t.module_root)), n),
        area: shares(tasks.iter().map(|t| t.area.clone()), n),
        component: shares(tasks.iter().map(|t| component(&t.module_root, &t.packages)), n),
        area_top: shares(tasks.iter().map(|t| fold(&t.area)), n),
        outcome: shares(tasks.iter().map(|t| transition(t)), n),
        date_min: min.map(|d| d.1.clone()),
        date_max: max.map(|d| d.1.clone()),
        span_days: min.zip(max).map(|(a, b)| (b.0 - a.0) as f64 / 86_400.0),
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

fn check(name: &'static str, ok: Option<bool>, detail: String) -> Check {
    let status = match ok {
        Some(true) => Status::Pass,
        Some(false) => Status::Warn,
        None => Status::NotApplicable,
    };
    Check { name, status, detail }
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

    checks.push(match (e.churn_median, h.churn_median) {
        (Some(em), Some(hm)) if em > 0.0 => {
            let r = hm / em;
            let ok = (CHURN_RATIO_MIN..=CHURN_RATIO_MAX).contains(&r);
            check("median churn", Some(ok), format!(
                "evolve {}, held-out {} \u{2192} {:.2}x, {} {}\u{2013}{}x",
                num(em), num(hm), r, if ok { "within" } else { "outside" },
                CHURN_RATIO_MIN, CHURN_RATIO_MAX))
        }
        _ => check("median churn", None, "a split is empty".into()),
    });

    for (name, what, m) in [("held-out area concentration", "area", &h.area),
                             ("held-out component concentration", "component", &h.component)] {
        checks.push(match m.iter().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0))) {
            Some((a, s)) => {
                let ok = *s <= MAX_HELDOUT_AREA_SHARE;
                check(name, Some(ok), format!(
                    "largest held-out {what} {a}: {} of {} ({}), {} max {}",
                    (s * h.n as f64).round(), h.n, pct(*s), if ok { "within" } else { "above" },
                    pct(MAX_HELDOUT_AREA_SHARE)))
            }
            None => check(name, None, "held-out is empty".into()),
        });
    }

    for (name, a, b, limit) in [
        ("module-tree mix", &e.module_tree, &h.module_tree, MAX_TREE_SHARE_DIFF),
        ("outcome mix", &e.outcome, &h.outcome, MAX_OUTCOME_SHARE_DIFF),
    ] {
        checks.push(match max_diff(a, b).filter(|_| both) {
            Some((k, x, y)) => {
                let d = (x - y).abs();
                let ok = d <= limit;
                check(name, Some(ok), format!(
                    "{k}: evolve {}, held-out {} \u{2192} {:.0} points, {} max {:.0}",
                    pct(x), pct(y), d * 100.0, if ok { "within" } else { "above" }, limit * 100.0))
            }
            None => check(name, None, "a split is empty".into()),
        });
    }

    checks.push(match (&h.span_days, &h.date_min, &h.date_max) {
        (Some(d), Some(a), Some(b)) => {
            let ok = *d >= MIN_HELDOUT_SPAN_DAYS;
            check("held-out date span", Some(ok), format!(
                "{} .. {} = {:.1} days, {} min {}", &a[..a.len().min(10)], &b[..b.len().min(10)],
                d, if ok { "at least" } else { "below" }, num(MIN_HELDOUT_SPAN_DAYS)))
        }
        _ => check("held-out date span", None, "held-out has no commit dates (--repo)".into()),
    });

    let warnings = checks.iter().filter(|c| c.status == Status::Warn).count();
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
    }
}
