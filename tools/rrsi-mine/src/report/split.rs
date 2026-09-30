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

//! The exam split, decided once here and shown by the report.
//!
//! A task is exam-ready when the miner found it valid (FAIL_TO_PASS) and,
//! once fairness has run for it, its gate passed. Of the exam-ready tasks the
//! newest `heldout` by committer date are held out and the rest evolve; ties
//! (and undated tasks, which count as oldest) are broken by sha ascending.
//! Every other task is excluded with the reason it is not ready.

use super::load::{Split, Task, STAGES};
use serde::Serialize;
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::BTreeMap;

/// The rule in one sentence, as the report states it.
pub fn rule_sentence(n: usize) -> String {
    format!("A commit becomes an exam question when its tests fail before the real fix and pass \
             after it and, once fairness checks have run, it passed all of them; the newest {n} \
             questions by commit date are the final exam (held-out), the rest are the practice set \
             (evolve), and every other commit is not used (excluded).")
}

#[derive(Debug, Default, PartialEq, Eq, Serialize)]
pub struct Splits {
    pub evolve: Vec<String>,
    pub heldout: Vec<String>,
    pub excluded: BTreeMap<String, String>,
}

fn stage_reason(v: &Value) -> String {
    v.get("reason").and_then(Value::as_str).unwrap_or("failed").to_string()
}

/// Why a task is not exam-ready, or `None` when it is.
pub fn not_ready(t: &Task) -> Option<String> {
    if t.pending_retry {
        return Some(format!("pending retry: {}", t.reason));
    }
    if !t.valid {
        return Some(t.reason.clone());
    }
    let gate = t.fairness.get("gate")?;
    if t.stage_pass("gate") == Some(true) {
        return None;
    }
    let failed = STAGES.iter().filter(|s| **s != "gate")
        .find(|s| t.stage_pass(s) == Some(false));
    Some(match failed {
        Some(s) => format!("fairness/{s}: {}", stage_reason(&t.fairness[*s])),
        None => format!("fairness/gate: {}", stage_reason(gate)),
    })
}

/// Assign every task its split and return the split lists (sha12s sorted).
pub fn assign(tasks: &mut [Task], heldout: usize) -> Splits {
    let mut ready: Vec<usize> = Vec::new();
    let mut out = Splits::default();
    for (i, t) in tasks.iter_mut().enumerate() {
        match not_ready(t) {
            Some(why) => {
                t.split = Split::Excluded;
                t.excluded_reason = Some(why.clone());
                out.excluded.insert(t.sha12.clone(), why);
            }
            None => {
                t.excluded_reason = None;
                ready.push(i);
            }
        }
    }
    // Newest first; undated last; sha ascending within a tie.
    ready.sort_by(|&a, &b| {
        let (x, y) = (&tasks[a], &tasks[b]);
        (x.ts.is_none(), Reverse(x.ts), &x.sha).cmp(&(y.ts.is_none(), Reverse(y.ts), &y.sha))
    });
    for (rank, &i) in ready.iter().enumerate() {
        let t = &mut tasks[i];
        if rank < heldout {
            t.split = Split::Heldout;
            out.heldout.push(t.sha12.clone());
        } else {
            t.split = Split::Evolve;
            out.evolve.push(t.sha12.clone());
        }
    }
    out.evolve.sort();
    out.heldout.sort();
    out
}

/// The funnel and rejection counts the report draws.
#[derive(Debug, Serialize)]
pub struct Summary {
    pub candidates: usize,
    pub valid: usize,
    pub pending_retry: usize,
    pub fairness_ran: usize,
    pub exam_ready: usize,
    pub evolve: usize,
    pub heldout: usize,
    /// Not valid (including pending retries).
    pub excluded_by_mine: usize,
    /// Valid but failed the fairness gate.
    pub excluded_by_fairness: usize,
    /// (label, kind "mine"|"fairness", count), largest first.
    pub rejections: Vec<(String, String, usize)>,
    /// Oldest held-out commit date (the timeline's cutoff rule).
    pub cutoff: Option<String>,
}

pub fn summarize(tasks: &[Task]) -> Summary {
    let mut rej: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in tasks {
        if t.pending_retry {
            *rej.entry((format!("pending retry: {}", t.reason), "mine".into())).or_default() += 1;
        } else if !t.valid {
            *rej.entry((t.reason.clone(), "mine".into())).or_default() += 1;
        }
        for s in STAGES {
            if t.stage_pass(s) == Some(false) {
                *rej.entry((format!("fairness/{s}"), "fairness".into())).or_default() += 1;
            }
        }
    }
    let mut rejections: Vec<_> = rej.into_iter().map(|((l, k), n)| (l, k, n)).collect();
    rejections.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let count = |s: Split| tasks.iter().filter(|t| t.split == s).count();
    let valid = tasks.iter().filter(|t| t.valid).count();
    let exam_ready = count(Split::Evolve) + count(Split::Heldout);
    let cutoff = tasks.iter().filter(|t| t.split == Split::Heldout)
        .filter_map(|t| t.ts.map(|ts| (ts, t.date.clone()))).min().and_then(|(_, d)| d);
    Summary {
        candidates: tasks.len(),
        valid,
        pending_retry: tasks.iter().filter(|t| t.pending_retry).count(),
        fairness_ran: tasks.iter().filter(|t| t.fairness_ran()).count(),
        exam_ready,
        evolve: count(Split::Evolve),
        heldout: count(Split::Heldout),
        excluded_by_mine: tasks.len() - valid,
        excluded_by_fairness: valid - exam_ready,
        rejections,
        cutoff,
    }
}

/// Up to two example tasks per split: the smallest-churn and the
/// median-churn task (ties by sha).
pub fn examples(tasks: &[Task]) -> BTreeMap<&'static str, Vec<String>> {
    let mut out = BTreeMap::new();
    for (name, s) in [("evolve", Split::Evolve), ("heldout", Split::Heldout), ("excluded", Split::Excluded)] {
        let mut ts: Vec<&Task> = tasks.iter().filter(|t| t.split == s).collect();
        ts.sort_by(|a, b| (a.src_churn, &a.sha).cmp(&(b.src_churn, &b.sha)));
        let mut picks: Vec<String> = Vec::new();
        for idx in [0, ts.len() / 2] {
            if let Some(t) = ts.get(idx) {
                if !picks.contains(&t.sha12) {
                    picks.push(t.sha12.clone());
                }
            }
        }
        out.insert(name, picks);
    }
    out
}
