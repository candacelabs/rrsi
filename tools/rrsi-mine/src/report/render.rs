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

//! Fill report/report.html with the tasks, the split and the summary as one
//! embedded JSON document.

use super::load::Task;
use super::split::{examples, rule_sentence, summarize, Splits};
use anyhow::{ensure, Result};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

const TEMPLATE: &str = include_str!("../../report/report.html");
const DATA_SLOT: &str = "__RRSI_DATA__";

/// JSON safe to place between `<script type="application/json">` tags: no
/// `<`, `>` or `&` can close the element or open a comment, and the JS line
/// separators are escaped too. The escapes are valid JSON string escapes, so
/// `JSON.parse` returns the original text.
pub fn embed_json(v: &serde_json::Value) -> String {
    let raw = serde_json::to_string(v).expect("a serde_json::Value always serializes");
    let mut out = String::with_capacity(raw.len() + raw.len() / 16);
    for c in raw.chars() {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c => out.push(c),
        }
    }
    out
}

/// The whole page.
pub fn render(tasks: &[Task], splits: &Splits, heldout: usize, exam_lines: Option<usize>,
              generated_at: &str) -> Result<String> {
    let data = json!({
        "generated_at": generated_at,
        "heldout_n": heldout,
        "rule": rule_sentence(heldout),
        "has_dates": tasks.iter().any(|t| t.ts.is_some()),
        "exam_lines": exam_lines,
        "summary": summarize(tasks),
        "splits": splits,
        "examples": examples(tasks),
        "tasks": tasks,
    });
    let (head, tail) = TEMPLATE.split_once(DATA_SLOT)
        .expect("report.html holds the data slot");
    ensure!(!tail.contains(DATA_SLOT), "report.html holds the data slot twice");
    Ok(format!("{head}{}{tail}", embed_json(&data)))
}

/// Now as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn now_utc() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    iso_utc(secs as i64)
}

/// Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` (proleptic Gregorian, UTC).
pub fn iso_utc(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}
