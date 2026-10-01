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

//! Shared, pure helpers for miners that read Claude Code session transcripts
//! (`~/.claude/projects/<project>/<session>.jsonl` plus nested subagent
//! transcripts): JSON field access, text normalization, similarity, stable
//! hashes and the transcript walk.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub fn field(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}

/// Text of a message `content` (a string or a list of blocks).
pub fn content_text(c: &Value) -> String {
    match c {
        Value::String(t) => t.clone(),
        Value::Array(bs) => bs.iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// Removes `<system-reminder>...</system-reminder>` blocks from a user turn.
pub fn strip_reminders(t: &str) -> String {
    let mut out = String::new();
    let mut rest = t;
    while let Some(a) = rest.find("<system-reminder>") {
        out.push_str(&rest[..a]);
        match rest[a..].find("</system-reminder>") {
            Some(b) => rest = &rest[a + b + "</system-reminder>".len()..],
            None => { rest = ""; }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

pub fn origin_kind(e: &Value) -> Option<String> {
    e.get("origin").and_then(|o| o.get("kind")).and_then(Value::as_str).map(str::to_string)
}

/// Lower-case words (letters, digits, apostrophes).
pub fn words(t: &str) -> Vec<String> {
    t.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}

pub const STOPWORDS: [&str; 40] = ["the", "and", "for", "you", "that", "this", "with", "are", "can",
    "what", "how", "was", "but", "not", "have", "has", "from", "your", "all", "any", "its", "it's",
    "just", "into", "out", "now", "then", "they", "them", "there", "here", "also", "use", "get",
    "did", "does", "will", "would", "should", "please"];

/// Content words of a turn: lower-case, 3+ characters, no stopwords.
pub fn content_words(t: &str) -> BTreeSet<String> {
    words(t).into_iter().filter(|w| w.chars().count() >= 3 && !STOPWORDS.contains(&w.as_str())).collect()
}

pub fn jaccard<T: Ord>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    a.intersection(b).count() as f64 / a.union(b).count() as f64
}

/// Character trigrams, for near-identical tool inputs.
pub fn trigrams(t: &str) -> BTreeSet<String> {
    let cs: Vec<char> = t.chars().collect();
    if cs.len() < 3 {
        return [t.to_string()].into();
    }
    cs.windows(3).map(|w| w.iter().collect()).collect()
}

/// FNV-1a 64: stable ids and content hashes with no extra dependency.
pub fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

pub fn truncate(t: &str, n: usize) -> String {
    let t = t.trim();
    if t.chars().count() <= n {
        return t.to_string();
    }
    let head: String = t.chars().take(n).collect();
    format!("{head}… [{} chars]", t.chars().count())
}

/// Every `*.jsonl` under `root`, as paths relative to it, sorted.
pub fn transcripts(root: &Path) -> Result<Vec<String>> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) -> Result<()> {
        for ent in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = ent?.path();
            if p.is_dir() {
                walk(&p, root, out)?;
            } else if p.extension().is_some_and(|e| e == "jsonl") {
                out.push(p.strip_prefix(root)?.to_string_lossy().into_owned());
            }
        }
        Ok(())
    }
    let mut out = vec![];
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

pub fn file_key(rel: &str) -> String {
    format!("{:016x}", fnv64(rel.as_bytes()))
}

pub fn mtime_size(p: &Path) -> Result<(u64, u64)> {
    let m = std::fs::metadata(p)?;
    let t = m.modified()?.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    Ok((t, m.len()))
}

/// Seconds since the epoch at the start of an ISO date (YYYY-MM-DD...), UTC.
pub fn since_epoch(since: &str) -> Option<u64> {
    let d = since.get(..10)?;
    let mut it = d.split('-').map(|p| p.parse::<i64>());
    let (y, m, day) = (it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?);
    // Days from civil (Howard Hinnant).
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    u64::try_from(days * 86400).ok()
}

pub fn root_default() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".claude/projects")
}

/// Seconds since the epoch of an ISO timestamp (`YYYY-MM-DDTHH:MM:SS...`, UTC).
pub fn iso_secs(ts: &str) -> Option<u64> {
    let day = since_epoch(ts)?;
    let t = ts.get(11..19)?;
    let mut it = t.split(':').map(|p| p.parse::<u64>());
    let (h, m, s) = (it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?);
    Some(day + h * 3600 + m * 60 + s)
}
