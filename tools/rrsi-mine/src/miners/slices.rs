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

//! Every merged slice PR becomes an exam candidate.
//!
//! ```text
//! rrsi-mine slices --repo PATH --out DIR [--since "2 weeks ago"] [--prs FILE]
//!                  [--gh-repo OWNER/NAME] [--ontology-score "CMD {tree}"] [--jobs 4] [--limit N]
//! ```
//!
//! A merged PR is a **slice PR** when its title names a slice (`Slice W1: …`,
//! `… (slice G1)`, `… (slice: zero leaked registrations)`, `… (S4)`,
//! `N1: …`), its body says "This PR is slice X", or its body has a `## Slice`
//! section, or a `## Proof` section and says "slice". Its commits on the mined revision (the squash commit, or the
//! branch commits of a merge commit) go through the same scan and FAIL_TO_PASS
//! validation as `git-history` (`crate::scan`, `crate::mine`); nothing else
//! is mined. Each resulting `<sha12>/task.json` gains a `slice` object: the
//! slice id, PR number and title, why it was classified as a slice, and
//! `ontology`, the signals the reference fix moved.
//!
//! Ontology signals come from an external scorer (`candace ontology score`
//! in candace-server) given as `--ontology-score`: a shell command run in a
//! detached checkout (a shared clone, so `git rev-parse HEAD` works; `{tree}`
//! is replaced by its quoted path) that prints one JSON object. Read are: a
//! `signals` list of `{"id", "count"}` (a null count is not measured and is
//! listed under `not_measured`), or numbers under a `signals`/`counts` object,
//! plus top-level `score` and `penalty`; without those keys, every top-level
//! number. It runs on the
//! parent and the commit tree of each valid task; `moved` holds the signals
//! whose value changed (after − before). Without a scorer, or when it fails,
//! `ontology.status` says so and why: nothing is guessed.
//!
//! PRs come from `--prs` (the JSON `gh pr list --json
//! number,title,body,mergeCommit,mergedAt,url` prints) or, without it, from
//! `gh pr list --state merged`. Records: `DIR/<sha12>/…` as `git-history`
//! writes them, plus `DIR/slices.jsonl` (one slice PR per line, with its
//! commits, candidates, valid tasks and rejections), `DIR/rejected.jsonl` and
//! `DIR/slices-summary.json`. `DIR` must be outside every git work tree.

use crate::miner::{parse_args, Miner};
use crate::{Candidate, Rejection};
use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

pub struct Slices;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    repo: PathBuf,
    out: PathBuf,
    #[serde(default = "since")]
    since: String,
    /// The revision whose history is mined (main's tip by default: HEAD).
    #[serde(default = "rev")]
    rev: String,
    /// `gh pr list --json ...` output; otherwise gh is asked.
    #[serde(default)]
    prs: Option<PathBuf>,
    #[serde(default)]
    gh_repo: Option<String>,
    #[serde(default = "pr_limit")]
    pr_limit: usize,
    #[serde(default)]
    ontology_score: Option<String>,
    /// Only select and write slices.jsonl; validate nothing.
    #[serde(default)]
    dry_run: bool,
    #[serde(default = "jobs")]
    jobs: usize,
    #[serde(default)]
    limit: usize,
    #[serde(default = "image")]
    image: String,
    #[serde(default = "modcache")]
    modcache: String,
    #[serde(default = "buildcache")]
    buildcache: String,
    #[serde(default = "test_timeout")]
    test_timeout: u64,
    #[serde(default)]
    csfc: Option<PathBuf>,
    #[serde(default)]
    csf_grammar: Option<PathBuf>,
    #[serde(default)]
    csf_source: Vec<String>,
    #[serde(default)]
    csf_model: Option<PathBuf>,
    #[serde(default)]
    csf_root: Option<String>,
    #[serde(default = "rev")]
    csf_model_rev: String,
}

fn since() -> String { "2 weeks ago".into() }
fn rev() -> String { "HEAD".into() }
fn pr_limit() -> usize { 300 }
fn jobs() -> usize { 4 }
fn image() -> String { "golang:1.26.5".into() }
fn modcache() -> String { "rrsi-gomodcache".into() }
fn buildcache() -> String { "rrsi-gobuildcache".into() }
fn test_timeout() -> u64 { 600 }

/// One merged pull request, as gh reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pr {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub body: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// `gh pr list --json number,title,body,mergeCommit,mergedAt,url` output
/// (or the same with snake_case keys) into PRs.
pub fn prs_from_json(v: &Value) -> Result<Vec<Pr>> {
    let Some(items) = v.as_array() else { bail!("PR list: expected a JSON array") };
    items.iter().map(|p| {
        let s = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
        let merge = p.get("mergeCommit").and_then(|m| m.get("oid").and_then(Value::as_str).or(m.as_str()))
            .or_else(|| p.get("merge_commit").and_then(Value::as_str)).map(str::to_string);
        Ok(Pr {
            number: p.get("number").and_then(Value::as_u64).context("PR without a number")?,
            title: s("title").unwrap_or_default(),
            body: s("body").unwrap_or_default(),
            merge_commit: merge,
            merged_at: s("mergedAt").or_else(|| s("merged_at")),
            url: s("url"),
        })
    }).collect()
}

/// Merged PRs from gh, newest first.
pub fn fetch_prs(repo: &Path, gh_repo: Option<&str>, limit: usize) -> Result<Vec<Pr>> {
    let mut cmd = Command::new("gh");
    cmd.current_dir(repo).args(["pr", "list", "--state", "merged", "--limit", &limit.to_string(),
                                "--json", "number,title,body,mergeCommit,mergedAt,url"]);
    if let Some(r) = gh_repo {
        cmd.args(["--repo", r]);
    }
    let out = cmd.output().context("running gh (pass --prs FILE to work without it)")?;
    if !out.status.success() {
        bail!("gh pr list failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    prs_from_json(&serde_json::from_slice(&out.stdout).context("parsing gh pr list output")?)
}

/// Why a PR is a slice PR, and which slice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceTag {
    /// The slice id (`S4`, `HX1`, `zero-leaked-runner-registrations`), or
    /// `pr-<N>` when the PR is a slice but names no id.
    pub id: String,
    pub pr: u64,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The rules that classified it: `title_slice`, `title_slice_tag`,
    /// `title_id`, `body_slice_section`, `body_proof_section`,
    /// `body_slice_statement` ("This PR is slice P"), `body_slice_id`.
    pub evidence: Vec<String>,
    /// The body has a Proof (or the PR template's Verification) section.
    pub proof_section: bool,
}

static TITLE_SLICE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?i:slice)\b[\s:-]*(?:([A-Z]{1,4}\d{0,3}[a-z]?)(?:\s*:|\s|$))?").unwrap());
static TITLE_SLICE_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\(\s*slice\s*:?\s*([^)]+?)\s*\)").unwrap());
static ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z]{1,4}\d{1,3}[a-z]?$").unwrap());
static TITLE_ID_PAREN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(([A-Z]{1,4}\d{1,3}[a-z]?)(?:,[^)]*)?\)").unwrap());
static TITLE_ID_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*([A-Z]{1,4}\d{1,3}[a-z]?):\s").unwrap());
static SLICE_HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?im)^#{1,4}\s*slice\b[^\n]*$").unwrap());
static PROOF_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?im)^#{1,4}\s*(?:proof|verification)\b").unwrap());
static PROOF_ONLY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?im)^#{1,4}\s*proof\b").unwrap());
static BODY_SLICE_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bslice[ \t]+([A-Z]{1,4}\d{1,3}[a-z]?)\b").unwrap());
static BODY_STATEMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bthis PR is (?:the )?slice\s+([A-Za-z0-9][A-Za-z0-9_-]*)").unwrap());
static SLICE_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bslice\b").unwrap());

/// A lowercase, hyphenated id from a slice's name.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The slice a merged PR delivers, if it is a slice PR at all.
pub fn classify(pr: &Pr) -> Option<SliceTag> {
    let (title, body) = (pr.title.as_str(), pr.body.as_str());
    let mut evidence = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    if let Some(c) = TITLE_SLICE.captures(title) {
        evidence.push("title_slice".to_string());
        ids.extend(c.get(1).map(|m| m.as_str().to_string()));
    }
    if let Some(c) = TITLE_SLICE_TAG.captures(title) {
        evidence.push("title_slice_tag".to_string());
        let name = c[1].trim();
        ids.push(if ID.is_match(name) { name.to_string() } else { slug(name) });
    }
    if let Some(c) = TITLE_ID_PAREN.captures(title).or_else(|| TITLE_ID_PREFIX.captures(title)) {
        evidence.push("title_id".to_string());
        ids.push(c[1].to_string());
    }
    if let Some(m) = SLICE_HEADING.find(body) {
        evidence.push("body_slice_section".to_string());
        // The id is the heading's own text after "Slice", else the section's first line.
        let rest = m.as_str().trim_start_matches('#').trim()[5..].trim_matches(|c: char| c == ':' || c.is_whitespace());
        let first = body[m.end()..].lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
        let named = if rest.is_empty() { first } else { rest };
        let token = named.split(|c: char| c.is_whitespace() || c == ':').find(|t| !t.is_empty()).unwrap_or("");
        let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if ID.is_match(token) {
            ids.push(token.to_string());
        } else if !named.is_empty() {
            ids.push(slug(named.split(['.', ':']).next().unwrap_or(named)));
        }
    }
    if PROOF_ONLY.is_match(body) && SLICE_WORD.is_match(body) && !evidence.iter().any(|e| e == "body_slice_section") {
        evidence.push("body_proof_section".to_string());
    }
    if let Some(c) = BODY_STATEMENT.captures(body) {
        evidence.push("body_slice_statement".to_string());
        ids.push(c[1].to_string());
    }
    if let Some(c) = BODY_SLICE_ID.captures(body) {
        if !evidence.is_empty() {
            evidence.push("body_slice_id".to_string());
            ids.push(c[1].to_string());
        }
    }
    if evidence.is_empty() {
        return None;
    }
    // A short code (S4) beats a slugged name; the first found wins otherwise.
    let id = ids.iter().find(|i| ID.is_match(i)).or(ids.first()).filter(|i| !i.is_empty()).cloned()
        .unwrap_or_else(|| format!("pr-{}", pr.number));
    Some(SliceTag { id, pr: pr.number, title: pr.title.clone(), url: pr.url.clone(), evidence,
                    proof_section: PROOF_HEADING.is_match(body) })
}

/// The commits `pr` put on `rev` within `window` (shas reachable from `rev`
/// since the mining date): its squash commit, or the branch commits of its
/// merge commit. Without a known merge commit, the first-parent commit whose
/// subject ends in `(#N)` is used.
pub fn commits_of(repo: &Path, pr: &Pr, subjects: &HashMap<u64, String>, window: &HashSet<String>)
    -> Result<Vec<String>> {
    let merge = pr.merge_commit.clone().filter(|m| window.contains(m)).or_else(|| subjects.get(&pr.number).cloned());
    let Some(merge) = merge else { return Ok(vec![]) };
    let parents = crate::git(repo, &["rev-list", "--parents", "-n", "1", &merge])?;
    if parents.split_whitespace().count() <= 2 {
        return Ok(vec![merge]);
    }
    let range = format!("{merge}^1..{merge}^2");
    Ok(crate::git(repo, &["rev-list", "--no-merges", &range])?.split_whitespace()
        .filter(|s| window.contains(*s)).map(str::to_string).collect())
}

static PR_SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(#(\d+)\)\s*$").unwrap());
static MERGE_SUBJECT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^Merge pull request #(\d+) ").unwrap());

/// Every sha reachable from `rev` since `since`, and the PR number each
/// first-parent commit names in its subject (`… (#N)` or `Merge pull request #N`).
pub fn window(repo: &Path, rev: &str, since: &str) -> Result<(HashSet<String>, HashMap<u64, String>)> {
    let since_arg = format!("--since={since}");
    let all = crate::git(repo, &["rev-list", &since_arg, rev])?.split_whitespace().map(str::to_string).collect();
    let mut subjects = HashMap::new();
    for line in crate::git(repo, &["log", "--first-parent", &since_arg, "--format=%H %s", rev])?.lines() {
        let Some((sha, subject)) = line.split_once(' ') else { continue };
        let n = PR_SUFFIX.captures(subject).or_else(|| MERGE_SUBJECT.captures(subject))
            .and_then(|c| c[1].parse::<u64>().ok());
        if let Some(n) = n {
            subjects.entry(n).or_insert_with(|| sha.to_string());
        }
    }
    Ok((all, subjects))
}

/// One slice PR and what mining made of it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SliceRecord {
    #[serde(flatten)]
    pub tag: SliceTag,
    /// Its commits on the mined revision within the window.
    pub commits: Vec<String>,
    /// Commits that became candidates.
    pub candidates: Vec<String>,
    pub rejected: Vec<Rejection>,
    /// Commits with no Go test change (never scanned as candidates).
    pub no_test_change: Vec<String>,
    /// Candidates that validated FAIL_TO_PASS (filled after mining).
    #[serde(default)]
    pub valid: Vec<String>,
}

/// Which PRs are slices, and which of their commits are candidates.
pub struct Selection {
    pub prs_seen: usize,
    pub slices: Vec<SliceRecord>,
    pub candidates: Vec<(Candidate, SliceTag)>,
}

/// Classify `prs`, find their commits in `rev`'s history since `since` and
/// keep the scan's candidates among them (`scanned` is `crate::scan`'s
/// output over the same window).
pub fn select(repo: &Path, rev: &str, since: &str, prs: &[Pr], scanned: (Vec<Candidate>, Vec<Rejection>))
    -> Result<Selection> {
    let (in_window, subjects) = window(repo, rev, since)?;
    let (cands, rejections) = scanned;
    let cand_by_sha: HashMap<&str, &Candidate> = cands.iter().map(|c| (c.sha.as_str(), c)).collect();
    let rej_by_sha: HashMap<&str, &Rejection> = rejections.iter().map(|r| (r.sha.as_str(), r)).collect();
    let mut sel = Selection { prs_seen: 0, slices: vec![], candidates: vec![] };
    let mut taken = HashSet::new();
    for pr in prs {
        let commits = commits_of(repo, pr, &subjects, &in_window)?;
        if commits.is_empty() {
            continue; // merged outside the window or not on `rev`
        }
        sel.prs_seen += 1;
        let Some(tag) = classify(pr) else { continue };
        let mut rec = SliceRecord { tag: tag.clone(), commits: commits.clone(), candidates: vec![], rejected: vec![],
                                    no_test_change: vec![], valid: vec![] };
        for sha in &commits {
            if let Some(c) = cand_by_sha.get(sha.as_str()) {
                rec.candidates.push(sha.clone());
                if taken.insert(sha.clone()) {
                    sel.candidates.push(((*c).clone(), tag.clone()));
                }
            } else if let Some(r) = rej_by_sha.get(sha.as_str()) {
                rec.rejected.push((*r).clone());
            } else {
                rec.no_test_change.push(sha.clone());
            }
        }
        sel.slices.push(rec);
    }
    Ok(sel)
}

/// The numeric signals of one scorer record (see the module docs).
pub fn signals_of(v: &Value) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let nested = ["signals", "counts"].iter().find_map(|k| v.get(*k)).filter(|x| x.is_object() || x.is_array());
    let count = |x: &Value| x.as_f64().or_else(|| x.get("count").and_then(Value::as_f64));
    match nested.or(Some(v)) {
        // `candace ontology score`: [{"id": "CS-16", "count": 2 | null, ...}]
        Some(Value::Array(items)) => {
            for item in items {
                if let (Some(id), Some(n)) = (item.get("id").and_then(Value::as_str), count(item)) {
                    out.insert(id.to_string(), n);
                }
            }
        }
        Some(Value::Object(m)) => {
            for (k, x) in m {
                if let Some(n) = count(x) {
                    out.insert(k.clone(), n);
                }
            }
        }
        _ => {}
    }
    if nested.is_some() {
        for k in ["score", "penalty"] {
            if let Some(n) = v.get(k).and_then(Value::as_f64) {
                out.insert(k.into(), n);
            }
        }
    }
    out
}

/// The ids of signals the scorer listed without a count (not measured).
pub fn unmeasured_of(v: &Value) -> Vec<String> {
    let Some(Value::Array(items)) = v.get("signals") else { return vec![] };
    items.iter().filter(|i| i.get("count").is_some_and(Value::is_null))
        .filter_map(|i| i.get("id").and_then(Value::as_str).map(str::to_string)).collect()
}

/// Signals whose value differs between `before` and `after` (after − before;
/// a signal present on one side only counts as 0 on the other).
pub fn moved(before: &BTreeMap<String, f64>, after: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let keys: std::collections::BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    keys.into_iter().filter_map(|k| {
        let d = after.get(k).copied().unwrap_or(0.0) - before.get(k).copied().unwrap_or(0.0);
        (d != 0.0).then(|| (k.clone(), d))
    }).collect()
}

fn shell_quote(p: &Path) -> String {
    format!("'{}'", p.display().to_string().replace('\'', r"'\''"))
}

/// Run the scorer `cmd` in `tree` and return the JSON record it printed.
pub fn score_tree(cmd: &str, tree: &Path) -> Result<Value> {
    let line = cmd.replace("{tree}", &shell_quote(tree));
    let out = Command::new("sh").arg("-c").arg(&line).current_dir(tree).output()
        .with_context(|| format!("running {line}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("exit {}: {}", out.status.code().unwrap_or(-1), err.trim().chars().take(300).collect::<String>());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // The record is the last line that parses as a JSON object (scorers may log first).
    let v = text.lines().rev().find_map(|l| serde_json::from_str::<Value>(l.trim()).ok().filter(Value::is_object))
        .or_else(|| serde_json::from_str::<Value>(text.trim()).ok().filter(Value::is_object))
        .context("the scorer printed no JSON object")?;
    Ok(v)
}

/// A detached checkout of `sha` at `<tempdir>/t`: a shared clone (objects
/// borrowed from `repo`, which is never modified), because a scorer may ask
/// git for the tree's revision and tracked files.
pub fn checkout(repo: &Path, sha: &str) -> Result<tempfile::TempDir> {
    let d = tempfile::Builder::new().prefix("rrsi-slice-").tempdir()?;
    let t = d.path().join("t");
    let t = t.to_str().context("temp path")?;
    let repo = repo.to_str().context("repo path")?;
    crate::git(d.path(), &["clone", "--quiet", "--shared", "--no-checkout", repo, t])?;
    crate::git(Path::new(t), &["-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", sha])?;
    Ok(d)
}

/// The `ontology` object for one task: the signals its fix moved.
pub fn ontology(cmd: Option<&str>, repo: &Path, cand: &Candidate, valid: bool) -> Value {
    let Some(cmd) = cmd else {
        return json!({"status": "unavailable", "reason": "no ontology scorer given (--ontology-score)"});
    };
    if !valid {
        return json!({"status": "skipped", "reason": "task is not FAIL_TO_PASS valid"});
    }
    let run = || -> Result<Value> {
        let (parent, commit) = (checkout(repo, &cand.parent)?, checkout(repo, &cand.sha)?);
        let before = score_tree(cmd, &parent.path().join("t")).context("parent tree")?;
        let after = score_tree(cmd, &commit.path().join("t")).context("commit tree")?;
        let (b, a) = (signals_of(&before), signals_of(&after));
        Ok(json!({"status": "measured", "command": cmd, "moved": moved(&b, &a), "before": b, "after": a,
                  "not_measured": unmeasured_of(&after)}))
    };
    run().unwrap_or_else(|e| json!({"status": "error", "command": cmd, "reason": format!("{e:#}")}))
}

/// Add `slice` (with `ontology`) to the task record of `cand` under `out`:
/// task.json, or retry.json when validation was inconclusive. Returns
/// whether the task is valid. A measured ontology already present is kept.
pub fn tag_task(out: &Path, repo: &Path, cand: &Candidate, tag: &SliceTag, scorer: Option<&str>)
    -> Result<Option<bool>> {
    let dir = out.join(&cand.sha[..12]);
    let Some(path) = ["task.json", "retry.json"].iter().map(|f| dir.join(f)).find(|p| p.is_file()) else {
        return Ok(None);
    };
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    let valid = v.get("valid").and_then(Value::as_bool).unwrap_or(false);
    let kept = v.pointer("/slice/ontology").filter(|o| o.get("status").and_then(Value::as_str) == Some("measured"))
        .cloned();
    let mut s = serde_json::to_value(tag)?;
    s["ontology"] = kept.unwrap_or_else(|| ontology(scorer, repo, cand, valid));
    v["slice"] = s;
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    Ok(Some(valid))
}

fn write_jsonl<T: Serialize>(path: &Path, rows: impl IntoIterator<Item = T>) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for r in rows {
        writeln!(f, "{}", serde_json::to_string(&r)?)?;
    }
    Ok(())
}

impl Miner for Slices {
    fn name(&self) -> &'static str { "slices" }
    fn about(&self) -> &'static str {
        "FAIL_TO_PASS Go tasks from merged slice PRs, tagged with the slice id and the ontology signals each moved"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("repo", "the git repository"), ("out", "output directory (outside every work tree)"),
          ("since", "first commit date, default \"2 weeks ago\""), ("rev", "revision mined, default HEAD"),
          ("prs", "gh pr list --json number,title,body,mergeCommit,mergedAt,url output (else gh is run)"),
          ("gh_repo", "OWNER/NAME for gh (else gh infers it from repo)"), ("pr_limit", "PRs gh lists, default 300"),
          ("ontology_score", "scorer shell command run in each tree, {tree} = its path (optional)"),
          ("dry_run", "select slice PRs and candidates only; validate nothing"),
          ("jobs", "parallel validations, default 4"), ("limit", "at most this many candidates, 0 = all"),
          ("image", "Go image, default golang:1.26.5"), ("modcache", "module cache volume"),
          ("buildcache", "build cache volume"), ("test_timeout", "seconds per go test, default 600"),
          ("csfc", "csfc binary (optional)"), ("csf_grammar", "grammar file for csfc (optional)"),
          ("csf_source", "extra architecture source paths (optional)"),
          ("csf_model", "csfc emit --format json model (optional)"),
          ("csf_root", "directory the csf_model paths are relative to (optional)"),
          ("csf_model_rev", "revision csfc reads, default HEAD")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("<sha12>/task.json", "a git-history task plus `slice`: id, PR, evidence, ontology signals moved"),
          ("slices.jsonl", "one merged slice PR: its commits, candidates, valid tasks and rejections"),
          ("rejected.jsonl", "slice commits that touched a test but are not candidates, with the reason"),
          ("slices-summary.json", "counts: PRs seen, slice PRs, commits, candidates, valid tasks")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = parse_args(self.name(), args)?;
        let repo = a.repo.canonicalize().context("repo")?;
        crate::miner::ensure_private_out(self.name(), &a.out)?;
        let prs = match &a.prs {
            Some(f) => prs_from_json(&serde_json::from_str(&std::fs::read_to_string(f).context("prs")?)?)?,
            None => fetch_prs(&repo, a.gh_repo.as_deref(), a.pr_limit)?,
        };
        let model = a.csf_model.as_deref().map(|f| (f, a.csf_root.as_deref().unwrap_or("")));
        let csf = crate::csf::MineCsf::resolve(&repo, &a.csf_model_rev, a.csfc.as_deref(), a.csf_grammar.as_deref(),
                                               &a.csf_source, model)?;
        let scanned = crate::scan_rev(&repo, &a.rev, &a.since, &csf)?;
        let mut sel = select(&repo, &a.rev, &a.since, &prs, scanned)?;
        if a.limit > 0 {
            sel.candidates.truncate(a.limit);
        }
        std::fs::create_dir_all(&a.out)?;
        let mut valid_shas = HashSet::new();
        let mut tagged = 0;
        if !a.dry_run && !sel.candidates.is_empty() {
            let docker = crate::Docker { image: &a.image, modcache: &a.modcache, buildcache: &a.buildcache,
                                         test_timeout: a.test_timeout };
            let cands: Vec<Candidate> = sel.candidates.iter().map(|(c, _)| c.clone()).collect();
            crate::mine(&repo, &a.out, cands, a.jobs, &docker, &csf)?;
            for (c, tag) in &sel.candidates {
                if let Some(valid) = tag_task(&a.out, &repo, c, tag, a.ontology_score.as_deref())? {
                    tagged += 1;
                    if valid {
                        valid_shas.insert(c.sha.clone());
                    }
                }
            }
        }
        for s in &mut sel.slices {
            s.valid = s.candidates.iter().filter(|c| valid_shas.contains(*c)).cloned().collect();
        }
        write_jsonl(&a.out.join("slices.jsonl"), &sel.slices)?;
        write_jsonl(&a.out.join("rejected.jsonl"), sel.slices.iter().flat_map(|s| s.rejected.iter()))?;
        let summary = json!({
            "since": a.since, "rev": a.rev, "prs_seen": sel.prs_seen, "slice_prs": sel.slices.len(),
            "slice_commits": sel.slices.iter().map(|s| s.commits.len()).sum::<usize>(),
            "candidates": sel.candidates.len(), "tagged": tagged, "valid": valid_shas.len(),
            "rejected": sel.slices.iter().map(|s| s.rejected.len()).sum::<usize>(),
            "slices_with_valid_task": sel.slices.iter().filter(|s| !s.valid.is_empty()).count(),
            "dry_run": a.dry_run, "ontology_scorer": a.ontology_score, "out": a.out,
        });
        std::fs::write(a.out.join("slices-summary.json"), serde_json::to_string_pretty(&summary)?)?;
        Ok(summary)
    }
}

#[cfg(test)]
mod tests;
