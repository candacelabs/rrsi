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

//! The git-history miner: candidate commits for any toolchain.
//!
//! A commit is a candidate when it changes source and tests of one project
//! (as the toolchain classifies and roots them) within the Go miner's size
//! limits (at most MAX_SRC_CHURN changed source lines in at most
//! MAX_SRC_PACKAGES directories) and its changed tests map to at least one
//! test unit. Files of other projects are neither fix nor tests (they stay in
//! the commit tree only), as the Go miner ignores non-Go files.
//!
//! Go keeps its original rules ([`crate::candidates`]: source and tests in
//! the same package of one module), so its listing is byte-for-byte what it
//! was before toolchains existed.

use crate::toolchain::{tracked_files, under, Changed, FileKind, Toolchain, Toolchains, REGISTRY};
use crate::{dir_of, git, Candidate, MAX_SRC_CHURN, MAX_SRC_PACKAGES};
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

/// The candidates of `tc` since `since`, newest first.
pub fn candidates(repo: &Path, since: &str, tc: &dyn Toolchain) -> Result<Vec<Candidate>> {
    if tc.name() == "go" {
        return crate::candidates(repo, since);
    }
    let since_arg = format!("--since={since}");
    let log = git(repo, &["log", &since_arg, "--no-merges", "--format=%H"])?;
    let mut out = Vec::new();
    for sha in log.split_whitespace() {
        let mut src: Vec<(u64, String)> = Vec::new();
        let mut tests: Vec<String> = Vec::new();
        for line in git(repo, &["show", "--numstat", "--format=", "--no-renames", sha])?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 3 {
                continue;
            }
            let churn = f[0].parse::<u64>().unwrap_or(0) + f[1].parse::<u64>().unwrap_or(0);
            match tc.classify_file(f[2]) {
                FileKind::Source => src.push((churn, f[2].to_string())),
                FileKind::Test => tests.push(f[2].to_string()),
                FileKind::Generated | FileKind::Other => {}
            }
        }
        if src.is_empty() || tests.is_empty() {
            continue;
        }
        let tracked: BTreeSet<String> = tracked_files(repo, sha)?.into_iter().collect();
        let mut ch = Changed { src, tests, tracked };
        let Some(root) = tc.project_root(&ch) else { continue };
        ch.src.retain(|(_, f)| under(&root, f));
        ch.tests.retain(|f| under(&root, f));
        if ch.src.is_empty() {
            continue;
        }
        let churn: u64 = ch.src.iter().map(|(c, _)| c).sum();
        let dirs: BTreeSet<String> = ch.src.iter().map(|(_, f)| dir_of(f)).collect();
        if churn > MAX_SRC_CHURN || dirs.len() > MAX_SRC_PACKAGES {
            continue;
        }
        let read = |path: &str| git(repo, &["show", &format!("{sha}:{path}")]);
        let units = tc.units(&ch, &root, &read)?;
        if units.is_empty() {
            continue;
        }
        out.push(Candidate {
            sha: sha.to_string(),
            parent: git(repo, &["rev-parse", &format!("{sha}^")])?.trim().to_string(),
            subject: git(repo, &["log", "-1", "--format=%s", sha])?.trim().to_string(),
            body: git(repo, &["log", "-1", "--format=%b", sha])?.trim().to_string(),
            module_root: root,
            packages: units,
            src_files: ch.src.into_iter().map(|(_, f)| f).collect(),
            test_files: ch.tests,
            src_churn: churn,
            toolchain: Some(tc.name().to_string()),
        });
    }
    Ok(out)
}

/// The toolchains `--toolchain` selects for `repo`: one by name, or for
/// `auto` every registered toolchain with a project root at HEAD, in
/// registry order.
pub fn select<'a>(tcs: &'a Toolchains, repo: &Path, choice: &str) -> Result<Vec<&'a dyn Toolchain>> {
    if choice != "auto" {
        return Ok(vec![tcs.get(Some(choice))?]);
    }
    let mut out = Vec::new();
    for r in REGISTRY {
        let t = tcs.get(Some(r.name))?;
        if !t.detect(repo, "HEAD")?.is_empty() {
            out.push(t);
        }
    }
    Ok(out)
}

/// The candidates of every selected toolchain, newest first per toolchain;
/// a commit two toolchains both claim goes to the first.
pub fn candidates_for(tcs: &Toolchains, repo: &Path, since: &str, choice: &str) -> Result<Vec<Candidate>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for t in select(tcs, repo, choice)? {
        for c in candidates(repo, since, t)? {
            if seen.insert(c.sha.clone()) {
                out.push(c);
            }
        }
    }
    Ok(out)
}
