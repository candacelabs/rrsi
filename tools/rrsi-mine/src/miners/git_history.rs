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

//! The git-history miner as a plugin: FAIL_TO_PASS-validated Go tasks from
//! a repository's commits (the library in `lib.rs`; `rrsi-mine mine` is the
//! same thing with typed flags).

use crate::miner::{parse_args, Miner};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct GitHistory;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    repo: PathBuf,
    out: PathBuf,
    #[serde(default = "since")]
    since: String,
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
}

fn since() -> String { "2026-06-01".into() }
fn jobs() -> usize { 4 }
fn image() -> String { "golang:1.26.5".into() }
fn modcache() -> String { "rrsi-gomodcache".into() }
fn buildcache() -> String { "rrsi-gobuildcache".into() }
fn test_timeout() -> u64 { 600 }

impl Miner for GitHistory {
    fn name(&self) -> &'static str { "git-history" }
    fn about(&self) -> &'static str {
        "Go tasks from commits whose tests fail before and pass after (FAIL_TO_PASS, in sealed containers)"
    }
    fn inputs(&self) -> &'static [(&'static str, &'static str)] {
        &[("repo", "the git repository"), ("out", "output directory (outside every work tree)"),
          ("since", "first commit date, default 2026-06-01"), ("jobs", "parallel validations, default 4"),
          ("limit", "at most this many candidates, 0 = all"), ("image", "Go image, default golang:1.26.5"),
          ("modcache", "module cache volume"), ("buildcache", "build cache volume"),
          ("test_timeout", "seconds per go test, default 600")]
    }
    fn records(&self) -> &'static [(&'static str, &'static str)] {
        &[("index.jsonl", "one validated or rejected candidate commit per line"),
          ("<sha12>/task.json", "the task: commit, packages, FAIL_TO_PASS verdict"),
          ("<sha12>/{src,tests}.patch", "the reference fix and the hidden tests")]
    }
    fn run(&self, args: Value) -> Result<Value> {
        let a: Args = parse_args(self.name(), args)?;
        let repo = a.repo.canonicalize().context("repo")?;
        let mut cands = crate::candidates(&repo, &a.since)?;
        if a.limit > 0 {
            cands.truncate(a.limit);
        }
        let n = cands.len();
        let docker = crate::Docker { image: &a.image, modcache: &a.modcache, buildcache: &a.buildcache,
                                     test_timeout: a.test_timeout };
        crate::mine(&repo, &a.out, cands, a.jobs, &docker)?;
        Ok(json!({"candidates": n, "out": a.out}))
    }
}
