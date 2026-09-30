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

//! `rrsi-mine`: command-line front of the task miner (see lib.rs).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rrsi_mine::{candidates, mine, Docker};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Mine FAIL_TO_PASS-validated Go tasks from git history")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the candidate commits as JSON lines, without running any tests.
    List {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value = "2026-06-01")]
        since: String,
    },
    /// Validate candidates in containers and write the task directory.
    Mine {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "2026-06-01")]
        since: String,
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, env = "RRSI_GO_IMAGE", default_value = "golang:1.26.5")]
        image: String,
        #[arg(long, env = "RRSI_GO_MODCACHE", default_value = "rrsi-gomodcache")]
        modcache: String,
        #[arg(long, env = "RRSI_GO_BUILDCACHE", default_value = "rrsi-gobuildcache")]
        buildcache: String,
        #[arg(long, env = "RRSI_GO_TEST_TIMEOUT", default_value_t = 600)]
        test_timeout: u64,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::List { repo, since } => {
            for c in candidates(&repo, &since)? {
                println!("{}", serde_json::to_string(&c)?);
            }
        }
        Cmd::Mine { repo, out, since, jobs, limit, image, modcache, buildcache, test_timeout } => {
            let repo = repo.canonicalize().context("--repo")?;
            let mut cands = candidates(&repo, &since)?;
            if limit > 0 {
                cands.truncate(limit);
            }
            let docker = Docker { image: &image, modcache: &modcache, buildcache: &buildcache, test_timeout };
            mine(&repo, &out, cands, jobs, &docker)?;
        }
    }
    Ok(())
}
