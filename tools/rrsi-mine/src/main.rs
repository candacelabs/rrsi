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
use clap::{Args, Parser, Subcommand};
use rrsi_mine::fairness::{self as fair, Task};
use rrsi_mine::llm::{Copilot, ProcessRunner};
use rrsi_mine::{candidates, miner, mine, Docker};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(version, about = "Mine FAIL_TO_PASS-validated Go tasks from git history; run any registered miner (`rrsi-mine miners`)")]
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
        #[command(flatten)]
        go: GoArgs,
    },
    /// Fairness 1: re-run the commit tree's tests; every run must pass.
    Flake {
        #[command(flatten)]
        stage: StageArgs,
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[command(flatten)]
        go: GoArgs,
    },
    /// Fairness 2: the API the hidden tests need that the parent lacks.
    Api {
        #[command(flatten)]
        stage: StageArgs,
    },
    /// Fairness 3: write instruction.md with an LLM and check it leaks nothing.
    Describe {
        #[command(flatten)]
        stage: StageArgs,
        /// Show the writer the parent's package source (never the fix).
        #[arg(long)]
        repo: Option<PathBuf>,
        #[command(flatten)]
        llm: LlmArgs,
    },
    /// Fairness 4: a second model judges instruction.md on its own.
    Probe {
        #[command(flatten)]
        stage: StageArgs,
        #[arg(long)]
        repo: PathBuf,
        #[command(flatten)]
        llm: LlmArgs,
    },
    /// Fairness 5: assertions pinning strings, private names or call counts.
    Specificity {
        #[command(flatten)]
        stage: StageArgs,
        /// Read the parent's package source so pre-existing strings are not pinned.
        #[arg(long)]
        repo: Option<PathBuf>,
    },
    /// Fairness 6: exam_ready per task, <tasks>/exam.jsonl and a summary.
    Gate {
        #[command(flatten)]
        stage: StageArgs,
    },
    /// List the registered miners (name, inputs, records) as JSON.
    Miners,
    /// Any registered miner: `rrsi-mine <name> --key value ...` (see `miners`).
    #[command(external_subcommand)]
    Miner(Vec<String>),
    /// Fairness stages 1-6 in order.
    Fairness {
        #[command(flatten)]
        stage: StageArgs,
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[command(flatten)]
        go: GoArgs,
        #[command(flatten)]
        llm: LlmArgs,
    },
}

#[derive(Args)]
struct StageArgs {
    /// A task directory written by `mine`.
    #[arg(long)]
    tasks: PathBuf,
    /// Only the task whose directory name starts with this (a sha12).
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value_t = 4)]
    jobs: usize,
    /// Redo verdicts that are already fresh.
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct GoArgs {
    #[arg(long, env = "RRSI_GO_IMAGE", default_value = "golang:1.26.5")]
    image: String,
    #[arg(long, env = "RRSI_GO_MODCACHE", default_value = "rrsi-gomodcache")]
    modcache: String,
    #[arg(long, env = "RRSI_GO_BUILDCACHE", default_value = "rrsi-gobuildcache")]
    buildcache: String,
    #[arg(long, env = "RRSI_GO_TEST_TIMEOUT", default_value_t = 600)]
    test_timeout: u64,
}

impl GoArgs {
    fn docker(&self) -> Docker<'_> {
        Docker { image: &self.image, modcache: &self.modcache, buildcache: &self.buildcache,
                 test_timeout: self.test_timeout }
    }
}

#[derive(Args)]
struct LlmArgs {
    /// The model that writes instruction.md.
    #[arg(long, env = "RRSI_DESCRIBE_MODEL", default_value = "claude-sonnet-5.5")]
    describe_model: String,
    /// The independent reviewer model; must differ from the writer.
    #[arg(long, env = "RRSI_PROBE_MODEL", default_value = "gpt-5.4")]
    probe_model: String,
    #[arg(long, env = "RRSI_CLI_REASONING", default_value = "low")]
    reasoning: String,
    #[arg(long, env = "RRSI_CLI_TIMEOUT", default_value_t = 900)]
    llm_timeout: u64,
}

impl LlmArgs {
    fn copilot(&self, model: &str) -> Copilot<'static> {
        Copilot { runner: &ProcessRunner, model: model.to_string(), reasoning: self.reasoning.clone(),
                  timeout: Duration::from_secs(self.llm_timeout) }
    }
}

fn load(s: &StageArgs) -> Result<Vec<Task>> {
    fair::load_tasks(&s.tasks, s.only.as_deref())
}

fn canonical(repo: &Path) -> Result<PathBuf> {
    repo.canonicalize().context("--repo")
}

fn gate(s: &StageArgs, tasks: &[Task]) -> Result<()> {
    fair::run_stage("gate", tasks, s.jobs, s.force, fair::gate)?;
    // exam.jsonl always covers every task, whatever --only selected.
    fair::write_exam(&s.tasks, &fair::load_tasks(&s.tasks, None)?)?;
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::List { repo, since } => {
            for c in candidates(&repo, &since)? {
                println!("{}", serde_json::to_string(&c)?);
            }
        }
        Cmd::Mine { repo, out, since, jobs, limit, go } => {
            let repo = repo.canonicalize().context("--repo")?;
            let mut cands = candidates(&repo, &since)?;
            if limit > 0 {
                cands.truncate(limit);
            }
            mine(&repo, &out, cands, jobs, &go.docker())?;
        }
        Cmd::Flake { stage, repo, runs, go } => {
            let (repo, docker) = (canonical(&repo)?, go.docker());
            fair::run_stage("flake", &load(&stage)?, stage.jobs, stage.force,
                            |t| fair::flake(&repo, t, &docker, runs))?;
        }
        Cmd::Api { stage } => fair::run_stage("api", &load(&stage)?, stage.jobs, stage.force, fair::api)?,
        Cmd::Describe { stage, repo, llm } => {
            let repo = repo.as_deref().map(canonical).transpose()?;
            let writer = llm.copilot(&llm.describe_model);
            fair::run_stage("describe", &load(&stage)?, stage.jobs, stage.force,
                            |t| fair::describe(repo.as_deref(), t, &writer))?;
        }
        Cmd::Probe { stage, repo, llm } => {
            anyhow::ensure!(llm.describe_model != llm.probe_model,
                            "the reviewer must be a different model from the writer");
            let (repo, reviewer) = (canonical(&repo)?, llm.copilot(&llm.probe_model));
            fair::run_stage("probe", &load(&stage)?, stage.jobs, stage.force, |t| fair::probe(&repo, t, &reviewer))?;
        }
        Cmd::Specificity { stage, repo } => {
            let repo = repo.as_deref().map(canonical).transpose()?;
            fair::run_stage("specificity", &load(&stage)?, stage.jobs, stage.force,
                            |t| fair::specificity(repo.as_deref(), t))?;
        }
        Cmd::Miners => {
            let all: Vec<_> = rrsi_mine::miners::registry().iter().map(|m| miner::describe(*m)).collect();
            println!("{}", serde_json::to_string_pretty(&all)?);
        }
        Cmd::Miner(argv) => {
            let (name, rest) = argv.split_first().context("miner name")?;
            let summary = miner::run(name, miner::args_from_cli(rest)?)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Cmd::Gate { stage } => gate(&stage, &load(&stage)?)?,
        Cmd::Fairness { stage, repo, runs, go, llm } => {
            anyhow::ensure!(llm.describe_model != llm.probe_model,
                            "the reviewer must be a different model from the writer");
            let (repo, docker) = (canonical(&repo)?, go.docker());
            let (writer, reviewer) = (llm.copilot(&llm.describe_model), llm.copilot(&llm.probe_model));
            let tasks = load(&stage)?;
            let (j, f) = (stage.jobs, stage.force);
            fair::run_stage("flake", &tasks, j, f, |t| fair::flake(&repo, t, &docker, runs))?;
            fair::run_stage("api", &tasks, j, f, fair::api)?;
            fair::run_stage("describe", &tasks, j, f, |t| fair::describe(Some(&repo), t, &writer))?;
            fair::run_stage("probe", &tasks, j, f, |t| fair::probe(&repo, t, &reviewer))?;
            fair::run_stage("specificity", &tasks, j, f, |t| fair::specificity(Some(&repo), t))?;
            gate(&stage, &tasks)?;
        }
    }
    Ok(())
}
