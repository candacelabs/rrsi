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
use rrsi_mine::toolchain::{bazel::Bazel, cpp::Cpp, go::Go, python::Python, Sandbox, Toolchains};
use rrsi_mine::mine;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(version, about = "Mine FAIL_TO_PASS-validated tasks (Go, Python, C++, Bazel) from git history")]
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
        #[command(flatten)]
        tc: ToolchainArg,
        #[command(flatten)]
        go: GoArgs,
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
        tc: ToolchainArg,
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
struct ToolchainArg {
    /// Which toolchain finds and validates candidates: go (the default),
    /// python (pytest), cpp (CMake + CTest), bazel (`bazel test`), or auto
    /// (every toolchain with a project root at HEAD; a commit goes to the
    /// first that claims it, in that order).
    #[arg(long, default_value = "go", value_parser = ["auto", "go", "python", "cpp", "bazel"])]
    toolchain: String,
}

/// Container settings of every toolchain. The Go flags are unchanged; the
/// others are pinned by digest and capped like the Go run.
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
    #[arg(long, env = "RRSI_PYTHON_IMAGE", default_value = PYTHON_IMAGE)]
    python_image: String,
    /// Named volume of the Python virtualenvs and uv's caches.
    #[arg(long, env = "RRSI_PYTHON_DEPS", default_value = "rrsi-pydeps")]
    python_deps: String,
    #[arg(long, env = "RRSI_CPP_IMAGE", default_value = CPP_IMAGE)]
    cpp_image: String,
    /// Extra cmake configure arguments (repeatable), e.g. -DFOO_TESTS=ON.
    #[arg(long = "cmake-arg", allow_hyphen_values = true)]
    cmake_args: Vec<String>,
    #[arg(long, env = "RRSI_BAZEL_IMAGE", default_value = BAZEL_IMAGE)]
    bazel_image: String,
    /// Named volume of Bazel's output base, repository and disk caches.
    #[arg(long, env = "RRSI_BAZEL_CACHE", default_value = "rrsi-bazelcache")]
    bazel_cache: String,
    #[arg(long, env = "RRSI_CPUS", default_value = "4")]
    cpus: String,
    #[arg(long, env = "RRSI_MEMORY", default_value = "6g")]
    memory: String,
    /// Seconds the networked prefetch of a non-Go toolchain may take.
    #[arg(long, env = "RRSI_PREFETCH_TIMEOUT", default_value_t = 3600)]
    prefetch_timeout: u64,
}

const PYTHON_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:85d4cb1afa769a7338e095b927bee941cf5ec92266c7424b3f6c0f2748567248";
const CPP_IMAGE: &str = "mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04@sha256:d51703c4fcbe93cd889d38005847521d87cca4d304f33423430daf10a384a332";
const BAZEL_IMAGE: &str = "gcr.io/bazel-public/bazel:9.2.0@sha256:e59bd66f8daf69f02dbfc18dbd72f0ecfe7926bbda95a5c9eb62433d83b8bd02";

impl GoArgs {
    fn sandbox(&self, image: &str) -> Sandbox {
        Sandbox { image: image.to_string(), cpus: self.cpus.clone(), memory: self.memory.clone(),
                  timeout: self.test_timeout, prefetch_timeout: self.prefetch_timeout }
    }

    fn toolchains(&self) -> Toolchains {
        Toolchains { all: vec![
            Box::new(Go { image: self.image.clone(), modcache: self.modcache.clone(),
                          buildcache: self.buildcache.clone(), test_timeout: self.test_timeout }),
            Box::new(Python { sandbox: self.sandbox(&self.python_image), deps: self.python_deps.clone() }),
            Box::new(Cpp { sandbox: self.sandbox(&self.cpp_image), cmake_args: self.cmake_args.clone() }),
            Box::new(Bazel { sandbox: self.sandbox(&self.bazel_image), cache: self.bazel_cache.clone(),
                             lock: Default::default() }),
        ] }
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
        Cmd::List { repo, since, tc, go } => {
            let tcs = go.toolchains();
            for c in tcs.candidates(&repo, &since, &tc.toolchain)? {
                println!("{}", serde_json::to_string(&c)?);
            }
        }
        Cmd::Mine { repo, out, since, jobs, limit, tc, go } => {
            let repo = repo.canonicalize().context("--repo")?;
            let tcs = go.toolchains();
            let mut cands = tcs.candidates(&repo, &since, &tc.toolchain)?;
            if limit > 0 {
                cands.truncate(limit);
            }
            mine(&repo, &out, cands, jobs, &tcs)?;
        }
        Cmd::Flake { stage, repo, runs, go } => {
            let (repo, tcs) = (canonical(&repo)?, go.toolchains());
            fair::run_stage("flake", &load(&stage)?, stage.jobs, stage.force,
                            |t| fair::flake(&repo, t, &tcs, runs))?;
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
        Cmd::Gate { stage } => gate(&stage, &load(&stage)?)?,
        Cmd::Fairness { stage, repo, runs, go, llm } => {
            anyhow::ensure!(llm.describe_model != llm.probe_model,
                            "the reviewer must be a different model from the writer");
            let (repo, tcs) = (canonical(&repo)?, go.toolchains());
            let (writer, reviewer) = (llm.copilot(&llm.describe_model), llm.copilot(&llm.probe_model));
            let tasks = load(&stage)?;
            let (j, f) = (stage.jobs, stage.force);
            fair::run_stage("flake", &tasks, j, f, |t| fair::flake(&repo, t, &tcs, runs))?;
            fair::run_stage("api", &tasks, j, f, fair::api)?;
            fair::run_stage("describe", &tasks, j, f, |t| fair::describe(Some(&repo), t, &writer))?;
            fair::run_stage("probe", &tasks, j, f, |t| fair::probe(&repo, t, &reviewer))?;
            fair::run_stage("specificity", &tasks, j, f, |t| fair::specificity(Some(&repo), t))?;
            gate(&stage, &tasks)?;
        }
    }
    Ok(())
}
