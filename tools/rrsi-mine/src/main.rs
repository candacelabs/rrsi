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
use rrsi_mine::csf::{self, MineCsf};
use rrsi_mine::{candidates, miner, mine, scan, Docker};
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
        /// Print the commits that are NOT candidates, with the reason, instead.
        #[arg(long)]
        rejected: bool,
        #[command(flatten)]
        csf: CsfArgs,
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
        #[command(flatten)]
        csf: CsfArgs,
    },
    /// CSF integration: detection, guards and component annotation (CSF.md).
    Csf {
        #[command(subcommand)]
        cmd: CsfCmd,
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

#[derive(Subcommand)]
enum CsfCmd {
    /// Is the repository CSF-instrumented at REV, and why. Exits 0 either way.
    Detect {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value = "HEAD")]
        rev: String,
        /// Print the full detection as JSON.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        gate: GateArgs,
    },
    /// Run CSF's gates on a checkout and print the verdicts as JSON.
    Guard {
        #[arg(long)]
        tree: PathBuf,
        #[command(flatten)]
        gate: GateArgs,
    },
    /// Add the `csf` component map to every task.json under --tasks.
    Annotate {
        #[arg(long)]
        tasks: PathBuf,
        #[arg(long)]
        repo: PathBuf,
        /// Also run CSF's gates on each task's commit tree into `csf_guards`
        /// (exports every commit; the same guards `mine` records).
        #[arg(long)]
        guards: bool,
        #[command(flatten)]
        csf: CsfArgs,
    },
}

/// Finding and feeding CSF's compiler.
#[derive(Args, Clone)]
struct GateArgs {
    /// csfc binary (else RRSI_CSFC, else `csfc` on PATH).
    #[arg(long)]
    csfc: Option<PathBuf>,
    /// CSF grammar csfc reads (else RRSI_CSF_GRAMMAR, else the tree's own
    /// csf/compiler/architecture/language.ebnf).
    #[arg(long)]
    csf_grammar: Option<PathBuf>,
    /// An architecture source outside <root>/csf/architecture/architecture.csf
    /// (repository-relative; repeatable).
    #[arg(long)]
    csf_source: Vec<String>,
}

/// Where the CSF model for mining comes from.
#[derive(Args, Clone)]
struct CsfArgs {
    #[command(flatten)]
    gate: GateArgs,
    /// A `csfc emit --format json` document to map tasks onto, instead of
    /// running csfc.
    #[arg(long, requires = "csf_root")]
    csf_model: Option<PathBuf>,
    /// Repository-relative directory the --csf-model paths are relative to.
    #[arg(long)]
    csf_root: Option<String>,
    /// Revision whose architecture csfc reads when there is no --csf-model.
    #[arg(long, default_value = "HEAD")]
    csf_model_rev: String,
}

impl CsfArgs {
    fn resolve(&self, repo: &Path) -> Result<MineCsf> {
        let file = self.csf_model.as_deref().map(|f| (f, self.csf_root.as_deref().unwrap_or("")));
        MineCsf::resolve(repo, &self.csf_model_rev, self.gate.csfc.as_deref(), self.gate.csf_grammar.as_deref(),
                         &self.gate.csf_source, file)
    }
}

fn detect(repo: &Path, rev: &str, gate: &GateArgs) -> Result<csf::detect::Detection> {
    let src = csf::source::Source::git(repo, rev)?;
    let (mut d, files) = csf::detect::detect(&src, &gate.csf_source)?;
    let info = csf::csfc::locate(gate.csfc.as_deref());
    if info.available && d.csf_files.iter().any(|f| f.status != "other") {
        let tree = csf::materialize(repo, rev)?;
        csf::detect::with_csfc(&mut d, &files, tree.path(), info, gate.csf_grammar.as_deref())?;
    } else {
        d.csfc = Some(info);
    }
    Ok(d)
}

fn print_detection(d: &csf::detect::Detection) {
    println!("instrumented: {}", if d.instrumented { "yes" } else { "no" });
    for s in &d.signals {
        println!("  signal {:<20} {} {}", s.signal, s.path, s.detail);
    }
    for m in &d.go_modules {
        println!("  go module {:?} {} {}", m.root, m.module.as_deref().unwrap_or("?"),
                 if m.requires_csf { "(requires csf)" } else { "" });
    }
    for f in &d.csf_files {
        println!("  csf file {} [{}] {}", f.path, f.status, f.detail);
        for g in f.diagnostics.iter().take(5) {
            println!("    {}:{}:{}: {}: {}", g.file, g.line, g.col, g.code, g.message);
        }
    }
    for m in &d.models {
        println!("  model {} (root {:?}): {} components, {} generated roots", m.model.architecture.name, m.root,
                 m.model.components.len(), m.model.generated_roots.len());
    }
    if let Some(c) = &d.csfc {
        println!("  csfc: {} ({}; {})", c.path.as_deref().unwrap_or("none"),
                 if c.available { "available" } else { "unavailable" }, c.detail);
    }
}

/// Write `csf` (and with `guards`, `csf_guards`) into each task.json under
/// `tasks`, keeping every other field.
fn annotate(tasks: &Path, repo: &Path, csf: &MineCsf, guards: bool) -> Result<()> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(tasks)?.flatten().map(|e| e.path())
        .filter(|p| p.join("task.json").is_file()).collect();
    dirs.sort();
    let (mut n, mut mapped) = (0, 0);
    for d in dirs {
        let path = d.join("task.json");
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        let files: Vec<String> = v.get("src_files").and_then(|f| f.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let t = csf.task(&files);
        mapped += t.as_ref().is_some_and(|t| !t.components.is_empty()) as usize;
        n += 1;
        match t {
            Some(t) => v["csf"] = serde_json::to_value(t)?,
            None => { v.as_object_mut().map(|o| o.remove("csf")); }
        }
        if guards {
            if let Some(sha) = v.get("sha").and_then(|s| s.as_str()).map(str::to_string) {
                let tree = csf::materialize(repo, &sha)?;
                match csf.guards(tree.path())? {
                    Some(g) => v["csf_guards"] = serde_json::to_value(g)?,
                    None => { v.as_object_mut().map(|o| o.remove("csf_guards")); }
                }
            }
        }
        std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    }
    println!("[csf] annotated {n} task(s); {mapped} map to at least one CSF component");
    Ok(())
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
        Cmd::List { repo, since, rejected, csf } => {
            let repo = canonical(&repo)?;
            let (cands, rejections) = scan(&repo, &since, &csf.resolve(&repo)?)?;
            if rejected {
                for r in rejections {
                    println!("{}", serde_json::to_string(&r)?);
                }
            } else {
                for c in cands {
                    println!("{}", serde_json::to_string(&c)?);
                }
            }
        }
        Cmd::Mine { repo, out, since, jobs, limit, go, csf } => {
            let repo = canonical(&repo)?;
            let csf = csf.resolve(&repo)?;
            println!("[mine] {}", csf.describe());
            let (mut cands, rejections) = scan(&repo, &since, &csf)?;
            if limit > 0 {
                cands.truncate(limit);
            }
            mine(&repo, &out, cands, jobs, &go.docker(), &csf)?;
            let mut f = std::fs::File::create(out.join("rejected.jsonl"))?;
            for r in &rejections {
                use std::io::Write;
                writeln!(f, "{}", serde_json::to_string(r)?)?;
            }
            println!("[mine] {} commit(s) rejected before validation (rejected.jsonl)", rejections.len());
        }
        Cmd::Csf { cmd } => match cmd {
            CsfCmd::Detect { repo, rev, json, gate } => {
                let d = detect(&canonical(&repo)?, &rev, &gate)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&d)?);
                } else {
                    print_detection(&d);
                }
            }
            CsfCmd::Guard { tree, gate } => {
                let v = csf::guard::guard(&tree, gate.csfc.as_deref(), gate.csf_grammar.as_deref(), &gate.csf_source)?;
                println!("{}", serde_json::to_string_pretty(&v)?);
            }
            CsfCmd::Annotate { tasks, repo, guards, csf } => {
                let repo = canonical(&repo)?;
                let csf = csf.resolve(&repo)?;
                println!("[csf] {}", csf.describe());
                annotate(&tasks, &repo, &csf, guards)?;
            }
        },
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
