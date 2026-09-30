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

//! `rrsi-report`: render a mined task exam as one self-contained HTML page
//! and decide its evolve / held-out split.
//!
//! ```text
//! rrsi-report --tasks DIR [--repo PATH] [--heldout 10] --out report.html [--splits-out splits.json]
//! ```
//!
//! The split rule lives here (report/split.rs), not in the page: the page and
//! `--splits-out` both show what this binary decided. The page embeds task
//! source, tests and logs, so neither output may be written inside a git work
//! tree.

#[path = "../report/mod.rs"]
mod report;

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rrsi-report", about = "Render a mined task exam as a self-contained HTML report")]
struct Cli {
    /// Directory of mined tasks (<sha12>/task.json ...).
    #[arg(long)]
    tasks: PathBuf,
    /// Repository the tasks were mined from; supplies commit dates. Without
    /// it there is no timeline and the held-out split falls back to sha order.
    #[arg(long)]
    repo: Option<PathBuf>,
    /// How many of the newest exam-ready tasks are held out.
    #[arg(long, default_value_t = 10)]
    heldout: usize,
    /// The HTML report to write.
    #[arg(long)]
    out: PathBuf,
    /// Also write the split as JSON {"evolve":[..],"heldout":[..],"excluded":{sha12: reason},
    /// "health":{..}} (health: see report/health.rs).
    #[arg(long)]
    splits_out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    for out in std::iter::once(&cli.out).chain(cli.splits_out.as_ref()) {
        if let Some(tree) = report::guard::enclosing_work_tree(out)? {
            bail!("refusing to write {}: it is inside the git work tree {} and the report \
                   holds private task source; write it outside any repository",
                  out.display(), tree.display());
        }
    }
    let mut tasks = report::load::load_tasks(&cli.tasks)
        .with_context(|| format!("loading tasks from {}", cli.tasks.display()))?;
    if tasks.is_empty() {
        bail!("no tasks under {}", cli.tasks.display());
    }
    match &cli.repo {
        Some(repo) => {
            let missing = report::load::attach_dates(&mut tasks, repo);
            if missing > 0 {
                eprintln!("rrsi-report: {missing} task(s) have no commit date in {}", repo.display());
            }
        }
        None => eprintln!("rrsi-report: no --repo, so no commit dates: the held-out split \
                           falls back to sha order and the timeline is skipped"),
    }
    let exam = report::load::exam_lines(&cli.tasks);
    let splits = report::split::assign(&mut tasks, cli.heldout);
    let health = report::health::health(&tasks);
    let html = report::render::render(&tasks, &splits, &health, cli.heldout, exam,
                                      &report::render::now_utc())?;
    std::fs::write(&cli.out, html).with_context(|| format!("writing {}", cli.out.display()))?;
    if let Some(p) = &cli.splits_out {
        let out = serde_json::json!({
            "evolve": splits.evolve, "heldout": splits.heldout, "excluded": splits.excluded,
            "health": health,
        });
        std::fs::write(p, serde_json::to_string_pretty(&out)? + "\n")
            .with_context(|| format!("writing {}", p.display()))?;
    }
    eprintln!("rrsi-report: {} tasks -> evolve {}, heldout {}, excluded {}; wrote {}",
              tasks.len(), splits.evolve.len(), splits.heldout.len(), splits.excluded.len(),
              cli.out.display());
    for c in &health.checks {
        eprintln!("  split health {:>4}: {}: {}",
                  serde_json::to_value(&c.status)?.as_str().unwrap_or("?"), c.name, c.detail);
    }
    Ok(())
}
