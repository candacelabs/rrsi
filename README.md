# RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

> ## Candace Labs fork — prototype status
>
> **Before:** to learn whether a change to an AI coding agent actually helps
> *on your own codebase*, you either trusted public benchmarks built from other
> people's code, or wrote test tasks by hand and checked each one yourself.
>
> **Now:** one command turns your repository's git history into a verified
> exam — 86 recent commits in about 11 minutes on one machine, 70 of them proven
> questions (the tests fail before the real fix and pass after it, offline),
> 25 of which pass every fairness check — and it warns you when the final exam
> is not a fair test.
>
> **Things you can now try that were not possible before:**
>
> - **Hand an agent a real past bug from your own repo, as a proven-fair
>   question.** Each of the 70 tasks gives the repo at the commit before the
>   fix and hidden tests that are known to fail without it and pass with it;
>   the fairness stages add a written instruction that never shows the fix
>   (written for 66 of 70; 4 failed the leak check).
> - **Measure how well your work is specified.** For each change, the fairness
>   stages ask whether what was written down (commit message + tests) lets
>   someone else rebuild it without guessing. Of 70 recent changes with valid
>   tests: 25 (36%) were fully specified; 39 (56%) left a reader guessing
>   behaviour or names (e.g. a "bounded grace" with no duration); 18 (26%) had
>   tests pinning details no written spec states. Today this scores the recorded
>   intent, a proxy for prompt quality; pointing it at the original prompts
>   scores them directly (not wired yet).
> - **Find tests that do not guard the change they shipped with.** 14 of 86
>   commits added or changed tests that already passed *before* the change.
>   For a pure refactor that is expected; for a bug fix it means the new test
>   would not catch the bug coming back. The miner lists them.
> - **See which parts of your codebase an exam covers**, per subsystem, per
>   code layout and per fix size, before trusting any score on it.
> - **Know in advance whether a final-exam result would mean anything.** The
>   split-health checks flag a final exam that is harder, narrower or from a
>   different part of the code than practice — here, 4 of 6 checks warned on
>   the first split and 2 of 6 after the fairness stages.
> - **Grow the exam automatically.** Re-running the miner only processes new
>   commits, so every merged fix with a test becomes a new candidate question.
> - **Mine any CSF-instrumented Go repository with CSF's own architecture and
>   gates.** (CSF is Candace Labs' Go framework for AI-agent systems, not yet
>   publicly released; a repository is *CSF-instrumented* when it uses the Go
>   module `github.com/candacelabs/csf` or declares its architecture in an
>   `architecture.csf` file. See the [definition](tools/rrsi-mine/CSF.md#what-csf-is).)
>   `rrsi-mine csf detect` says whether a repo uses CSF and why; each
>   task records which declared components its fix touches (4 of our 70) and
>   whether the fix passes `csfc check` / `check-generated`, so only gates the
>   real fix passes are ever required of an agent. See
>   [tools/rrsi-mine/CSF.md](tools/rrsi-mine/CSF.md).
> - **Do it on any Go repository** (`--repo PATH`), with no API key: the
>   search roles and the agent can run on a logged-in Copilot or Codex CLI, or
>   a local model.
> - *One step away (grader and harness not wired yet):* compare two agent
>   setups — two models, two prompts, two tool sets — by pass rate on your own
>   code instead of on public benchmarks, and let RRSI evolve the harness
>   against the practice set while the final exam checks for memorizing.
>
> Not shown yet: whether RRSI then improves an agent on that exam. That needs
> the baseline and RRSI rounds, and will replace this headline.
>
> | What we measured | Result |
> |---|---|
> | Recent commits in our Go monorepo that change code + its tests | 86 candidates |
> | ...that make a valid question (tests **fail** before the real fix, **pass** after, offline, pinned Go) | **70 of 86**; 14 dropped because the tests already passed before the fix, 2 because the commit itself did not build |
> | Automatic check "is the final exam comparable to the practice set?" on the naive split (newest 10 held out) | **4 of 6 checks warn**: final-exam fixes 1.6x larger, 100% in one code layout vs 23%, half one subsystem, all from ~2 hours of work. A score on that split would not distinguish learning from memorizing. |
> | Fairness stages on all 70 valid tasks (flaky tests, required API, instruction writing + leak check, independent solvability probe, over-specific tests) | **25 of 70 exam-ready**. Excluded: 25 by the probe alone (instruction ambiguous or names must be guessed), 10 by probe + over-specific tests, 6 by over-specific tests alone, 4 by the leak check. 0 flaky. |
> | Exam after fairness (newest 10 exam-ready held out) | 15 practice + 10 final exam; split health 2 of 6 warn (final exam 0% vs practice 87% in the older `go/` layout). **15 practice tasks is too few for RRSI rounds** — next: mine further back and re-check the probe's strictness. |
> | Toy domain end to end on a local 4B model (vLLM) | smoke 2/2 passed; loop, critic, sandbox and grader all run |
> | Bugs the pipeline's own checks found and now pin with regression tests | download failures misread as failing tests; a new in-repo package misread as a download failure |
>
> **What this fork adds** (upstream RRSI below is unchanged):
>
> | Piece | What it does | Where |
> |---|---|---|
> | LLM backends | Run the search roles without Vertex: Anthropic API, any OpenAI-compatible server (vLLM), or a logged-in **Copilot** / **Codex** CLI | [`rrsi/llm.py`](rrsi/llm.py), [`rrsi/cli_llm.py`](rrsi/cli_llm.py) |
> | `toy` domain | 30 small Python tasks with hidden tests, a deliberately weak harness, a network-less container sandbox — the cheapest full RRSI loop | [`domains/toy/`](domains/toy/README.md) |
> | `rrsi-mine` | Rust: mine commits → validate FAIL_TO_PASS in sealed `golang` containers → fairness stages → `exam.jsonl`; pyo3 bindings (`import rrsi_mine`) for the Python side | [`tools/rrsi-mine/`](tools/rrsi-mine) |
> | `rrsi-report` | Rust: one self-contained HTML report — plain-language "start here", funnel, why tasks were dropped, timeline, practice-set vs final-exam split with health checks and interpretation, task browser with example cards (the reference fix stays a collapsed spoiler) | [`tools/rrsi-mine/src/bin/rrsi-report.rs`](tools/rrsi-mine/src/bin/rrsi-report.rs) |
> | Harness miner | Rust `rrsi-mine traces` (a plugin of the miner registry, `rrsi-mine miners`) finds where the agent struggled in Claude Code session transcripts; `python -m rrsi harness mine` clusters recurring struggles into harness tasks and a private report | [`tools/rrsi-mine/src/miners/traces.rs`](tools/rrsi-mine/src/miners/traces.rs), [`rrsi/harness/`](rrsi/harness/mine.py) |
>
> **How it works, in one example.** A commit "suppress unsafe notification
> retries" added a test: *a delivery error that says it is not retryable must be
> attempted once*. On the commit before, that test fails; with the real 24-line
> fix it passes — so it is a fair question. The agent gets a written
> description of the required behaviour and API (never the fix), and is graded
> by those hidden tests. RRSI then rewrites the agent's *harness* (prompts,
> tools, loop — not the model) and keeps only changes that beat measured noise;
> the newest tasks are held back as a final exam the process never sees.
>
> **Counterfactuals — why each check exists:** keep a task whose tests already
> passed → an agent that does nothing "passes"; count a download failure as a
> failing test → questions look valid for the wrong reason (this happened, and
> is now a pinned regression test); let instructions quote the fix → the agent
> copies it; hide a required function name → nobody can pass without guessing;
> make the final exam harder or narrower than practice → a lower score looks
> like overfitting when it is really a harder exam.
>
> **Run it** (mined tasks contain private source: `--out` must be outside every
> git work tree, which the tool enforces):
>
> ```bash
> cargo build --release --manifest-path tools/rrsi-mine/Cargo.toml
> tools/rrsi-mine/target/release/rrsi-mine mine --repo PATH --out DIR --jobs 8
> tools/rrsi-mine/target/release/rrsi-mine fairness --tasks DIR --jobs 4
> tools/rrsi-mine/target/release/rrsi-report --tasks DIR --repo PATH --out DIR/report.html --splits-out DIR/splits.json
> ```
>
> **Next:** fairness on all 70 tasks, a balanced split (newest ~2 weeks,
> spread across subsystems), then the `house_go` domain: baseline → noise band
> → RRSI rounds → final exam. The headline above will be replaced by that
> result.
>
> **Harness miner: learn from the sessions themselves.** Where an exam asks
> "can the agent fix this bug?", the harness miner asks "where did the agent
> struggle while we actually worked with it, and what harness change would have
> prevented it?" It reads Claude Code transcripts (`~/.claude/projects`,
> including subagent transcripts) in two stages:
>
> 1. `rrsi-mine traces` (Rust, no LLM) parses every transcript and runs nine
>    named, unit-tested detectors: `tool_error`, `retry` (the same call again
>    after it failed), `hook_timeout`, `permission_denial`, `user_interrupt`,
>    `user_correction` (a short pushback such as "no", "why", "again" or
>    shouting right after an agent action), `reask` (the user asks nearly the
>    same thing again after an answer), `silence` (the harness told the agent
>    the user has not heard from it) and `test_failure` (go test, cargo,
>    pytest, bazel, GitHub Actions). Hits close together become one *episode*
>    with a bounded, truncated context window. Unchanged transcripts are
>    skipped by mtime + content hash, so re-runs take seconds.
> 2. `python -m rrsi harness mine` runs stage 1, then has a model (default:
>    `claude-opus-5-5` through the Claude Agent SDK on the logged-in Claude
>    Code, no API key; `--backend copilot|codex` use those logged-in CLIs)
>    label each episode with a recurring-struggle pattern, merge patterns into
>    clusters and write one task per top cluster: title, pattern, evidence
>    (episode ids and counts), root-cause hypothesis, proposed harness fix
>    (CLAUDE.md rule, skill, house-lint gate rule, memory, tool/CLI fix or
>    doc), acceptance check and priority, plus RRSI-style exam candidates where
>    a before/after is mechanically checkable. Every count is computed from
>    the episode records, never by the model.
>
> ```bash
> python -m rrsi harness mine                  # ~/.claude/projects -> ~/rrsi-private/harness
> python -m rrsi harness mine --since 2026-09-01 --backend copilot
> tools/rrsi-mine/target/release/rrsi-mine traces --out DIR   # stage 1 only
> ```
>
> Output: `DIR/episodes.jsonl`, `DIR/tasks/<id>.json` + `index.json`,
> `DIR/exam_candidates.jsonl`, `DIR/REPORT.md` (the top recurring struggles
> with episode, session and project counts and the proposed fix).
>
> **Handoffs: when should agents have talked to each other?** A second
> transcript miner, `handoffs`, looks at several concurrent sessions at once.
> In one transcript it finds the operator relaying between sessions ("tell the
> other session..."), messages that arrived from other sessions or a
> coordinator (scored: did the receiver act, did it reply?), retractions and
> rename churn in those messages, outgoing message calls, merge/rebase
> conflicts, writes refused because a file belongs to another worktree,
> "already done by..." discoveries, waiting on another session's work,
> re-running a status check instead of asking, ownership questions and
> claim/release comments. Across transcripts it finds two sessions editing
> the same repository file (worktrees folded together) or branch at the same
> time, and near-identical issue or PR titles from two sessions.
> `python -m rrsi harness mine --miner handoffs` turns the recurring patterns
> into tasks of the same shape, each with a trigger rule: "when
> <detectable condition>, message <owner, and how the owner is resolved> with
> <payload>", ready to become a harness rule or an ownership-state hook.
>
> ```bash
> python -m rrsi harness mine --miner handoffs   # -> ~/rrsi-private/harness/handoffs
> tools/rrsi-mine/target/release/rrsi-mine handoffs --out DIR   # deterministic stage only
> ```
>
> **PR gap: do actively working agents have a PR?** A third transcript
> miner, `pr-gap`, measures each agent run (a main session or one subagent):
> the time from its first `git commit` to its first `git push`, how many
> commits sat unpushed, the time from the first push to `gh pr create`, and
> whether the run ended with commits but no PR (the gap). Commit messages and
> heredocs are not mistaken for commands. It also flags briefs (`Agent`
> prompts, or brief files a subagent reads) that defer the PR ("don't open a
> separate PR", "when complete, open...") and later operator corrections, and
> splits subagent gap rates by brief class (defers / asks for an early draft
> PR / silent). With `--github` it joins pushed branches against
> `gh pr list --head` to confirm each gap and to find branches that already
> had a PR. It scores the rule family "when an agent's first commit is N
> minutes old with no push, or its pushed branch has had no PR for N minutes,
> open a draft PR for it and tell the agent" on the measured runs and writes
> the best one as a harness task.
>
> ```bash
> python -m rrsi harness mine --miner pr-gap   # -> ~/rrsi-private/harness/pr-gap (joins GitHub)
> tools/rrsi-mine/target/release/rrsi-mine pr-gap --out DIR [--github]   # deterministic stage only
> ```
>
> **Privacy rules.** Transcripts hold private source, hostnames, addresses and
> personal text. Both stages refuse to write inside any git work tree; keep
> the output (default `~/rrsi-private/harness`) out of every repository. Text
> sent to a model is redacted first (e-mail and IP addresses, token-like
> strings, long hex, home paths) and truncated; the SDK backend runs with no
> tools, no settings and no session persistence, so the miner's own calls
> never become transcripts it mines. This repository holds only the code and
> synthetic test fixtures: no transcript text, episode or finding is ever
> committed here.
>
> **Miners are plugins; write your own.** `rrsi-mine` runs any registered
> miner: `rrsi-mine miners` lists them (name, inputs, the records each
> writes) and `rrsi-mine <name> --key value ...` runs one. Five ship today:
> `git-history` (the FAIL_TO_PASS task miner above), `slices` (FAIL_TO_PASS
> tasks from merged slice PRs), `traces` (the struggle miner), `handoffs`
> (cross-session coordination) and `pr-gap` (active agents without a PR). A miner is one file in
> [`tools/rrsi-mine/src/miners/`](tools/rrsi-mine/src/miners/mod.rs)
> implementing the [`Miner`](tools/rrsi-mine/src/miner.rs) trait plus one
> registration line; deleting both removes it. Arguments arrive as a plain
> JSON object (the CLI turns `--key value` into `{"key": value}`) and the
> summary goes back as JSON, so any front end can drive a miner without
> linking against it. The privacy rule is enforced for every miner before
> it runs: an `out` inside a git work tree is refused.
>
> ```rust
> // tools/rrsi-mine/src/miners/todo_comments.rs
> use crate::miner::{parse_args, Miner};
> use serde_json::{json, Value};
>
> pub struct TodoComments;
>
> #[derive(serde::Deserialize)]
> #[serde(deny_unknown_fields)]
> struct Args { repo: std::path::PathBuf, out: std::path::PathBuf }
>
> impl Miner for TodoComments {
>     fn name(&self) -> &'static str { "todo-comments" }
>     fn about(&self) -> &'static str { "one record per TODO comment in a repository" }
>     fn inputs(&self) -> &'static [(&'static str, &'static str)] {
>         &[("repo", "the repository"), ("out", "output directory")]
>     }
>     fn records(&self) -> &'static [(&'static str, &'static str)] {
>         &[("todos.jsonl", "one TODO: file, line, text")]
>     }
>     fn run(&self, args: Value) -> anyhow::Result<Value> {
>         let a: Args = parse_args(self.name(), args)?;
>         let grep = crate::git(&a.repo, &["grep", "-n", "TODO"]).unwrap_or_default();
>         std::fs::create_dir_all(&a.out)?;
>         let mut n = 0;
>         let mut text = String::new();
>         for line in grep.lines() {
>             let mut p = line.splitn(3, ':');
>             let (file, no, body) = (p.next(), p.next(), p.next());
>             text += &json!({"file": file, "line": no, "text": body}).to_string();
>             text.push('\n');
>             n += 1;
>         }
>         std::fs::write(a.out.join("todos.jsonl"), text)?;
>         Ok(json!({"todos": n}))
>     }
> }
> ```
>
> Then add `todo_comments => TodoComments,` to the `register!` list in
> `src/miners/mod.rs` and run `rrsi-mine todo-comments --repo PATH --out DIR`.

Check out our [paper](https://arxiv.org/abs/2609.24972) and [project page](https://regularized-rsi.com/) for more details.

## 🔥 Updates

- [09/21/2026] Our [paper](https://arxiv.org/abs/2609.24972) is out! Check it out! [[project page]](https://regularized-rsi.com/)

## 🧬 Overview

<p align="center">
  <img src="./assets/rrsi_overview.png" width="92%" alt="RRSI overview: proposal-side and selection-side regularization of the harness search">
</p>

An LLM agent's capability is largely set by its harness: the prompts, control
flow, tools, memory and context management around a frozen model. Evolving the
harness against a fixed evolve set is effective but overfits: the harness
memorizes the training tasks, and large in-distribution gains shrink or vanish
out of distribution. RRSI keeps the harness edit space open and regularizes the
search trajectory through it instead.

On the proposal side, an annealed budget caps how many independent edits one
candidate may bundle, the proposer is conditioned on the full edit history so a
falsified hypothesis is not redrawn, and a stalled run is redirected toward
components it has never exercised. On the selection side, a critic screens
every candidate for suite-specific logic before it is evaluated, a
noise-adjusted floor blocks gains within evaluation variance, a cost rule
requires added inference tokens to be paid for by measured gain, and
components that stop helping are pruned.

### ✨ Key features

* **Open edit space, regularized search.** Prompts, control flow, configuration, context management, tools, skills, memory and sub-agents may all be modified; the constraints act on how the search moves, not on what the harness may contain.
* **One method, three instances.** The same loop drives a terminal agent (Terminal-Bench 2.1), a document-work agent (Harvey LAB) and an engineering-design agent (EngDesign); each instance is a `Domain` adapter plus its starting harness.
* **Candidates in git worktrees.** Every candidate harness is drafted, screened and evaluated in its own worktree on a branch off `evolve/<domain>`; accepting one fast-forwards the branch, so the incumbent is always a commit.
* **Evidence you can audit.** The edit history records, per edit, the component, the hypothesis, the measured score and cost change and the verdict; the prompts the proposer, analyst and critic receive are plain files in `domains/<name>/`.

## 🧩 Method to code

| Paper | Code |
|---|---|
| Empirical score and cost estimate | `rrsi/evaluate.py: aggregate` (weighted per-trial rewards; a missing trial counts 0 with the full denominator) |
| Annealed edit budget b_t | `rrsi/schedule.py: edit_budget`, enforced in the proposer's done() |
| Edit history L_t, tried set T_t, recent yield g_t | `rrsi/history.py: History` (one JSONL record per edit; a = 1 only for the edits of the candidate that became H_{t+1}) |
| Stall flag, untried components, exploration directives | `rrsi/history.py: stall_flag, exploration`; reserved slots enforced in `rrsi/propose.py` |
| Analyze(H_t, D) | `rrsi/analyst.py` dispatching `rrsi/digester.py` |
| Proposer with (component, hypothesis, diff) tags | `rrsi/propose.py`; tags validated against the diff by `rrsi/components.py` |
| Critic (leakage screen before evaluation) | `rrsi/critic.py` (domain regex denylist plus LLM review, bounded repair) |
| Evaluate in parallel | `rrsi/evaluate.py`, `Run.round` thread pool |
| Noise-adjusted floor, cost rule, within-band rule, argmax | `rrsi/selection.py` |
| Novelty nu_t (structural component types never in a winning edit) | `rrsi/components.py: novelty` over K_str = client_tool, skill, memory, subagent |
| Prune set B_t | `History.prune_set`, handed to the proposer with the accepted machinery to remove |
| Noise band delta | fixed per instance in `rrsi.json` (0.017 / 0.004 / 0.020); `rrsi/calibrate.py` re-estimates it when `delta` is `null` (bootstrap over trials of the base evaluation, or repeated base evaluations) |
| Non-compensatory domain criteria | `Domain.guards` (engineering: valid-rate drop, no-submission rise) |

---

## ⚡️ Quickstart

### 0. Install

```bash
git clone https://github.com/google-research/rrsi.git && cd rrsi
pip install -e ".[dev]"            # the search core (Python 3.10 or newer)
python3 -m pytest tests
```

The benchmark runners live in their own environments: harbor for the coding instance (`domains/coding/.venv`), and a Python 3.11 environment with `pip install -e ".[agentic]"` for the workspace and engineering instances (`RRSI_AGENT_PYTHON`).

### 1. LLM configuration

The proposer, the analyst, the critic and the frozen policy are Claude Opus 4.8 on Vertex AI (`policy_model` in `domains/coding/rrsi.json`, `ORCHESTRATOR_MODEL` for the other two instances; any LiteLLM model string works). The Harvey LAB judge is Gemini 3.5 Flash.

```bash
gcloud auth application-default login
export VERTEX_PROJECT="your-project-id" VERTEXAI_PROJECT="your-project-id"
export VERTEX_LOCATION=global VERTEXAI_LOCATION=global
export RRSI_VERTEX_PROJECTS="your-project-id"
```

### 2. Run an instance

Every instance follows the same shape:

```bash
python3 rrsi.py --domain <coding|workspace|eng> smoke     # liveness: compile, construct, a couple of tasks
python3 rrsi.py --domain <name> baseline                   # Evaluate(H_0), seed runs/<name>/frontier.json
python3 rrsi.py --domain <name> run                        # rounds 0..T-1, resumable; touch runs/<name>/STOP to stop
python3 rrsi.py --domain <name> status
```

Each round drafts two candidates in their own git worktrees, screens them, evaluates both on the full evolve set and fast-forwards `evolve/<name>` to the winner. `runs/<name>/` holds the frontier, the edit history and the raw trials. Hyperparameters live in `domains/<name>/rrsi.json` and can be overridden on the command line (`--T`, `--k`, `--delta`, `--beta1`, ...); `readjudicate --t <t>` re-applies Algorithm 2 to a stored round and `reevaluate --t <t>` re-measures one after an infrastructure failure.

Please refer to the specific document for the instance you want to run for its environment, its evaluation protocol and the out-of-distribution runs:

- [`domains/coding`](domains/coding/README.md): Terminal-Bench 2.1, then SWE-bench Verified
- [`domains/workspace`](domains/workspace/README.md): Harvey LAB, then JobBench, GDPval and APEX-Agents
- [`domains/eng`](domains/eng/README.md): EngDesign, then EngDesign v1 and Frontier-Eng

The short version of each:

```bash
# coding: Docker + harbor
python3 -m venv domains/coding/.venv && domains/coding/.venv/bin/pip install "harbor>=0.18"
python3 rrsi.py --domain coding baseline && python3 rrsi.py --domain coding run
bash domains/coding/scripts/swe_eval.sh                       # H_0 and the incumbent on SWE-bench Verified

# workspace: a Harvey LAB checkout at the pinned commit; the split is generated from it on first use
git clone https://github.com/harveyai/harvey-labs.git && (cd harvey-labs && git checkout 1da4750 && uv sync)
export HARVEY_LAB_ROOT=$PWD/harvey-labs RRSI_AGENT_PYTHON=~/venvs/rrsi-agentic/bin/python
python3 rrsi.py --domain workspace baseline && python3 rrsi.py --domain workspace run
python3 rrsi.py --domain workspace heldout --label champ      # the 40 held-out tasks; ood/run_{jobbench,gdpval,apex}.sh for the rest

# eng: the official EngDesign tasks in the verifier layout, a grading venv, a jailed tool gateway
git clone https://github.com/AGI4Engineering/EngDesign.git
python3 domains/eng/scripts/engdesign/build_engdesign_bench.py --engdesign-open EngDesign/EngDesign-Open --out domains/eng/engdesign_bench
python3 -m venv domains/eng/.venvs/engdesign && domains/eng/.venvs/engdesign/bin/pip install -r domains/eng/scripts/engdesign/requirements.txt
bash domains/eng/scripts/preflight.sh
python3 rrsi.py --domain eng baseline && python3 rrsi.py --domain eng run
bash domains/eng/scripts/final_eval.sh frontier               # Frontier-Eng, from a Frontier-Engineering checkout
```

## 📊 Results

Numbers from the paper, with Claude Opus 4.8 as the frozen policy in every instance and every number measured against the unevolved harness H_0 in the same window. "Evolve" is the split the harness was searched on; the other rows never entered selection. Terminal-Bench, SWE-bench, JobBench, GDPval, APEX-Agents and EngDesign report pass rate, Harvey LAB the fraction of rubric criteria passed and Frontier-Eng Medal points.

| Domain | Benchmark | Role | H_0 | RRSI | Δ |
|:---|:---|:---|:---:|:---:|:---:|
| Coding | Terminal-Bench 2.1 | evolve | 74.2 | **80.2** | +6.0 |
| Coding | SWE-bench Verified | OOD | 82.0 | **83.8** | +1.8 |
| Agentic workspace | Harvey LAB | evolve | 89.4 | **90.5** | +1.1 |
| Agentic workspace | Harvey LAB | ID held-out | 86.9 | **89.2** | +2.3 |
| Agentic workspace | JobBench | OOD | 36.0 | **40.7** | +4.7 |
| Agentic workspace | GDPval | OOD | 48.8 | **52.3** | +3.5 |
| Agentic workspace | APEX-Agents | OOD | 34.2 | **37.9** | +3.7 |
| Engineering design | EngDesign | evolve | 50.0 | **54.9** | +4.9 |
| Engineering design | Frontier-Eng | OOD | 17.7 | **22.0** | +4.3 |

The search is not tied to one policy family: with Gemini 3.5 Flash as the frozen policy, the same coding instance goes from 64.6 to 78.7 on Terminal-Bench 2.1 and from 76.8 to 79.0 on SWE-bench Verified.

## 🧱 Adding a domain

A domain is one module, `domains/<name>/adapter.py`, exporting `DOMAIN`, an
instance of `rrsi.domain.Domain` that implements:

* `evolve_ids`, `heldout_ids`, `smoke_ids`: the task splits;
* `run(root, runs_dir, job, ids, k)` and `score(runs_dir, job, ids, k)`: run the harness checked out under `root` and return per-task trial rewards (Evaluate);
* `load_trial`, `render_trace`, `task_row`: the evidence the analyst, digester and proposer read;
* `smoke`: a liveness check of a candidate before it is evaluated;
* `critic_patterns`, `component_signals`, `briefs`, `guards`: the domain's leakage denylist, diff-to-component signals, role prompts and non-compensatory acceptance criteria;

plus `harness_path` (the evolvable directory), `SKILL.md` and `PATTERNS.md`
(the proposer's constitution) and `rrsi.json` (hyperparameters). The core
never reads a trajectory format or a benchmark directory itself.

## 🧪 Tests

```bash
python3 -m pytest tests            # or: python3 tests/test_core.py
```

## 🙏 Acknowledgements

The starting harnesses are the Terminus-2 agent from [harbor](https://github.com/laude-institute/harbor) and the react_toolbelt agent and runner from [archipelago](https://github.com/Mercor-Intelligence/archipelago). The instances evaluate on [Terminal-Bench](https://github.com/harbor-framework/terminal-bench), [SWE-bench Verified](https://github.com/SWE-bench/SWE-bench), [Harvey LAB](https://github.com/harveyai/harvey-labs), [JobBench](https://github.com/Job-Bench/job-bench-eval), [GDPval](https://openai.com/index/gdpval/), [APEX-Agents](https://www.mercor.com/apex/apex-agents-leaderboard/), [EngDesign](https://github.com/AGI4Engineering/EngDesign) and [Frontier-Eng](https://github.com/Einsia/Frontier-Engineering).

## 💬 Citation

```bibtex
@article{xia2026rrsi,
  title={RRSI: Regularized Recursive Self-Improvement of Agent Harnesses},
  author={Xia, Peng and Han, Rujun and Wang, Zifeng and Chen, Yanfei and Zhuang, Yufan and Lee, Yoonho and Huang, Chengsong and Yu, Han and CuiZhu, Zhongying and Ming, Yifei and Yao, Huaxiu and Gokturk, Burak and Pfister, Tomas and Lee, Chen-Yu},
  journal={arXiv preprint arXiv:2609.24972},
  year={2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0; see [LICENSE](LICENSE). Third-party code under `third_party/` carries its own license.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
