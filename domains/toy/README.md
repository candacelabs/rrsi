# Toy instance: short Python function tasks

A small, cheap domain for prototyping the RRSI loop end to end on local or
low-cost models. It is **not** a benchmark and its numbers are not comparable
to the paper's.

* Evolve set: 20 Python function tasks (`data/tasks.py`), `k = 2` trials each.
  Held out: 10 more tasks (`python3 rrsi.py --domain toy heldout --label champ`).
* Starting harness H_0 (`harness/`): a deliberately minimal JSON-action loop
  with one sandboxed `run_python` tool. It does not strip markdown fences, never
  asks the model to test, and submits nothing when the step budget runs out.
* Score: the fraction of trials whose hidden unit tests pass.
* `data/reference.py` proves every hidden test passable; `tests/test_toy_domain.py`
  runs it.

## Sandbox

Policy-written code and the grader run in a fresh `python:3.12-slim` container
per program: no network, 512 MB memory, one CPU, 64 processes, read-only root,
unprivileged user (`sandbox.py`). `RRSI_TOY_SANDBOX=local` runs a plain
subprocess and is for tests only.

## Models

| Role | Variable |
|---|---|
| Proposer, analyst, critic | `RRSI_LLM_BACKEND` = `anthropic` or `openai` (see `rrsi/llm.py`) |
| Frozen policy | `RRSI_POLICY_BACKEND` = `openai` or `anthropic`, `RRSI_POLICY_MODEL` |

Local, one OpenAI-compatible server for every role (for example vLLM):

```bash
export RRSI_LLM_BACKEND=openai RRSI_POLICY_BACKEND=openai
export RRSI_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export RRSI_OPENAI_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 RRSI_POLICY_MODEL=$RRSI_OPENAI_MODEL
export RRSI_OPENAI_MAX_PROMPT_CHARS=80000   # keep prompts inside a 32K context
```

Anthropic API (Sonnet searches, Haiku is the policy):

```bash
export ANTHROPIC_API_KEY=...
export RRSI_LLM_BACKEND=anthropic RRSI_POLICY_BACKEND=anthropic
export RRSI_POLICY_MODEL=claude-haiku-4-5-20251001
```

## Run

```bash
python3 rrsi.py --domain toy smoke
python3 rrsi.py --domain toy baseline
python3 rrsi.py --domain toy calibrate
python3 rrsi.py --domain toy run
python3 rrsi.py --domain toy heldout --label champ
python3 rrsi.py --domain toy heldout --label base --ref <H_0 commit>
```

Keep one run directory per backend with `--runs`, because `evolve/toy` is
shared: run the second backend from a separate clone or reset `evolve/toy`
between runs.
