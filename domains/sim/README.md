# Simulation instance: driving controllers on CSF scenarios

RRSI evolves the harness of an agent that writes a **controller** for a car,
and the controller is judged in a **simulator**. Tasks are scenarios, not
commits. The code is plain Python; the scenario format, the controller checker
and the simulator workers come from CSF.

**CSF** ("Cerebrospinal Fluid") is Candace Labs' Go framework for AI-agent
systems: one Go runtime that hosts agent work, typed tools callable over HTTP
and MCP, shared knowledge and recorded experiment evidence, plus an
architecture compiler (`csfc`) that checks a repository's declared components
against its code. It is in developer preview and not yet publicly released.
This domain uses four of its pieces: the `Scenario` and `Controller` protobuf
messages, the bounded controller compiler and runtime (a Go binary that reads
JSON lines), the CPU driving harness (HighwayEnv), and the scenario worker that
runs a controller in CARLA 0.9.16 inside a container. `candace` is the
operator command-line tool of the Candace Labs monorepo that hosts CSF; this
domain calls only its `candace csf simulator run carla` verb, which starts that
container.

| Piece | What it is | Where |
|---|---|---|
| Tasks | 8 practice + 4 held-out straight-path scenarios: CSF `Scenario` JSON (seeded target speed 7-14 m/s, 12-20 s, start offset up to 1.2 m, heading up to 0.08 rad) plus a goal distance | `data/scenarios.py` |
| Agent | frozen policy LLM + the evolvable harness H_0 (a JSON-action loop with a CSF `check` tool; a `rollout` tool on a cheap surrogate plant is passed in but unused by H_0) | `harness/` |
| Check | CSF's bounded controller compiler (the Go runtime, reading JSON lines on stdin) | `bench/engines.py` |
| Execute | one CSF scenario batch per evaluation: `fake` (in process), `cpu` (HighwayEnv through CSF's scenario worker), `carla` (CARLA 0.9.16 in its pinned container via `candace csf simulator run carla`) | `bench/engines.py` |
| Grade | four oracles from CSF's `events.jsonl`: reached goal in time, no collision or lane departure, within the speed limit with no runtime fallback, settled at the end. Reward = fraction passed | `bench/oracles.py` |
| Cost | policy tokens + `sim_second_tokens` (20) x simulated seconds | `adapter.py` |
| Evidence | per trial: the CSF episode (manifest, events, trace), `verdict.json`, and `receipt.json` hashing scenario, controller and evidence | `runs/sim/jobs/<job>/<task>/t<i>/` |

A simulator crash or a runtime transport failure is **infra**: the trial gets
no verdict, is retried once, and otherwise counts as missing. It is never
scored as a controller failure. A controller CSF rejects is a controller
failure (reward 0).

Isaac Sim 6.0 is not an engine yet: CSF's Isaac worker has no vehicle asset or
actuator mapping, so straight-path driving cannot run there (see the CSF
scenario projector's execution blockers).

## Engines

```bash
export RRSI_SIM_ENGINE=fake        # no dependencies; CI and plumbing only
export RRSI_SIM_ENGINE=cpu         # CSF CPU harness, no GPU
export RRSI_SIM_CSF_ROOT=/path/to/candace/csf
export RRSI_SIM_CSF_RUNTIME=/path/to/csf   # Go runtime binary (JSONL mode)
export RRSI_SIM_ENGINE=carla       # needs the GPU to itself
export RRSI_SIM_CANDACE=/path/to/candace-server/server_admin_scripts/candace-cli/candace
```

`candace csf simulator build carla` builds the worker image first. The
`rollout` tool uses `RRSI_SIM_SURROGATE` (default: `cpu` when grading on CARLA
and the CPU engine is configured, otherwise `fake`). Smoke tests run on the
surrogate.

## Models

The policy and the search roles use the same backends as the toy domain
(`RRSI_POLICY_BACKEND`, `RRSI_LLM_BACKEND`); see `domains/toy/README.md`. A
logged-in Copilot CLI keeps the GPU free for the simulator:

```bash
export RRSI_LLM_BACKEND=copilot RRSI_CLI_MODEL=claude-sonnet-5.5
export RRSI_POLICY_BACKEND=copilot RRSI_POLICY_MODEL=gpt-5-mini
```

## Run

```bash
python3 rrsi.py --domain sim smoke
python3 rrsi.py --domain sim baseline --job base
python3 rrsi.py --domain sim heldout --label base2 --set evolve   # repeat H_0 for the noise band
python3 rrsi.py --domain sim calibrate --jobs base,heldout_base2
python3 rrsi.py --domain sim round --t 0
python3 rrsi.py --domain sim heldout --label champ
python3 rrsi.py --domain sim heldout --label base --ref <H_0 commit>
```
