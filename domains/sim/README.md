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
| Tasks | 24 practice + 8 held-out straight-path scenarios: CSF `Scenario` JSON (seeded target speed 5-18 m/s, 10-24 s, start offset up to 1.6 m in a 2 m half-lane, heading up to 0.15 rad) plus a goal distance | `data/scenarios.py` |
| Agent | frozen policy LLM + the evolvable harness H_0 (a JSON-action loop with a CSF `check` tool; a `rollout` tool on a cheap surrogate plant is passed in but unused by H_0) | `harness/` |
| Check | CSF's bounded controller compiler (the Go runtime, reading JSON lines on stdin) | `bench/backends.py` |
| Execute | one CSF scenario batch per evaluation: `fake` (in process), `cpu` (HighwayEnv through CSF's scenario worker), `carla` (CARLA 0.9.16 in its pinned container via `candace csf simulator run carla`) | `bench/backends.py` |
| Grade | four oracles from CSF's `events.jsonl`: reached goal in time, no collision or lane departure, within the speed limit with no runtime fallback, settled at the end. Reward = fraction passed | `bench/oracles.py` |
| Cost | policy tokens + `sim_second_tokens` (20) x simulated seconds | `adapter.py` |
| Evidence | per trial: the CSF episode (manifest, events, trace), `verdict.json`, and `receipt.json` hashing scenario, controller and evidence | `runs/sim/jobs/<job>/<task>/t<i>/` |

A simulator crash or a runtime transport failure is **infra**: the trial gets
no verdict, is retried once, and otherwise counts as missing. It is never
scored as a controller failure. A controller CSF rejects is a controller
failure (reward 0).

Isaac Sim 6.0 is not a simulator backend yet: CSF's Isaac worker has no vehicle asset or
actuator mapping, so straight-path driving cannot run there (see the CSF
scenario projector's execution blockers).

## Simulator backends

The domain only sees a simulator-backend interface: submit a batch of episodes
(scenario + controller + goal), get one result directory per episode
(`manifest.json` with completed / rejected / infra, `events.jsonl`,
`trace.jsonl`). `bench/backends.py` documents the contract. Anything that
honours it can be a backend, so the domain can later target another controller
host (for example a ROS-side controller) or a direct simulator container
without changing.

```bash
export RRSI_SIM_BACKEND=fake        # no dependencies; CI and plumbing only
export RRSI_SIM_BACKEND=command     # any executable honouring the contract:
export RRSI_SIM_BACKEND_COMMAND="/path/to/backend --flag"   # gets --jobs/--output/--run-id appended
export RRSI_SIM_BACKEND=cpu         # preset: CSF's scenario worker on the CPU HighwayEnv plant
export RRSI_SIM_CSF_ROOT=/path/to/candace/csf
export RRSI_SIM_CSF_RUNTIME=/path/to/csf   # CSF Go runtime binary (reads JSON lines)
export RRSI_SIM_BACKEND=carla       # preset: the same worker in CARLA 0.9.16; needs the GPU to itself
export RRSI_SIM_CANDACE=/path/to/candace-server/server_admin_scripts/candace-cli/candace
```

The `rollout` tool uses `RRSI_SIM_SURROGATE` (default: `cpu` when grading on
CARLA and the CPU backend is configured, otherwise `fake`). Smoke tests run on
the surrogate.

## Noise band and first results (CPU backend)

The physics is deterministic: the CSF reference controller (steering
-250 x lateral - 500 x heading, acceleration 500 x speed gap) run 3 times on
12 scenarios gave identical rewards (36 episodes, 0.0 m spread). All the noise
comes from the policy model, whose sampling the Copilot CLI backend cannot
make deterministic, so the band is shrunk with more episodes instead:

| Setting | Episodes per evaluation | Unchanged H_0, three evaluations | Noise band delta (z = 2) |
|---|---|---|---|
| 8 practice scenarios, k = 2 | 16 | 0.953, 0.812 (two) | 0.281 |
| 24 practice scenarios, k = 3 | 72 | 0.861, 0.934, 0.879 | **0.108** |

Calibration at 72 episodes cost 216 episodes, about 27 minutes and about
260k policy tokens (gpt-5-mini through the Copilot CLI).

One RRSI round at delta 0.108 (search roles: claude-sonnet-5.5 through the
Copilot CLI) accepted a candidate that adds sign rules and a gain template to
the prompt and, before the first submit, runs `check` and a surrogate
`rollout` and shows the result to the policy:

| | Practice (72 episodes) | Held out (24 episodes) | Policy tokens per trial |
|---|---|---|---|
| H_0 | 0.861 (repeats 0.934, 0.879) | 0.969, 0.906 | 1,174 |
| H_1 | **1.000** | **1.000** | 2,534 |

Caveat: on the `cpu` backend the rollout surrogate is the `fake` kinematic
bicycle, which is nearly the graded HighwayEnv plant, so the rollout almost
reveals the grade. Grading on CARLA keeps a real gap between surrogate and
graded simulator; that run is pending.

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
