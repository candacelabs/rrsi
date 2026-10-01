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
| Execute | one CSF scenario batch per evaluation: `fake` (in process), `cpu` (HighwayEnv through CSF's scenario worker), `carla` (CARLA 0.9.16 in its pinned container via `candace csf simulator run carla`), `phys` (held-out physics on the CPU, see below) | `bench/backends.py`, `grader/` |
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
export RRSI_SIM_BACKEND=phys        # preset: held-out physics grader, CPU only (see "Held-out physics")
export RRSI_SIM_PHYS_SALT_FILE=/path/to/private/phys/salt   # owner-only hex file, never committed
```

The `rollout` tool uses `RRSI_SIM_SURROGATE` (default: `cpu` when grading on
CARLA and the CPU backend is configured, otherwise `fake`; `none` switches the
preview off for an ablation, and `phys` is refused). Smoke tests run on the
surrogate.

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

Caveat: on the `cpu` backend the rollout surrogate (the `fake` kinematic
bicycle) is identical to the graded plant (line-for-line HighwayEnv; all 164
stored preview verdicts equalled the grade; traces agree to 2.2e-16), so a
preview reveals the grade exactly. That leak is not where H_1's gain came
from: H_1 submitted the prompt's gain template (steering gains -300 on input
0 and -600 on input 1, acceleration gain 800 on input 2) verbatim in 96 of 96
trials, and no preview ever changed its controller, so its gain on `cpu` came
from the prompt template, not the preview. The arm that does rely on the preview is
candidate B (it lost round 0 at 0.993): 12 of its 72 first previews failed and
it repaired 11 of them to reward 1.0. Held-out physics (next section) takes
the leak away so the two can be told apart. Also note that H_1 was admitted
against the lowest of the three H_0 measurements (0.861), not their mean.

## Held-out physics (the `phys` simulator backend)

A controller tuned in a simulator meets a real car whose tyres, actuators and
sensors are not the simulator's. The `phys` simulator backend models that gap:
the agent's preview is unchanged, and every graded episode runs a different
vehicle whose parameters are hidden and change from episode to episode.

| | Preview (`rollout` tool, `fake`) | Graded (`phys`) |
|---|---|---|
| Vehicle | HighwayEnv's kinematic bicycle: 5 m long, slip angle beta = atan(0.5 tan delta), one Euler step per 100 ms tick | dynamic single-track vehicle: mass, yaw inertia, axle loads, linear tyres saturating at friction x load, 10 Euler substeps per tick |
| Parameters | fixed and public | drawn per episode from `physics-v1`, hidden |
| Actuators | exact and instant | steering gain, offset, backlash, rate limit and dead time; acceleration gain and jerk limit; rolling and air resistance |
| What the controller sees | the exact state | noise and a constant bias on lateral offset and heading, noise on speed, one tick of latency in 40% of episodes |
| Controller runtime | in-process mirror of the CSF runtime | the CSF Go runtime (the mirror when `RRSI_SIM_CSF_RUNTIME` is unset) |
| Oracles | `bench/oracles.py` on the preview's state | the same oracles on the true state |

`physics-v1` (`grader/heldout_physics.py`). One uniform draw per row, in this
order; the nominal column is the vehicle at scale 0:

| # | Parameter | Nominal | Range | Why |
|---|---|---|---|---|
| 1 | mass (kg) | 1500 | 1275-1950 | occupants and cargo |
| 2 | yaw inertia factor | 1.0 | 0.9-1.2 | inertia = 2500 kg m^2 x mass/1500 x factor |
| 3 | centre of mass to front axle (m) | 1.2 | 1.10-1.40 | load distribution; rear = 2.7 - front |
| 4 | cornering stiffness scale | 1.0 | 0.6-1.25 | tyre wear, pressure, wet road; front = 90,000 N/rad x scale |
| 5 | rear stiffness margin | 0.2 | 0.05-0.40 | rear = front x (a/b + margin): always understeer |
| 6 | tyre friction | 0.9 | 0.5-1.0 | wet to dry asphalt |
| 7 | steering gain | 1.0 | 0.80-1.15 | steering ratio and compliance |
| 8 | steering offset (rad) | 0 | -0.004-0.004 | alignment pull |
| 9 | steering backlash (rad) | 0 | 0-0.004 | rack play |
| 10 | steering rate (rad/s) | unlimited | 0.4-1.5 | drawn as its inverse |
| 11 | jerk limit (m/s^3) | unlimited | 3-10 | drawn as its inverse |
| 12 | actuator dead time (ms) | 0 | 0-120 | rounded to 10 ms substeps |
| 13 | probability of one tick of sensing latency | 0 | 0.4 | the draw is still consumed |
| 14 | rolling resistance c_rr | 0 | 0.002-0.006 | kept low, see below |
| 15 | drag area C_d A (m^2) | 0 | 0.50-0.70 | air density 1.2 kg/m^3 |
| 16 | acceleration gain | 1.0 | 0.90-1.10 | powertrain and brake response |
| 17 | lateral sensing noise, s.d. (m) | 0 | 0-0.04 | lane-marking detection |
| 18 | heading sensing noise, s.d. (rad) | 0 | 0-0.004 | |
| 19 | speed sensing noise, s.d. (m/s) | 0 | 0-0.08 | |
| 20 | lateral sensing bias (m) | 0 | -0.05-0.05 | camera mounting |
| 21 | heading sensing bias (rad) | 0 | -0.003-0.003 | camera yaw |

Each episode's generator is seeded with the first 8 bytes of
HMAC-SHA256(salt, `rrsi-sim/physics-v1|<task>--t<i>`): 21 uniforms, one
draw for latency, then the seed of the sensing noise (the vehicle makes
exactly three Gaussian draws per tick, so every arm meets the same noise).
The episode id is the same in every evaluation, so trial i of scenario X
meets the same vehicle whichever harness is evaluated and however often:
comparisons between harnesses are paired. Nothing is drawn from a job name, a
batch id or a run id. Changing a range, the order or the seeding is a new
distribution name (`physics-v1.1`), never an edit in place.

Two compromises keep the grade about lane keeping. Real tyres have c_rr near
0.010-0.015, but a CSF controller is stateless (no integrator), so a steady
drag leaves a speed error a proportional law cannot remove; c_rr stays at
0.002-0.006. And the ranges are wide enough that aggressive or undamped gains
fail, but not so wide that a sound controller does: every one of the 96
campaign episode ids is solved by some controller of the 168-controller grid
below.

**What is hidden, and how.** The salt is an owner-only file outside the
repository; no default exists in code, and the worker exits before writing
`batch.json` without a valid one, so a batch is infra rather than graded on
nominal physics. Evidence carries only the distribution name, a salt id (the
first 16 hex digits of SHA-256 of the salt) and a SHA-256 of each episode's
draw, never the drawn values; `grader/phys_worker.py --explain <task>--t<i>`
prints one draw for an operator.

Five layers keep the grade out of the agent's reach, in order of strength:

1. **The policy and search models have no tools.** They run through logged-in
   agent CLIs; `rrsi/cli_llm.py` launches every call with an allowlist naming
   a tool that does not exist (Copilot CLI 1.0.90 reads `--available-tools ""`
   as *no restriction*, so that spelling is wrong), Codex with its shell
   disabled, and *verifies from the CLI's own event stream* that no tool was
   requested or run, dropping the reply otherwise. A model therefore cannot
   read a file, the salt or another arm's verdict. `grader/validity.py` is the
   after-the-fact backstop: it scans the CLI session logs for tool-execution
   events in a campaign window and voids any evaluation that overlaps one.
2. **The grade runs in a separate process with no salt in its environment.**
   Each evaluation runs the evaluated runner as `--phase propose` (policy,
   harness and preview, with every `RRSI_SIM_PHYS_*` variable stripped, never
   constructing the graded backend) and `--phase grade` (backend and oracles,
   no harness import). The adapter hands the salt-file path to the grade
   process over an inherited pipe (`RRSI_SIM_PHYS_SALT_FD`), not a variable, so
   the path is in no process environment the harness could read back through
   `/proc`. Launch the campaign with the salt variable set only for the
   campaign command, never `export`ed into the shell or the loop.
3. **The grade phase is the sole author of its evidence.** Verdict, receipt
   and episode are rebuilt every grade phase unless a receipt this grader wrote
   still binds the verdict and the episode files on disk, so a harness cannot
   plant a `verdict.json` in its own or a sibling trial and have it kept; a
   forged receipt would need the real episode, which needs the hidden grader.
4. **The propose phase can run in a container.** With
   `RRSI_SIM_PROPOSE_SANDBOX=docker` the harness runs with the grader package
   shadowed by an empty mount, no salt mounted or exported, its own PID
   namespace (so `/proc` holds no host process) and only its job's runs
   directory writable. This is the prevention layer for evolved rounds.
5. **The critic precheck is a lint, not the boundary.** It rejects the obvious
   reaches in a diff (grader module names, `os.environ`, `open(`, `Path(`,
   `importlib`, interpreter internals, frame access, `exec`/`compile`,
   threads), and the critic brief says the same, but a regex over text cannot
   catch string-concatenation obfuscation; prevention is layers 1-4.

So, at a single Unix user without the sandbox, reaching the grader is
**forbidden (critic and brief), not prevented: the harness runs in process as
the same user.** The fixed-arm campaign (H_0, H_1, B) is unaffected -- none of
those harnesses reads a file, the environment or the interpreter (git-grep: 0
hits) and the policy has no tools -- and the sandbox turns the general case
into prevention. Evaluating a commit whose runner predates the phase split
fails loudly: overlay its `domains/sim/harness/` onto the current grader
commit instead.

**Commit, then reveal.** Before a campaign, the operator logs SHA-256 of the
salt and its salt id; after the campaign, the salt is published, so anyone can
recompute every episode's draw hash and replay any episode. A new campaign
gets a new salt.

**Before and after.** `grader/agreement.py` measures how much a preview pass
says about the grade at zero policy tokens. Numbers here use the test fixture
salt; campaign numbers will come from the campaign salt.

| Preview reward vs graded reward | Before: `cpu` (HighwayEnv) | After: `phys` (`physics-v1`) |
|---|---|---|
| 168 linear controllers (gain grid) | 5,376 episodes (32 scenarios; deterministic) | 16,128 episodes (32 scenarios x 3 trials) |
| Exact reward agreement | 1.000 | 0.584 |
| Pearson correlation | 1.000 | 0.458 |
| Mean absolute gap | 0.000 | 0.229 |
| P(grade pass \| preview pass) | 1.000 | 0.397 |
| P(grade fail \| preview fail) | 1.000 | 0.985 |
| Stored trials that called the preview (H_1, B) | 164 of 164 previews equal the grade | no `phys` campaign yet |

The structural difference alone (the graded vehicle at nominal parameters)
already takes the grid's agreement from 1.000 to 0.826 (P(grade pass | preview
pass) 0.753); randomization takes it to 0.713 at half scale and 0.584 at full
scale. Of the 54 controllers that are perfect on the preview, the graded mean
ranges from 0.25 to 1.00.

| Fixed controller (32 scenarios x 30 vehicles) | Preview | Graded (practice / held out) | Full passes |
|---|---|---|---|
| CSF reference: -250 / -500 / 500 | 1.000 | 0.997 (0.996 / 1.000) | 0.989 |
| H_1's prompt template: -300 / -600 / 800 | 1.000 | 0.988 (0.986 / 0.995) | 0.974 |
| Aggressive: -1500 / -800 / 800 | 1.000 | 0.534 (0.517 / 0.583) | 0.367 |
| Lateral only, no heading damping: -400 / 0 / 500 | 0.992 | 0.421 (0.414 / 0.441) | 0.024 |

Over five further throwaway salts the reference scored 0.994-0.998 and the
aggressive controller 0.52-0.54.

H_0 never sees simulator output and H_1 and B saw only the unchanged
preview, so their stored `cpu` controllers are exact samples of what each arm
submits under `phys`. Replayed on the fixture salt (the same 24 practice
scenarios, paired per scenario, bootstrap 95% interval):

| Arm (stored trials) | `cpu` S | `phys` S | Paired gain over H_0 on `cpu` | Paired gain over H_0 on `phys` |
|---|---|---|---|---|
| H_0, practice (3 x 72) | 0.891 | 0.580 | | |
| H_1, practice (72) | 1.000 | 0.969 | +0.109 [0.076, 0.144] | +0.389 [0.317, 0.464] |
| B, practice (72) | 0.993 | 0.688 | +0.102 [0.066, 0.140] | +0.108 [0.030, 0.182] |

These replays are predictions, not results: the live campaign samples the
policy again.

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

# Held-out physics: paired, zero-token predictions from stored trials
python3 domains/sim/grader/agreement.py --runs runs/sim --jobs base,r0A \
    --salt-file /path/to/private/phys/salt --arms H0=base,H1=r0A --baseline H0 --grid --live
```
