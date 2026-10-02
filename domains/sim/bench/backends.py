# Copyright 2026 Candace Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The backend interface: submit episodes (scenario + controller), get results.

The domain never talks to a simulator directly. A backend takes a batch of
episodes, each {"id", "scenario", "goal_metres", "controller"} (the scenario
and controller as generated protobuf JSON of CSF, Candace Labs' Go framework
for AI-agent systems, in developer preview; see domains/sim/README.md), and
writes one result directory per episode id into `out`, plus `batch.json`:

  manifest.json  "status": completed | rejected | infra, "termination",
                 "steps_completed", "simulation_seconds", "reason"
  events.jsonl   one ResearchEvent JSON row per measurement per physics step
                 (scenario_longitudinal_metres, scenario_lateral_metres,
                 scenario_heading_error_radians, simulator_speed_mps,
                 scenario_collisions, runtime_fallbacks)
  trace.jsonl    one row per physics step (observation, action, state)

`rejected` means the controller failed admission (a controller result);
`infra` means the backend failed (never a controller result). Anything that
honours this contract can be a backend: CSF's scenario worker today, a
ROS-side controller or a direct simulator container later, with no change to
the domain. RRSI_SIM_BACKEND selects one:

  fake     bench/fake_backend.py, in process, no dependencies (tests and CI)
  command  any executable: RRSI_SIM_BACKEND_COMMAND is its argv, and the backend
           appends `--jobs FILE --output DIR --run-id ID`
  cpu      a `command` preset: CSF's scenario worker on the CPU HighwayEnv
           plant of CSF's training harness with the Go runtime (needs
           RRSI_SIM_CSF_ROOT, the `csf` directory of a candace checkout,
           RRSI_SIM_CSF_RUNTIME and `uv`)
  carla    a `command` preset: the same worker in the pinned CARLA 0.9.16
           container through `candace csf simulator run carla` (`candace` is
           the operator CLI of the monorepo that hosts CSF; RRSI_SIM_CANDACE).
           Needs the GPU to itself; peak VRAM is sampled with nvidia-smi
           during the batch and saved as vram.json.
  phys     a `command` preset: grader/phys_worker.py, the held-out-physics
           grader. Every graded episode runs a dynamic single-track vehicle
           whose parameters are drawn per episode from a secret salt
           (RRSI_SIM_PHYS_SALT_FILE, a file outside the repository) and the
           episode id; the controller runs in the CSF Go runtime when
           RRSI_SIM_CSF_RUNTIME is set. CPU only. See README "Held-out physics".
  none     no simulator: every batch is infra. Only as the rollout surrogate,
           to switch the preview off for an ablation.

The agent's `rollout` tool runs one episode on a surrogate backend
(`surrogate_name`): RRSI_SIM_SURROGATE, or `cpu` when grading on CARLA, or
`fake`. The `fake` bicycle is identical to the HighwayEnv plant the `cpu`
backend grades on, so on `cpu` the preview reveals the grade exactly; on
`phys` it is the nominal kinematic model and the grade is not. `phys` can
never be the surrogate.

`check` is the admission test behind the agent's `check` tool: the CSF Go
runtime's compile when RRSI_SIM_CSF_RUNTIME is set, otherwise the fake mirror.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import fake_backend

BACKEND = os.environ.get("RRSI_SIM_BACKEND", "fake").strip().lower()
CSF_ROOT = os.environ.get("RRSI_SIM_CSF_ROOT", "")
CSF_RUNTIME = os.environ.get("RRSI_SIM_CSF_RUNTIME", "")
CANDACE = os.environ.get("RRSI_SIM_CANDACE", "candace")
BATCH_TIMEOUT = float(os.environ.get("RRSI_SIM_BATCH_TIMEOUT", "3600"))
def _salt_file_at_import() -> str:
    """The grader's salt-file path, read once, before any harness code runs.

    The adapter hands the grade subprocess its path over an inherited file
    descriptor (`RRSI_SIM_PHYS_SALT_FD`), not a variable, so the path is not
    in this process's environment for a same-user harness to read back through
    `/proc`. `RRSI_SIM_PHYS_SALT_FILE` stays as the fallback for `--phase all`
    and tests. Either way only the path is read here; the salt bytes are read
    only by the phys worker it launches."""
    fd = os.environ.get("RRSI_SIM_PHYS_SALT_FD", "")
    if fd:
        try:
            with os.fdopen(int(fd), "r", closefd=True) as handle:
                return handle.read().strip()
        except (OSError, ValueError):
            return ""
    return os.environ.get("RRSI_SIM_PHYS_SALT_FILE", "")


# Captured at import: the runner removes it from its own environment before
# any harness code runs, and only the grading process is given it, over a fd.
_PHYS_SALT_FILE = _salt_file_at_import()
SIM = Path(__file__).resolve().parent.parent
BACKENDS = ("fake", "command", "cpu", "carla", "phys", "none")


class InfraError(RuntimeError):
    """The backend could not run (not evidence about any controller)."""


def check(controller) -> tuple[bool, str]:
    """-> (admitted, error). Never raises for a bad controller."""
    if CSF_RUNTIME:
        return _go_compile(controller)
    try:
        fake_backend.check(controller)
    except fake_backend.Rejected as error:
        return False, str(error)
    return True, ""


def _go_compile(controller) -> tuple[bool, str]:
    request = json.dumps({"kind": "REQUEST_KIND_COMPILE", "controller": controller})
    try:
        out = subprocess.run([CSF_RUNTIME], input=request + "\n", capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InfraError(f"CSF runtime unavailable: {error}") from error
    try:
        response = json.loads(out.stdout.splitlines()[0])
    except (IndexError, ValueError) as error:
        raise InfraError(f"CSF runtime gave no response: {out.stderr[-300:]}") from error
    if response.get("error"):
        return False, response["error"]
    return True, ""


class Backend:
    """Submit a batch of episodes; results land in `out` (see the module doc)."""

    name = "base"

    def execute(self, jobs: list[dict], out: Path, run_id: str) -> dict:
        """Run the batch into `out`; returns backend facts (describe, VRAM, wall seconds)."""
        raise NotImplementedError


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, crash_ids=()):
        self.crash_ids = frozenset(crash_ids or os.environ.get("RRSI_SIM_FAKE_CRASH", "").split(",")) - {""}

    def execute(self, jobs, out, run_id):
        started = time.monotonic()
        fake_backend.run_batch(jobs, out, run_id, self.crash_ids)
        return {"backend": self.name, "simulator": fake_backend.describe(),
                "wall_seconds": time.monotonic() - started}


def write_jobs(jobs: list[dict], path: Path) -> None:
    path.write_text(json.dumps({"format": "csf-scenario-jobs-v1", "episodes": [
        {"id": j["id"], "scenario": j["scenario"], "goal_metres": j["goal_metres"],
         "controller": j.get("controller")} for j in jobs]}, indent=1))


class CommandBackend(Backend):
    """Any executable honouring the contract: argv + --jobs/--output/--run-id."""

    def __init__(self, name: str, argv: list[str], env: dict | None = None, sample_vram: bool = False):
        if not argv:
            raise InfraError(f"backend {name!r} has no command")
        self.name, self.argv, self.env, self.sample_vram = name, list(argv), env, sample_vram

    def execute(self, jobs, out, run_id):
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = out.parent / f"{out.name}.jobs.json"
        write_jobs(jobs, staging)
        cmd = [*self.argv, "--jobs", str(staging), "--output", str(out), "--run-id", run_id]
        env = {**os.environ, **(self.env or {})}
        started = time.monotonic()
        sampler = VramSampler() if self.sample_vram else None
        try:
            if sampler:
                sampler.__enter__()
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=BATCH_TIMEOUT + 120)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise InfraError(f"backend {self.name} did not run: {error}") from error
        finally:
            if sampler:
                sampler.__exit__()
        facts = {"backend": self.name, "exit": result.returncode, "wall_seconds": time.monotonic() - started,
                 "log_tail": (result.stdout + result.stderr)[-1500:]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "backend.log").write_text(result.stdout + result.stderr)
        if sampler:
            facts["vram"] = sampler.summary()
            (out / "vram.json").write_text(json.dumps({"samples": sampler.samples, **facts["vram"]}))
        if not (out / "batch.json").is_file():
            raise InfraError(f"backend {self.name} exited {result.returncode} without batch.json: {facts['log_tail']}")
        facts["simulator"] = json.loads((out / "batch.json").read_text()).get("simulator")
        return facts


class VramSampler:
    """Peak GPU memory in use while a batch runs (nvidia-smi, once a second)."""

    def __init__(self, interval: float = 1.0):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                text = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], capture_output=True,
                                      text=True, timeout=10).stdout
                used, total = (int(x) for x in text.splitlines()[0].split(","))
                self.samples.append((time.time(), used, total))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        used = [u for _, u, _ in self.samples]
        return {"samples": len(used), "baseline_mib": used[0], "peak_mib": max(used),
                "total_mib": self.samples[0][2]}


def cpu_backend() -> CommandBackend:
    if not CSF_ROOT or not CSF_RUNTIME:
        raise InfraError("cpu backend needs RRSI_SIM_CSF_ROOT and RRSI_SIM_CSF_RUNTIME")
    root = Path(CSF_ROOT)
    # The worker imports the training runtime client and the generated
    # protobuf contract from PYTHONPATH, as it does inside the CARLA image.
    env = {"PYTHONPATH": os.pathsep.join([str(root / "examples" / "training"),
                                          str(root / "tools" / "codegen" / "generated" / "python")])}
    return CommandBackend("cpu", ["uv", "run", "--quiet", "--project", str(root / "examples" / "training"),
                                 "--locked", "python", str(root / "examples" / "simulators" / "scenario_worker.py"),
                                 "--plant", "highway", "--runtime", CSF_RUNTIME], env)


class NoneBackend(Backend):
    """No simulator: the preview-off ablation's surrogate."""

    name = "none"

    def execute(self, jobs, out, run_id):
        raise InfraError("preview disabled")


def salt_problem(path: str) -> str:
    """Why the phys salt file is unusable, or "". Only stats the file: the
    process that checks it never reads the salt."""
    if not path:
        return "RRSI_SIM_PHYS_SALT_FILE is unset"
    try:
        info = os.stat(path)
    except OSError as error:
        return f"salt file unreadable: {error.strerror}"
    if info.st_mode & 0o077:
        return "salt file must be readable only by its owner (chmod 600)"
    if info.st_size < 64:
        return "salt file must hold at least 64 hex digits"
    return ""


def phys_backend() -> CommandBackend:
    problem = salt_problem(_PHYS_SALT_FILE)
    if problem:
        raise SystemExit(f"phys simulator backend: {problem}")
    argv = [sys.executable, str(SIM / "grader" / "phys_worker.py")]
    if CSF_RUNTIME:
        argv += ["--runtime", CSF_RUNTIME]
    return CommandBackend("phys", argv, {"RRSI_SIM_PHYS_SALT_FILE": _PHYS_SALT_FILE})


def carla_backend() -> CommandBackend:
    if shutil.which(CANDACE) is None and not Path(CANDACE).is_file():
        raise InfraError(f"candace CLI not found ({CANDACE}); set RRSI_SIM_CANDACE")
    return CommandBackend("carla", [CANDACE, "csf", "simulator", "run", "carla",
                                   "--max-wall-seconds", str(int(BATCH_TIMEOUT))], sample_vram=True)


def get(name: str | None = None) -> Backend:
    name = (name or BACKEND).strip().lower()
    if name == "fake":
        return FakeBackend()
    if name == "command":
        return CommandBackend("command", shlex.split(os.environ.get("RRSI_SIM_BACKEND_COMMAND", "")))
    if name == "cpu":
        return cpu_backend()
    if name == "carla":
        return carla_backend()
    if name == "phys":
        return phys_backend()
    if name == "none":
        return NoneBackend()
    raise SystemExit(f"unknown RRSI_SIM_BACKEND {name!r}; expected one of {', '.join(BACKENDS)}")


def surrogate_name() -> str:
    """The backend behind the agent's rollout tool. On `cpu` the fake bicycle
    is identical to the graded HighwayEnv plant, so the preview reveals the
    grade; on `phys` it is the nominal kinematic model, not the graded
    vehicle; `none` switches the preview off (see README)."""
    explicit = os.environ.get("RRSI_SIM_SURROGATE", "").strip().lower()
    if explicit == "phys":
        raise SystemExit("the held-out-physics grader cannot serve as the rollout preview")
    if explicit:
        return explicit
    if BACKEND == "carla" and CSF_ROOT and CSF_RUNTIME:
        return "cpu"
    return "fake"
