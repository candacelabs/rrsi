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
"""Engines that execute a batch of (scenario, controller) episodes.

CSF (Candace Labs' Go framework for AI-agent systems, developer preview; see
domains/sim/README.md) owns the worker these engines run. `candace` is the
operator CLI of the monorepo that hosts CSF.

RRSI_SIM_ENGINE selects one:

  fake   bench/fake_engine.py, in process, no dependencies (tests and CI)
  cpu    CSF's scenario worker on the CPU HighwayEnv plant of CSF's training
         harness, with the Go runtime; needs RRSI_SIM_CSF_ROOT (the `csf`
         directory of a candace checkout), RRSI_SIM_CSF_RUNTIME and `uv`
  carla  the same worker in the pinned CARLA 0.9.16 container, started with
         `candace csf simulator run carla` (RRSI_SIM_CANDACE, default
         `candace`); needs the GPU to itself. Peak VRAM is sampled with
         nvidia-smi during the batch and saved as vram.json.

Every engine writes the CSF batch layout into `out`: batch.json plus one
directory per episode id with manifest.json, events.jsonl and trace.jsonl.
`check` is the admission test the agent's `check` tool uses: the Go runtime's
compile when RRSI_SIM_CSF_RUNTIME is set, otherwise the fake mirror.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import fake_engine

ENGINE = os.environ.get("RRSI_SIM_ENGINE", "fake").strip().lower()
CSF_ROOT = os.environ.get("RRSI_SIM_CSF_ROOT", "")
CSF_RUNTIME = os.environ.get("RRSI_SIM_CSF_RUNTIME", "")
CANDACE = os.environ.get("RRSI_SIM_CANDACE", "candace")
BATCH_TIMEOUT = float(os.environ.get("RRSI_SIM_BATCH_TIMEOUT", "3600"))


class InfraError(RuntimeError):
    """The engine could not run (not evidence about any controller)."""


def check(controller) -> tuple[bool, str]:
    """-> (admitted, error). Never raises for a bad controller."""
    if CSF_RUNTIME:
        return _go_compile(controller)
    try:
        fake_engine.check(controller)
    except fake_engine.Rejected as error:
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


class Engine:
    name = "base"

    def execute(self, jobs: list[dict], out: Path, run_id: str) -> dict:
        """Run the batch into `out`; returns engine facts (describe, VRAM, wall seconds)."""
        raise NotImplementedError


class FakeEngine(Engine):
    name = "fake"

    def __init__(self, crash_ids=()):
        self.crash_ids = frozenset(crash_ids or os.environ.get("RRSI_SIM_FAKE_CRASH", "").split(",")) - {""}

    def execute(self, jobs, out, run_id):
        started = time.monotonic()
        fake_engine.run_batch(jobs, out, run_id, self.crash_ids)
        return {"engine": self.name, "simulator": fake_engine.describe(),
                "wall_seconds": time.monotonic() - started}


def _write_jobs(jobs: list[dict], path: Path) -> None:
    path.write_text(json.dumps({"format": "csf-scenario-jobs-v1", "episodes": [
        {"id": j["id"], "scenario": j["scenario"], "goal_metres": j["goal_metres"],
         "controller": j.get("controller")} for j in jobs]}, indent=1))


class CpuEngine(Engine):
    name = "cpu"

    def execute(self, jobs, out, run_id):
        if not CSF_ROOT or not CSF_RUNTIME:
            raise InfraError("cpu engine needs RRSI_SIM_CSF_ROOT and RRSI_SIM_CSF_RUNTIME")
        root = Path(CSF_ROOT)
        out.mkdir(parents=True, exist_ok=True)
        jobs_path = out / "jobs.json"
        _write_jobs(jobs, jobs_path)
        cmd = ["uv", "run", "--quiet", "--project", str(root / "examples" / "training"), "--locked",
               "python", str(root / "examples" / "simulators" / "scenario_worker.py"),
               "--plant", "highway", "--jobs", str(jobs_path), "--output", str(out),
               "--run-id", run_id, "--runtime", CSF_RUNTIME]
        # The worker imports the training runtime client and the generated
        # protobuf contract from PYTHONPATH, as it does inside the CARLA image.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(root / "examples" / "training"), str(root / "tools" / "codegen" / "generated" / "python")])}
        started = time.monotonic()
        with (out / "worker.log").open("w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=BATCH_TIMEOUT, env=env)
        if not (out / "batch.json").is_file():
            raise InfraError(f"cpu worker exited {code} without batch.json (see {out / 'worker.log'})")
        batch = json.loads((out / "batch.json").read_text())
        return {"engine": self.name, "simulator": batch.get("simulator"), "exit": code,
                "wall_seconds": time.monotonic() - started}


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


class CarlaEngine(Engine):
    name = "carla"

    def execute(self, jobs, out, run_id):
        if shutil.which(CANDACE) is None and not Path(CANDACE).is_file():
            raise InfraError(f"candace CLI not found ({CANDACE}); set RRSI_SIM_CANDACE")
        staging = out.parent / f"{out.name}.jobs.json"
        _write_jobs(jobs, staging)
        cmd = [CANDACE, "csf", "simulator", "run", "carla", "--jobs", str(staging.resolve()),
               "--output", str(out.resolve()), "--run-id", run_id,
               "--max-wall-seconds", str(int(BATCH_TIMEOUT))]
        started = time.monotonic()
        with VramSampler() as vram:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BATCH_TIMEOUT + 120)
        facts = {"engine": self.name, "exit": result.returncode, "wall_seconds": time.monotonic() - started,
                 "vram": vram.summary(), "cli_tail": (result.stdout + result.stderr)[-1500:]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "vram.json").write_text(json.dumps({"samples": vram.samples, **facts["vram"]}))
        if not (out / "batch.json").is_file():
            raise InfraError(f"CARLA batch exited {result.returncode} without batch.json: {facts['cli_tail']}")
        facts["simulator"] = json.loads((out / "batch.json").read_text()).get("simulator")
        return facts


def get(name: str | None = None) -> Engine:
    name = (name or ENGINE).strip().lower()
    engines = {"fake": FakeEngine, "cpu": CpuEngine, "carla": CarlaEngine}
    if name not in engines:
        raise SystemExit(f"unknown RRSI_SIM_ENGINE {name!r}; expected one of {sorted(engines)}")
    return engines[name]()


def surrogate_name() -> str:
    """The cheaper engine behind the agent's rollout tool; never the graded one."""
    explicit = os.environ.get("RRSI_SIM_SURROGATE", "").strip().lower()
    if explicit:
        return explicit
    if ENGINE == "carla" and CSF_ROOT and CSF_RUNTIME:
        return "cpu"
    return "fake"
