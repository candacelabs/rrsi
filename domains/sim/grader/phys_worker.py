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
"""The `phys` simulator backend: graded episodes on hidden per-episode physics.

    python3 phys_worker.py --jobs FILE --output DIR --run-id ID [--runtime PATH]
    python3 phys_worker.py --explain straight-100--t0      (operator only)

It honours the backend contract of bench/backends.py: one result directory
per episode (manifest.json, events.jsonl, trace.jsonl, written by the same
episode loop as the fake backend) plus batch.json. Each episode runs the
grader-side vehicle (heldout_vehicle.py) with the parameters
heldout_physics.draw derives from the salt and the episode id. With
--runtime, the controller is compiled and evaluated by the CSF Go runtime
(CSF is Candace Labs' Go framework for AI-agent systems; the runtime is its
bounded controller compiler and executor, speaking JSON lines); otherwise by
the in-process mirror of it.

The salt comes from the file RRSI_SIM_PHYS_SALT_FILE names. Without a valid
salt the worker exits 2 before writing batch.json, so the whole batch is
infra: no episode is ever graded on nominal physics by mistake. The output
names the distribution, the salt id and a hash of each episode's draw, never
the drawn values. Episode ids that are not "<task>--t<i>" of a graded task
(the rollout preview's "rollout", for one) get an infra manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
for _path in (SIM / "bench", SIM / "data", HERE):
    sys.path.insert(0, str(_path))

import fake_backend                                   # noqa: E402
from heldout_physics import (DISTRIBUTION, EPISODE_ID, draw, load_salt,  # noqa: E402
                             physics_sha256, salt_id)
from heldout_vehicle import HeldoutVehicle            # noqa: E402
from scenarios import BY_ID                           # noqa: E402

NAME, VERSION = "phys", "1"
PLANT = "dynamic single-track vehicle, hidden per-episode physics"
RESPONSE_TIMEOUT = 10.0
LINE_LIMIT = 1_048_576


class InfraError(RuntimeError):
    """The worker could not run an episode (never evidence about a controller)."""


class GoStepper:
    """One persistent CSF Go runtime process per batch: COMPILE once per
    episode, then EVALUATE once per tick, as plain JSON lines."""

    runtime_name = "csf-go"

    def __init__(self, runtime: str):
        self.command = [runtime]
        self.process = None
        self.selector = None
        self.buffer = b""
        self.controller_hash = ""

    def _start(self) -> None:
        self.close()
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except OSError as error:
            raise InfraError(f"CSF runtime did not start: {error}") from error
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""

    def _request(self, request: dict) -> dict:
        if self.process is None or self.process.poll() is not None:
            self._start()
        try:
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            self.process.stdin.flush()
        except OSError as error:
            self.close()
            raise InfraError(f"CSF runtime closed its input: {error}") from error
        deadline = time.monotonic() + RESPONSE_TIMEOUT
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                self.close()
                raise InfraError("CSF runtime response deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                status = self.process.poll()
                self.close()
                raise InfraError(f"CSF runtime exited with status {status}")
            self.buffer += chunk
            if len(self.buffer) > LINE_LIMIT:
                self.close()
                raise InfraError("CSF runtime response exceeds one MiB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            response = json.loads(line)
        except ValueError as error:
            self.close()
            raise InfraError("CSF runtime wrote a line that is not JSON") from error
        if not isinstance(response, dict):
            self.close()
            raise InfraError("CSF runtime wrote a non-object response")
        return response

    def make(self, controller):
        """features -> (steering, acceleration); raises Rejected when CSF does
        not admit the controller."""
        response = self._request({"kind": "REQUEST_KIND_COMPILE", "controller": controller})
        if response.get("error"):
            raise fake_backend.Rejected(str(response["error"]))
        program = response.get("program")
        if not isinstance(program, dict):
            raise InfraError("CSF runtime compile returned no program")
        self.controller_hash = str(program.get("controller_hash", ""))

        def step(features: list[int]) -> tuple[int, int]:
            reply = self._request({"kind": "REQUEST_KIND_EVALUATE", "program": program,
                                   "observation": {"features": features}})
            action = reply.get("action")
            if reply.get("error") or not isinstance(action, dict):
                raise InfraError(f"CSF runtime evaluate failed: {reply.get('error', 'no action')}")
            try:
                steering, acceleration = int(action.get("steering", 0)), int(action.get("acceleration", 0))
            except (TypeError, ValueError) as error:
                raise InfraError("CSF runtime action is not an integer") from error
            if not (-1000 <= steering <= 1000 and -1000 <= acceleration <= 1000):
                raise InfraError("CSF runtime action outside the numeric profile")
            return steering, acceleration

        return step

    def close(self) -> None:
        if self.selector is not None:
            self.selector.close()
            self.selector = None
        if self.process is not None:
            process, self.process = self.process, None
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class MirrorStepper:
    """The in-process mirror of the CSF runtime (bench/fake_backend.py)."""

    runtime_name = "mirror"
    controller_hash = ""

    def make(self, controller):
        return fake_backend.mirror_stepper(controller)

    def close(self) -> None:
        pass


def simulator_block(salt: bytes, runtime_name: str) -> dict:
    return {"name": NAME, "version": VERSION, "distribution": DISTRIBUTION, "salt_id": salt_id(salt),
            "controller_runtime": runtime_name, "plant": PLANT, "gpu": False}


def _infra(output: Path, episode_id: str, run_id: str, reason: str) -> dict:
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "format": "csf-scenario-episode-v1", "status": "infra", "run_id": run_id, "episode": episode_id,
        "reason": reason}, indent=1, sort_keys=True))
    return {"id": episode_id, "status": "infra", "reason": reason}


def run_job(job: dict, out: Path, run_id: str, salt: bytes, stepper) -> dict:
    episode_id = str(job.get("id", ""))
    match = EPISODE_ID.match(episode_id)
    if not match or match.group(1) not in BY_ID:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", episode_id) or "_unnamed"
        return _infra(out / safe, episode_id, run_id, "episode id is not a graded task trial")
    target = out / episode_id
    if target.exists():
        return {"id": episode_id, "status": "infra", "reason": "duplicate episode id in the batch"}
    physics, noise_seed = draw(salt, episode_id)
    simulator = {**simulator_block(salt, stepper.runtime_name),
                 "physics_sha256": physics_sha256(physics, noise_seed)}
    try:
        summary = fake_backend.run_episode(
            job, target, run_id,
            plant_factory=lambda scenario: HeldoutVehicle(scenario, physics, random.Random(noise_seed)),
            stepper_factory=stepper.make, simulator=simulator)
    except Exception as error:  # noqa: BLE001 - anything but Rejected is infrastructure
        return _infra(target, episode_id, run_id, f"{type(error).__name__}: {error}"[:500])
    if stepper.controller_hash and summary["status"] == "completed":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["controller_hash"] = stepper.controller_hash
        (target / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs")
    ap.add_argument("--output")
    ap.add_argument("--run-id")
    ap.add_argument("--runtime", default="")
    ap.add_argument("--explain", metavar="EPISODE_ID", help="print one episode's draw (operator only)")
    a = ap.parse_args()
    try:
        salt = load_salt(os.environ.get("RRSI_SIM_PHYS_SALT_FILE", ""))
    except SystemExit as error:
        print(f"phys simulator backend: {error}", file=sys.stderr)
        return 2
    if a.explain:
        physics, noise_seed = draw(salt, a.explain)
        print(json.dumps({"episode": a.explain, "distribution": DISTRIBUTION, "salt_id": salt_id(salt),
                          "physics": physics.to_dict(), "noise_seed": noise_seed,
                          "physics_sha256": physics_sha256(physics, noise_seed)}, indent=1, sort_keys=True))
        return 0
    if not (a.jobs and a.output and a.run_id):
        ap.error("--jobs, --output and --run-id are required")
    jobs = json.loads(Path(a.jobs).read_text())["episodes"]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stepper = GoStepper(a.runtime) if a.runtime else MirrorStepper()
    try:
        results = [run_job(job, out, a.run_id, salt, stepper) for job in jobs]
    finally:
        stepper.close()
    (out / "batch.json").write_text(json.dumps({
        "format": "csf-scenario-batch-v1", "run_id": a.run_id,
        "simulator": simulator_block(salt, stepper.runtime_name), "episodes": results},
        indent=1, sort_keys=True))
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("completed", "rejected", "infra")}
    print(f"[phys] {DISTRIBUTION} salt {salt_id(salt)} runtime {stepper.runtime_name}: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
