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
"""Evaluate the driving harness checked out next to this file: k trials per task.

    python3 bench/run_tasks.py --runs RUNS --job JOB --ids a,b,c --n K

Trial i of task X is RUNS/jobs/JOB/X/t<i>/:

  traj.json, meta.json   the agent's conversation; meta.status ok|crash|infra
  controller.json        the submitted controller (absent: no submission)
  episode/               the CSF episode evidence (manifest, events, trace)
  verdict.json           oracles, reward, simulated seconds
  receipt.json           hashes binding scenario, controller and evidence

Three resume-safe phases: propose (the policy writes a controller), execute
(one engine batch for every proposed trial still without evidence) and grade.
A policy-endpoint failure or a simulator crash leaves the trial without a
verdict, recorded as infra, and the next invocation retries it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SIM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "bench"))
sys.path.insert(0, str(SIM / "data"))

import engines                       # noqa: E402
import oracles                       # noqa: E402
from harness.agent import run_agent  # noqa: E402
from scenarios import BY_ID, brief   # noqa: E402

MAX_STEPS = int(os.environ.get("RRSI_SIM_MAX_STEPS", "6"))
CONCURRENCY = int(os.environ.get("RRSI_SIM_CONCURRENCY", "4"))


def _load_policy():
    """The frozen policy transport is shared with the toy domain."""
    path = SIM.parent / "toy" / "policy.py"
    spec = importlib.util.spec_from_file_location("rrsi_sim_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True))


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def make_tools(task: dict, surrogate: engines.Engine, scratch: Path):
    counter = {"n": 0}

    def check(controller) -> dict:
        ok, error = engines.check(controller)
        return {"ok": ok, "error": error}

    def rollout(controller) -> dict:
        counter["n"] += 1
        out = scratch / f"rollout-{counter['n']}"
        shutil.rmtree(out, ignore_errors=True)
        job = {"id": "rollout", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
               "controller": controller}
        try:
            surrogate.execute([job], out, f"rollout-{counter['n']}")
        except engines.InfraError as error:
            return {"error": f"surrogate unavailable: {error}"}
        verdict = oracles.grade(out / "rollout", task)
        verdict["engine"] = surrogate.name
        return verdict

    return check, rollout


def propose(runs: Path, job: str, tid: str, i: int, policy, surrogate) -> str:
    tdir = runs / "jobs" / job / tid / f"t{i}"
    meta = read(tdir / "meta.json")
    if meta and meta.get("status") in ("ok", "crash"):
        return "kept"
    tdir.mkdir(parents=True, exist_ok=True)
    task = BY_ID[tid]
    check, rollout = make_tools(task, surrogate, tdir / "rollouts")
    t0 = time.time()
    meta = {"task": tid, "trial": i, "max_steps": MAX_STEPS, "model": policy.MODEL}
    try:
        out = run_agent(brief(task), policy.chat, check, rollout, MAX_STEPS)
        controller = out.get("controller")
        msgs = out.get("messages") or []
        meta.update(status="ok", tokens=int(out.get("tokens") or 0),
                    steps=sum(1 for m in msgs if m.get("role") == "assistant"))
    except (policy.PolicyError, engines.InfraError) as e:
        meta.update(status="infra", error=str(e)[:500], seconds=round(time.time() - t0, 1))
        write(tdir / "meta.json", meta)
        return "infra"
    except Exception:  # noqa: BLE001 - a harness crash is the harness's failure
        controller, msgs = None, []
        meta.update(status="crash", error=traceback.format_exc()[-2000:], tokens=0, steps=0)
    meta["seconds"] = round(time.time() - t0, 1)
    write(tdir / "traj.json", {"messages": msgs})
    if controller is not None:
        write(tdir / "controller.json", controller)
    write(tdir / "meta.json", meta)
    return "proposed"


def _batch_id(job: str) -> str:
    stem = re.sub(r"[^a-z0-9-]+", "-", job.lower()).strip("-") or "job"
    return f"{stem[:40]}-{time.strftime('%Y%m%dt%H%M%S')}-{os.getpid() % 10000}"


def execute(runs: Path, job: str, pending: list[tuple[str, int]], engine: engines.Engine) -> dict:
    """One engine batch for every pending trial; moves each episode into its trial."""
    if not pending:
        return {}
    jobs = []
    for tid, i in pending:
        task = BY_ID[tid]
        jobs.append({"id": f"{tid}--t{i}", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
                     "controller": read(runs / "jobs" / job / tid / f"t{i}" / "controller.json")})
    batch_id = _batch_id(job)
    out = runs / "jobs" / job / "_batches" / batch_id
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        facts = engine.execute(jobs, out, batch_id)
    except (engines.InfraError, OSError) as error:
        facts = {"engine": engine.name, "error": str(error)}
    facts["batch"] = batch_id
    facts["episodes"] = len(jobs)
    write(out.parent / f"{batch_id}.json", facts)
    for (tid, i), j in zip(pending, jobs):
        src, dst = out / j["id"], runs / "jobs" / job / tid / f"t{i}" / "episode"
        if (src / "manifest.json").is_file() and read(src / "manifest.json").get("status") != "infra":
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            meta = read(dst.parent / "meta.json") or {}
            meta["batch"] = batch_id
            write(dst.parent / "meta.json", meta)
    return facts


def grade_trial(tdir: Path, task: dict, engine_name: str) -> str:
    meta = read(tdir / "meta.json") or {}
    if (tdir / "verdict.json").is_file():
        return "kept"
    if meta.get("status") == "infra":
        return "infra"
    if meta.get("status") == "crash" or not (tdir / "controller.json").is_file():
        verdict = {"status": "no_submission", "reward": 0.0,
                   "oracles": {name: False for name in oracles.ORACLES}, "simulation_seconds": 0.0}
    else:
        verdict = oracles.grade(tdir / "episode", task)
        if verdict["status"] == "infra":
            return "infra"
    verdict["engine"] = engine_name
    write(tdir / "verdict.json", verdict)
    episode = tdir / "episode"
    write(tdir / "receipt.json", {
        "format": "rrsi-sim-receipt-v1", "task": task["id"], "engine": engine_name,
        "batch": meta.get("batch", ""), "scenario_sha256": sha256_json(task["scenario"]),
        "goal_metres": task["goal_metres"],
        "controller_sha256": sha256_file(tdir / "controller.json"),
        "csf_controller_hash": (read(episode / "manifest.json") or {}).get("controller_hash", ""),
        "evidence_sha256": {name: sha256_file(episode / name)
                            for name in ("manifest.json", "events.jsonl", "trace.jsonl")},
        "verdict_sha256": sha256_file(tdir / "verdict.json")})
    return "passed" if verdict.get("reward") == 1.0 else "graded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--engine", default=None, help="override RRSI_SIM_ENGINE (smoke uses the surrogate)")
    a = ap.parse_args()
    runs = Path(a.runs)
    work = [(t, i) for t in a.ids.split(",") if t for i in range(a.n)]
    engine = engines.get(a.engine)
    surrogate = engines.get(engines.surrogate_name())
    policy = _load_policy()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        proposed = list(ex.map(lambda w: propose(runs, a.job, *w, policy, surrogate), work))
    pending = [(t, i) for t, i in work
               if (runs / "jobs" / a.job / t / f"t{i}" / "controller.json").is_file()
               and (read(runs / "jobs" / a.job / t / f"t{i}" / "meta.json") or {}).get("status") == "ok"
               and not (runs / "jobs" / a.job / t / f"t{i}" / "verdict.json").is_file()
               and not (runs / "jobs" / a.job / t / f"t{i}" / "episode" / "manifest.json").is_file()]
    facts = execute(runs, a.job, pending, engine)
    graded = [grade_trial(runs / "jobs" / a.job / t / f"t{i}", BY_ID[t], engine.name) for t, i in work]
    counts = {k: graded.count(k) for k in sorted(set(graded))}
    print(f"[sim] job={a.job} engine={engine.name} proposals={ {k: proposed.count(k) for k in sorted(set(proposed))} } "
          f"executed={len(pending)} grades={counts} batch={facts.get('batch', '-')} "
          f"vram={facts.get('vram', {}).get('peak_mib', '-')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
