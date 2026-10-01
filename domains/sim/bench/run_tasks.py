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

Three resume-safe steps: propose (the policy writes a controller), execute
(one backend batch for every proposed trial still without evidence) and grade.
A policy-endpoint failure or a simulator crash leaves the trial without a
verdict, recorded as infra, and the next invocation retries it.

`--phase propose` runs only the agent (policy, harness, rollout preview) and
never constructs the graded backend; `--phase grade` runs only execute and
grade, with no policy and no harness import. The adapter runs them as two
processes and gives the propose process no RRSI_SIM_PHYS_* variable, so the
harness never runs in the process that grades. `--phase all` (the default)
does both in one process, for tests and manual runs.
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

import backends                       # noqa: E402
import oracles                       # noqa: E402
from scenarios import BY_ID, brief   # noqa: E402

MAX_STEPS = int(os.environ.get("RRSI_SIM_MAX_STEPS", "6"))
CONCURRENCY = int(os.environ.get("RRSI_SIM_CONCURRENCY", "4"))


def _load_harness():
    """The evolvable agent, imported only by the propose phase."""
    from harness.agent import run_agent
    return run_agent


RECEIPT_FORMAT = "rrsi-sim-receipt-v1"


def _scrub(tdir: Path) -> None:
    """Evidence cannot exist before the grade phase; remove anything the
    agent left where the grader writes."""
    for name in ("verdict.json", "receipt.json"):
        (tdir / name).unlink(missing_ok=True)
    shutil.rmtree(tdir / "episode", ignore_errors=True)


def _verdict_is_grader_written(tdir: Path) -> bool:
    """A verdict counts only with a receipt this grader wrote that still binds
    the verdict and the episode on disk. The propose phase writes neither, so a
    harness that drops a verdict.json (its own trial or a sibling's) leaves no
    matching receipt; and it cannot forge the receipt's evidence hashes without
    producing the real episode, which needs the hidden grader and its salt.
    Without that binding the artifacts are discarded and the trial is graded
    again, so forged evidence never survives (README "Held-out physics")."""
    receipt = read(tdir / "receipt.json")
    if not (receipt and receipt.get("format") == RECEIPT_FORMAT):
        return False
    if not (tdir / "verdict.json").is_file():
        return False
    if sha256_file(tdir / "verdict.json") != receipt.get("verdict_sha256"):
        return False
    episode = tdir / "episode"
    for name, want in (receipt.get("evidence_sha256") or {}).items():
        if sha256_file(episode / name) != want:
            return False
    return bool(receipt.get("evidence_sha256"))


def _drop_unverified_evidence(runs: Path, job: str, work: list) -> None:
    """At the start of every grade phase, discard any verdict, receipt or
    episode that this grader did not write. The grade phase is the sole author
    of those three; keeping a consistent set makes resume cheap, and scrubbing
    everything else closes the cross-trial planting hole."""
    for tid, i in work:
        tdir = runs / "jobs" / job / tid / f"t{i}"
        if not tdir.is_dir():
            continue
        if not _verdict_is_grader_written(tdir):
            _scrub(tdir)


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


def make_tools(task: dict, surrogate: backends.Backend, scratch: Path):
    counter = {"n": 0}

    def check(controller) -> dict:
        ok, error = backends.check(controller)
        return {"ok": ok, "error": error}

    def rollout(controller) -> dict:
        counter["n"] += 1
        out = scratch / f"rollout-{counter['n']}"
        shutil.rmtree(out, ignore_errors=True)
        job = {"id": "rollout", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
               "controller": controller}
        try:
            surrogate.execute([job], out, f"rollout-{counter['n']}")
        except backends.InfraError as error:
            return {"error": f"surrogate unavailable: {error}"}
        verdict = oracles.grade(out / "rollout", task)
        verdict["backend"] = surrogate.name
        return verdict

    return check, rollout


def propose(runs: Path, job: str, tid: str, i: int, policy, surrogate, run_agent=None) -> str:
    tdir = runs / "jobs" / job / tid / f"t{i}"
    meta = read(tdir / "meta.json")
    if meta and meta.get("status") in ("ok", "crash"):
        return "kept"
    tdir.mkdir(parents=True, exist_ok=True)
    task = BY_ID[tid]
    check, rollout = make_tools(task, surrogate, tdir / "rollouts")
    run_agent = run_agent or _load_harness()
    t0 = time.time()
    meta = {"task": tid, "trial": i, "max_steps": MAX_STEPS, "model": policy.MODEL}
    try:
        out = run_agent(brief(task), policy.chat, check, rollout, MAX_STEPS)
        controller = out.get("controller")
        msgs = out.get("messages") or []
        meta.update(status="ok", tokens=int(out.get("tokens") or 0),
                    steps=sum(1 for m in msgs if m.get("role") == "assistant"))
    except (policy.PolicyError, backends.InfraError) as e:
        _scrub(tdir)
        meta.update(status="infra", error=str(e)[:500], seconds=round(time.time() - t0, 1))
        write(tdir / "meta.json", meta)
        return "infra"
    except Exception:  # noqa: BLE001 - a harness crash is the harness's failure
        controller, msgs = None, []
        meta.update(status="crash", error=traceback.format_exc()[-2000:], tokens=0, steps=0)
    _scrub(tdir)
    meta["seconds"] = round(time.time() - t0, 1)
    write(tdir / "traj.json", {"messages": msgs})
    if controller is not None:
        write(tdir / "controller.json", controller)
    write(tdir / "meta.json", meta)
    return "proposed"


def _batch_id(job: str) -> str:
    stem = re.sub(r"[^a-z0-9-]+", "-", job.lower()).strip("-") or "job"
    return f"{stem[:40]}-{time.strftime('%Y%m%dt%H%M%S')}-{os.getpid() % 10000}"


def execute(runs: Path, job: str, pending: list[tuple[str, int]], backend: backends.Backend) -> dict:
    """One backend batch for every pending trial; moves each episode into its trial."""
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
        facts = backend.execute(jobs, out, batch_id)
    except (backends.InfraError, OSError) as error:
        facts = {"backend": backend.name, "error": str(error)}
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


def grade_trial(tdir: Path, task: dict, backend_name: str) -> str:
    meta = read(tdir / "meta.json") or {}
    if _verdict_is_grader_written(tdir):
        return "kept"
    # A verdict without the grader's own binding receipt is forged; discard it
    # (the episode is re-graded below), so a harness cannot plant a result.
    (tdir / "verdict.json").unlink(missing_ok=True)
    if meta.get("status") == "infra":
        return "infra"
    if meta.get("status") == "crash" or not (tdir / "controller.json").is_file():
        verdict = {"status": "no_submission", "reward": 0.0,
                   "oracles": {name: False for name in oracles.ORACLES}, "simulation_seconds": 0.0}
    else:
        verdict = oracles.grade(tdir / "episode", task)
        if verdict["status"] == "infra":
            return "infra"
    verdict["backend"] = backend_name
    write(tdir / "verdict.json", verdict)
    episode = tdir / "episode"
    write(tdir / "receipt.json", {
        "format": RECEIPT_FORMAT, "task": task["id"], "backend": backend_name,
        "batch": meta.get("batch", ""), "scenario_sha256": sha256_json(task["scenario"]),
        "goal_metres": task["goal_metres"],
        "controller_sha256": sha256_file(tdir / "controller.json"),
        "csf_controller_hash": (read(episode / "manifest.json") or {}).get("controller_hash", ""),
        "evidence_sha256": {name: sha256_file(episode / name)
                            for name in ("manifest.json", "events.jsonl", "trace.jsonl")},
        "verdict_sha256": sha256_file(tdir / "verdict.json")})
    return "passed" if verdict.get("reward") == 1.0 else "graded"


def propose_all(runs: Path, job: str, work: list[tuple[str, int]]) -> dict:
    """The agent's phase: policy, harness and rollout preview only."""
    surrogate = backends.get(backends.surrogate_name())
    policy = _load_policy()
    run_agent = _load_harness()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        proposed = list(ex.map(lambda w: propose(runs, job, *w, policy, surrogate, run_agent), work))
    return {k: proposed.count(k) for k in sorted(set(proposed))}


def grade_all(runs: Path, job: str, work: list[tuple[str, int]], backend: backends.Backend) -> str:
    """The grader's phase: execute and grade, with no policy and no harness."""
    _drop_unverified_evidence(runs, job, work)
    pending = [(t, i) for t, i in work
               if (runs / "jobs" / job / t / f"t{i}" / "controller.json").is_file()
               and (read(runs / "jobs" / job / t / f"t{i}" / "meta.json") or {}).get("status") == "ok"
               and not (runs / "jobs" / job / t / f"t{i}" / "verdict.json").is_file()
               and not (runs / "jobs" / job / t / f"t{i}" / "episode" / "manifest.json").is_file()]
    facts = execute(runs, job, pending, backend)
    graded = [grade_trial(runs / "jobs" / job / t / f"t{i}", BY_ID[t], backend.name) for t, i in work]
    counts = {k: graded.count(k) for k in sorted(set(graded))}
    return (f"backend={backend.name} executed={len(pending)} grades={counts} batch={facts.get('batch', '-')} "
            f"vram={facts.get('vram', {}).get('peak_mib', '-')}")


def main() -> int:
    # backends captured the salt path at import; harness code never finds it
    # in this process's environment.
    os.environ.pop("RRSI_SIM_PHYS_SALT_FILE", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--backend", default=None, help="override RRSI_SIM_BACKEND (smoke uses the surrogate)")
    ap.add_argument("--phase", choices=("propose", "grade", "all"), default="all")
    a = ap.parse_args()
    runs = Path(a.runs)
    work = [(t, i) for t in a.ids.split(",") if t for i in range(a.n)]
    # A misconfigured graded backend fails before any policy call.
    backend = backends.get(a.backend) if a.phase in ("grade", "all") else None
    line = f"[sim] job={a.job} phase={a.phase}"
    if a.phase in ("propose", "all"):
        line += f" proposals={propose_all(runs, a.job, work)}"
    if backend is not None:
        line += " " + grade_all(runs, a.job, work, backend)
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
