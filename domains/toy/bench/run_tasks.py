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
"""Evaluate the toy harness checked out next to this file: k trials per task.

    python3 bench/run_tasks.py --runs RUNS --job JOB --ids a,b,c --n K

Trial i of task X is RUNS/jobs/JOB/X/t<i>/{traj.json, meta.json,
solution.py, verdict.json}. Resume-safe: a trial with a verdict is kept. A
trial whose policy endpoint failed is recorded as status=infra and retried on
the next invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOY))
sys.path.insert(0, str(TOY / "data"))

import policy                     # noqa: E402
import sandbox                    # noqa: E402
from harness.agent import run_agent  # noqa: E402
from tasks import BY_ID           # noqa: E402

MAX_STEPS = int(os.environ.get("RRSI_TOY_MAX_STEPS", "8"))
CONCURRENCY = int(os.environ.get("RRSI_TOY_CONCURRENCY", "8"))
GRADE_TIMEOUT = 20


def grade(task: dict, code: str | None) -> dict:
    if not code or not str(code).strip():
        return {"passed": False, "status": "no_submission", "output": ""}
    res = sandbox.run_python(str(code) + "\n\n# ---- hidden tests ----\n" + task["tests"],
                             timeout=GRADE_TIMEOUT)
    return {"passed": res["ok"], "status": "timeout" if res["timeout"] else "ok",
            "output": (res["stdout"] + res["stderr"])[-3000:]}


def one(runs: Path, job: str, tid: str, i: int) -> str:
    tdir = runs / "jobs" / job / tid / f"t{i}"
    if (tdir / "verdict.json").is_file():
        return "kept"
    tdir.mkdir(parents=True, exist_ok=True)
    task = BY_ID[tid]
    t0 = time.time()
    meta = {"task": tid, "trial": i, "max_steps": MAX_STEPS, "model": policy.MODEL}
    try:
        out = run_agent(task["prompt"], task["entry"], policy.chat, sandbox.run_python,
                        MAX_STEPS)
        code = out.get("code")
        msgs = out.get("messages") or []
        meta.update(status="ok", tokens=int(out.get("tokens") or 0),
                    steps=sum(1 for m in msgs if m.get("role") == "assistant"))
    except policy.PolicyError as e:
        meta.update(status="infra", error=str(e)[:500], seconds=round(time.time() - t0, 1))
        (tdir / "meta.json").write_text(json.dumps(meta, indent=1))
        return "infra"
    except Exception:  # noqa: BLE001 - a harness crash is the harness's failure
        code, msgs = None, []
        meta.update(status="crash", error=traceback.format_exc()[-2000:], tokens=0, steps=0)
    meta["seconds"] = round(time.time() - t0, 1)
    (tdir / "traj.json").write_text(json.dumps({"messages": msgs}, indent=1))
    (tdir / "solution.py").write_text(str(code) if code else "")
    (tdir / "meta.json").write_text(json.dumps(meta, indent=1))
    verdict = grade(task, code)
    (tdir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    return "passed" if verdict["passed"] else "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    work = [(t, i) for t in a.ids.split(",") if t for i in range(a.n)]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        res = list(ex.map(lambda w: one(runs, a.job, *w), work))
    counts = {k: res.count(k) for k in sorted(set(res))}
    print(f"[toy] job={a.job} {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
