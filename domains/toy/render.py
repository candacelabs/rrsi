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
"""Render one toy trial into the text the analyst, digester and proposer read:
the conversation step by step, the submitted module and the grader's verdict
(hidden-test output tail). The hidden test source is never rendered."""

from __future__ import annotations

import json
from pathlib import Path

CAPS = {"asst": 1500, "user": 1200}
DETAIL_CAPS = {"asst": 6000, "user": 4000}


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def load_trial(job_dir: Path, task: dict, trial: int):
    tdir = job_dir / task["id"] / f"t{trial}"
    meta = _read(tdir / "meta.json")
    if meta is None:
        return None
    sol = tdir / "solution.py"
    return {"task": {"id": task["id"], "prompt": task["prompt"], "entry": task["entry"]},
            "trial_dir": str(tdir), "meta": meta,
            "traj": _read(tdir / "traj.json") or {"messages": []},
            "verdict": _read(tdir / "verdict.json") or {},
            "solution": sol.read_text() if sol.exists() else ""}


def failure_class(rec: dict) -> str:
    meta, v = rec.get("meta") or {}, rec.get("verdict") or {}
    if meta.get("status") == "crash":
        return "HARNESS-CRASH"
    if meta.get("status") == "infra":
        return "INFRA -- not evidence about the harness"
    if v.get("passed"):
        return "passed"
    if v.get("status") == "no_submission":
        return "NO-SUBMISSION(step budget spent or never submitted)"
    if v.get("status") == "timeout":
        return "TIMEOUT(submitted code too slow or hung)"
    out = str(v.get("output") or "")
    if "SyntaxError" in out or "NameError" in out or "IndentationError" in out:
        return "BROKEN-MODULE(submitted code does not import or lacks the function)"
    if "RecursionError" in out:
        return "RECURSION-LIMIT(deep input)"
    return "WRONG-ANSWER(hidden edge case failed)"


def render_full(rec: dict, detail: bool = False) -> str:
    caps = DETAIL_CAPS if detail else CAPS
    task, meta, v = rec["task"], rec.get("meta") or {}, rec.get("verdict") or {}
    lines = [f"=== TASK {task['id']} (entry `{task['entry']}`) ===", task["prompt"],
             "=== TRAJECTORY ==="]
    step = 0
    for m in (rec.get("traj") or {}).get("messages") or []:
        role = m.get("role")
        if role == "system":
            lines.append(f"[system] {_clip(m.get('content'), caps['user'])}")
        elif role == "assistant":
            step += 1
            lines.append(f"[step {step}] AGENT: {_clip(m.get('content'), caps['asst'])}")
        else:
            lines.append(f"[step {step}] ENV: {_clip(m.get('content'), caps['user'])}")
    if meta.get("error"):
        lines.append(f"[harness error] {_clip(meta['error'], 2000)}")
    lines += ["=== SUBMITTED MODULE ===", _clip(rec.get("solution") or "(none)", 4000),
              "=== GRADING (hidden tests) ===",
              f"FAILURE CLASS: {failure_class(rec)}",
              f"passed={v.get('passed')} status={v.get('status')} "
              f"steps={meta.get('steps')} tokens={meta.get('tokens')}",
              "grader output tail:", _clip(v.get("output") or "(empty)", 2500)]
    return "\n".join(lines)
