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
"""Render one simulation trial into the text the analyst, digester and proposer
read: the scenario brief, the conversation step by step, the submitted
controller, and the graded episode (oracles plus a sampled trace). The oracle
source and the held-out scenarios are never rendered."""

from __future__ import annotations

import json
from pathlib import Path

CAPS = {"asst": 1500, "user": 1200}
DETAIL_CAPS = {"asst": 6000, "user": 4000}
TRACE_ROWS = 12


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _trace(episode: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()]
    except (OSError, ValueError):
        return []
    if len(rows) <= TRACE_ROWS:
        return rows
    stride = max(1, len(rows) // (TRACE_ROWS - 1))
    picked = rows[::stride]
    return picked if picked[-1] is rows[-1] else picked + [rows[-1]]


def load_trial(job_dir: Path, task: dict, brief: str, trial: int):
    tdir = job_dir / task["id"] / f"t{trial}"
    meta = _read(tdir / "meta.json")
    if meta is None:
        return None
    return {"task": {"id": task["id"], "brief": brief}, "trial_dir": str(tdir), "meta": meta,
            "traj": _read(tdir / "traj.json") or {"messages": []},
            "controller": _read(tdir / "controller.json"),
            "verdict": _read(tdir / "verdict.json") or {},
            "manifest": _read(tdir / "episode" / "manifest.json") or {},
            "trace": _trace(tdir / "episode")}


def failure_class(rec: dict) -> str:
    meta, v = rec.get("meta") or {}, rec.get("verdict") or {}
    if meta.get("status") == "crash":
        return "HARNESS-CRASH"
    if meta.get("status") == "infra" or v.get("status") == "infra" or not v:
        return "INFRA -- not evidence about the harness"
    if v.get("status") == "no_submission":
        return "NO-SUBMISSION(step budget spent or never submitted)"
    if v.get("status") == "rejected":
        return "REJECTED(controller failed CSF admission)"
    o = v.get("oracles") or {}
    if all(o.values()):
        return "passed"
    if not o.get("no_collision"):
        return "LANE-DEPARTURE(left the lane or collided)"
    if not o.get("reached_goal"):
        return "GOAL-NOT-REACHED(too slow or stopped)"
    if not o.get("within_limits"):
        return "OVER-LIMIT(speed limit or runtime fallback)"
    return "NOT-SETTLED(offset or oscillation at the end)"


def render_full(rec: dict, detail: bool = False) -> str:
    caps = DETAIL_CAPS if detail else CAPS
    task, meta, v = rec["task"], rec.get("meta") or {}, rec.get("verdict") or {}
    lines = [f"=== TASK {task['id']} ===", task["brief"], "=== TRAJECTORY ==="]
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
    lines += ["=== SUBMITTED CONTROLLER ===",
              _clip(json.dumps(rec.get("controller")) if rec.get("controller") is not None else "(none)", 3000),
              "=== SIMULATION (graded backend) ===",
              f"FAILURE CLASS: {failure_class(rec)}",
              f"backend={v.get('backend')} reward={v.get('reward')} oracles={json.dumps(v.get('oracles'))}",
              f"termination={v.get('termination')} steps={v.get('steps')} sim_seconds={v.get('simulation_seconds')} "
              f"max|lateral|={v.get('max_abs_lateral_metres')} max_speed={v.get('max_speed_mps')} "
              f"final_lateral={v.get('final_lateral_metres')} final_heading={v.get('final_heading_error_radians')}",
              f"reason={_clip(v.get('reason') or '', 600)}",
              "sampled trace (step: longitudinal m, lateral m, heading rad, speed m/s | steering, accel):"]
    for row in rec.get("trace") or []:
        s, a = row.get("state") or {}, row.get("action") or {}
        lines.append(f"  {row.get('step')}: {s.get('longitudinal_metres', 0):.1f}, {s.get('lateral_metres', 0):+.2f}, "
                     f"{s.get('heading_error_radians', 0):+.3f}, {s.get('speed_mps', 0):.1f} | "
                     f"{a.get('steering')}, {a.get('acceleration')}")
    return "\n".join(lines)
