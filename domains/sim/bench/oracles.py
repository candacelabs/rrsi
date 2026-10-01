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
"""The scenario oracles: grade one executed episode from its CSF evidence.

Input is the episode directory a CSF scenario batch writes: `manifest.json`
(terminal status and why the episode ended) and `events.jsonl` (generated
ResearchEvent rows, one measurement per metric per physics step). Four oracles,
each pass/fail, and the reward is the fraction passed:

  reached_goal   travelled goal_metres along the road within the step budget
  no_collision   no collision-sensor event and never left the lane (on this
                 empty road, leaving the lane is the only contact available)
  within_limits  never faster than the scenario's speed limit and no runtime
                 fallback (a stale, replayed or wrong-epoch observation)
  settled        over the last quarter of the episode the mean |lateral offset|
                 is at most 0.3 m, and the final |heading error| at most 0.05 rad

A controller the runtime rejected scores 0 on every oracle. An episode whose
simulator or runtime transport failed has status "infra" and no reward: it is
never evidence about the controller.
"""

from __future__ import annotations

import json
from pathlib import Path

ORACLES = ("reached_goal", "no_collision", "within_limits", "settled")
SETTLED_LATERAL_METRES = 0.3
SETTLED_HEADING_RADIANS = 0.05


def series(events_path: Path) -> dict:
    """metric -> [value by step], from generated ResearchEvent JSON rows."""
    out: dict = {}
    for line in events_path.read_text().splitlines():
        row = json.loads(line)
        m = row.get("measurement")
        if m:
            out.setdefault(m["metric"], []).append((int(m.get("step", 0)), float(m.get("value", 0.0))))
    return {k: [v for _, v in sorted(rows)] for k, rows in out.items()}


def grade(episode: Path, task: dict) -> dict:
    try:
        manifest = json.loads((episode / "manifest.json").read_text())
    except (OSError, ValueError):
        return {"status": "infra", "reason": "no episode manifest"}
    status = manifest.get("status")
    if status == "infra" or status not in ("completed", "rejected"):
        return {"status": "infra", "reason": manifest.get("reason") or f"episode status {status!r}"}
    if status == "rejected":
        return {"status": "rejected", "reason": manifest.get("reason", ""), "reward": 0.0,
                "oracles": {name: False for name in ORACLES}, "simulation_seconds": 0.0, "steps": 0}
    data = series(episode / "events.jsonl")
    longitudinal = data.get("scenario_longitudinal_metres") or [0.0]
    lateral = data.get("scenario_lateral_metres") or [0.0]
    heading = data.get("scenario_heading_error_radians") or [0.0]
    speed = data.get("simulator_speed_mps") or [0.0]
    collisions = data.get("scenario_collisions") or [0.0]
    fallbacks = data.get("runtime_fallbacks") or [0.0]
    half = float(task["scenario"]["lane_half_width_metres"])
    tail = lateral[-max(1, len(lateral) // 4):]
    departed = max(abs(v) for v in lateral) > half
    oracles = {
        "reached_goal": max(longitudinal) >= float(task["goal_metres"]),
        "no_collision": max(collisions) == 0 and not departed,
        "within_limits": max(speed) <= float(task["speed_limit_mps"]) and max(fallbacks) == 0,
        "settled": (not departed and sum(abs(v) for v in tail) / len(tail) <= SETTLED_LATERAL_METRES
                    and abs(heading[-1]) <= SETTLED_HEADING_RADIANS),
    }
    return {"status": "ok", "reward": sum(oracles.values()) / len(ORACLES), "oracles": oracles,
            "termination": manifest.get("termination", ""),
            "simulation_seconds": float(manifest.get("simulation_seconds") or 0.0),
            "steps": int(manifest.get("steps_completed") or 0),
            "max_abs_lateral_metres": max(abs(v) for v in lateral),
            "max_speed_mps": max(speed), "final_longitudinal_metres": longitudinal[-1],
            "final_lateral_metres": lateral[-1], "final_heading_error_radians": heading[-1],
            "final_speed_mps": speed[-1]}
