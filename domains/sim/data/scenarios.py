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
"""The simulation suite: straight-path driving scenarios, graded by oracles.

CSF is Candace Labs' Go framework for AI-agent systems (developer preview; see
domains/sim/README.md). Every task is one CSF `Scenario` (generated protobuf
JSON, schema_version 1) in
the straight-path, no-fault profile that CSF's scenario projector accepts for
CARLA and Isaac today, plus a goal distance. Seeds realize the target speed,
episode length, initial lateral offset and heading. PRACTICE (the evolve set)
and HELDOUT are disjoint seed ranges drawn from the same distribution, so a
held-out score measures transfer to unseen scenarios, not a harder exam.

The agent sees a plain-language brief (`brief`); the grader (bench/oracles.py)
sees the same scenario plus the simulator's events.
"""

from __future__ import annotations

import random

PROFILE = {"lane_half_width_metres": 2.0, "tick_milliseconds": 100, "curvature": 0.0}
GOAL_FRACTION = 0.75       # goal = 75% of the distance at target speed for the whole episode
SPEED_LIMIT_FACTOR = 1.25  # oracle: never faster than 1.25 x target speed
# 24 practice and 8 held-out scenarios: with k = 3 that is 72 graded episodes
# per evaluation, which is what keeps the measured noise band near 0.1 when
# the policy's sampling cannot be made deterministic (see README "Noise band").
PRACTICE_SEEDS = tuple(range(100, 124))
HELDOUT_SEEDS = tuple(range(300, 308))


def make(seed: int) -> dict:
    rng = random.Random(seed)
    target = round(rng.uniform(5.0, 18.0), 1)
    seconds = rng.choice((10, 12, 14, 16, 18, 20, 24))
    scenario = {
        "schema_version": 1,
        "name": f"straight-{seed}",
        "seed": str(seed),
        "steps": seconds * 1000 // PROFILE["tick_milliseconds"],
        "tick_milliseconds": PROFILE["tick_milliseconds"],
        "target_speed_mps": target,
        "lane_half_width_metres": PROFILE["lane_half_width_metres"],
        "initial_lateral_metres": round(rng.uniform(-1.6, 1.6), 2),
        "initial_heading_radians": round(rng.uniform(-0.15, 0.15), 3),
    }
    return {"id": scenario["name"], "seed": seed, "scenario": scenario,
            "goal_metres": round(GOAL_FRACTION * target * seconds, 1),
            "speed_limit_mps": round(SPEED_LIMIT_FACTOR * target, 2)}


def brief(task: dict) -> str:
    """What the agent is told: the scenario in physical units, never the grader."""
    s = task["scenario"]
    seconds = s["steps"] * s["tick_milliseconds"] / 1000
    side = "left" if s["initial_lateral_metres"] > 0 else "right"
    turn = "left" if s["initial_heading_radians"] > 0 else "right"
    return (
        f"Straight road, lane half width {s['lane_half_width_metres']} m. "
        f"Target speed {s['target_speed_mps']} m/s; the vehicle starts at 60% of it, "
        f"{abs(s['initial_lateral_metres'])} m {side} of the lane centre, heading "
        f"{abs(s['initial_heading_radians'])} rad to the {turn} of the road. "
        f"Reach {task['goal_metres']} m along the road within {seconds:g} s "
        f"({s['steps']} control ticks of {s['tick_milliseconds']} ms), stay in the lane, "
        f"never exceed {task['speed_limit_mps']} m/s, and finish centred and aligned with the lane.")


PRACTICE = [make(seed) for seed in PRACTICE_SEEDS]
HELDOUT = [make(seed) for seed in HELDOUT_SEEDS]
BY_ID = {task["id"]: task for task in PRACTICE + HELDOUT}
EVOLVE = [task["id"] for task in PRACTICE]
HELDOUT_IDS = [task["id"] for task in HELDOUT]
