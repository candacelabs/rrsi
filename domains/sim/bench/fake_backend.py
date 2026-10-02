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
"""The fake backend: a dependency-free stand-in for a CSF scenario batch.

(CSF: Candace Labs' Go framework for AI-agent systems, developer preview; see
domains/sim/README.md.)

It exists so the domain's plumbing (propose -> check -> execute -> grade) runs
in CI with no Go runtime, no HighwayEnv and no GPU. It mirrors CSF's numeric
profile v1 contract: the controller check follows csf/compiler.go's
admission rules (a JSON null means the field's default, as in ProtoJSON),
evaluation uses the same bounded integer arithmetic (SCALE truncates toward
zero), and the plant is HighwayEnv's kinematic bicycle, line for line. That
bicycle is the only model the agent's rollout preview uses.

It writes the same files a CSF scenario_worker batch writes (batch.json and,
per episode, events.jsonl / trace.jsonl / manifest.json). It is not evidence
about any real simulator; `tests/test_sim_domain.py` compares its controller
semantics with the Go runtime whenever RRSI_SIM_CSF_RUNTIME is set.

`run_episode` takes a plant factory and a controller stepper so a grader-side
backend can reuse the episode loop and evidence format with a different
vehicle and with the Go runtime evaluating the controller; the defaults are
the kinematic bicycle and the in-process mirror.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCALE = 1000
VALUE_LIMIT = 1_000_000_000
COEFFICIENT_LIMIT = 1_000_000
FEATURE_LIMIT = 10_000
MAX_INSTRUCTIONS = 128
MAX_DEPTH = 16
OPCODES = {"OPCODE_CONSTANT": 0, "OPCODE_INPUT": 0, "OPCODE_ADD": 2, "OPCODE_SCALE": 1, "OPCODE_CLAMP": 1}
EXPRESSION_FIELDS = {"opcode", "value", "input_index", "lower", "upper", "arguments"}
CONTROLLER_FIELDS = {"schema_version", "name", "steering", "acceleration"}
METRICS = ("simulation_steps_completed", "scenario_longitudinal_metres", "scenario_lateral_metres",
           "scenario_heading_error_radians", "simulator_speed_mps", "scenario_collisions",
           "runtime_fallbacks")


class Rejected(ValueError):
    """The controller fails admission (a controller result, not infrastructure)."""


def _int(value, name: str) -> int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise Rejected(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise Rejected(f"{name} must be an integer") from error


def _compile(expression, depth: int, budget: list) -> None:
    if not isinstance(expression, dict) or depth >= MAX_DEPTH or budget[0] >= MAX_INSTRUCTIONS:
        raise Rejected("missing expression or expression budget exceeded")
    unknown = set(expression) - EXPRESSION_FIELDS
    if unknown:
        raise Rejected(f"unknown expression fields {sorted(unknown)}")
    opcode = expression.get("opcode", "OPCODE_UNSPECIFIED")
    if opcode not in OPCODES:
        raise Rejected(f"unsupported opcode {opcode}")
    value = _int(expression.get("value", 0), "value")
    lower = _int(expression.get("lower", 0), "lower")
    upper = _int(expression.get("upper", 0), "upper")
    index = _int(expression.get("input_index", 0), "input_index")
    if any(abs(v) > VALUE_LIMIT for v in (value, lower, upper)):
        raise Rejected("operand outside numeric profile")
    if opcode != "OPCODE_INPUT" and index != 0:
        raise Rejected("unexpected input index")
    if opcode not in ("OPCODE_CONSTANT", "OPCODE_SCALE") and value != 0:
        raise Rejected("unexpected value")
    if opcode != "OPCODE_CLAMP" and (lower or upper):
        raise Rejected("unexpected clamp bounds")
    if opcode == "OPCODE_INPUT" and not 0 <= index < 4:
        raise Rejected("input index outside numeric profile")
    if opcode == "OPCODE_SCALE" and abs(value) > COEFFICIENT_LIMIT:
        raise Rejected("coefficient outside numeric profile")
    if opcode == "OPCODE_CLAMP" and lower > upper:
        raise Rejected("inverted clamp bounds")
    arguments = expression.get("arguments") or []
    if not isinstance(arguments, list) or len(arguments) != OPCODES[opcode]:
        raise Rejected(f"opcode {opcode} needs {OPCODES[opcode]} arguments")
    for argument in arguments:
        _compile(argument, depth + 1, budget)
    if budget[0] >= MAX_INSTRUCTIONS:
        raise Rejected("instruction budget exceeded")
    budget[0] += 1


def normalize(value):
    """A copy without the object keys whose value is null (lists are kept).

    ProtoJSON reads a null field as its default, so the Go runtime admits
    {"value": null}; the mirror must read it the same way."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def check(controller) -> None:
    """Raise Rejected unless CSF would admit this Controller JSON."""
    controller = normalize(controller)
    if not isinstance(controller, dict):
        raise Rejected("controller must be a JSON object")
    unknown = set(controller) - CONTROLLER_FIELDS
    if unknown:
        raise Rejected(f"unknown controller fields {sorted(unknown)}")
    if _int(controller.get("schema_version", 0), "schema_version") != 1:
        raise Rejected("schema_version must be 1")
    name = controller.get("name")
    if not isinstance(name, str) or not 0 < len(name) <= 128:
        raise Rejected("name must be 1..128 characters")
    for output in ("steering", "acceleration"):
        try:
            _compile(controller.get(output), 0, [0])
        except Rejected as error:
            raise Rejected(f"{output}: {error}") from error


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, value))


def _evaluate(expression: dict, features: list[int]) -> int:
    opcode = expression["opcode"]
    arguments = [_evaluate(a, features) for a in expression.get("arguments") or []]
    if opcode == "OPCODE_CONSTANT":
        result = int(expression.get("value", 0))
    elif opcode == "OPCODE_INPUT":
        result = features[int(expression.get("input_index", 0))]
    elif opcode == "OPCODE_ADD":
        result = arguments[0] + arguments[1]
    elif opcode == "OPCODE_SCALE":
        product = arguments[0] * int(expression.get("value", 0))
        result = int(math.copysign(abs(product) // SCALE, product)) if product else 0
    else:
        result = _clamp(arguments[0], int(expression.get("lower", 0)), int(expression.get("upper", 0)))
    return _clamp(result, -VALUE_LIMIT, VALUE_LIMIT)


def act(controller: dict, features: list[int]) -> tuple[int, int]:
    """The action of an admitted, normalized controller."""
    return (_clamp(_evaluate(controller["steering"], features), -SCALE, SCALE),
            _clamp(_evaluate(controller["acceleration"], features), -SCALE, SCALE))


def fixed_point(value: float) -> int:
    return max(-FEATURE_LIMIT, min(FEATURE_LIMIT, round(value * 1000)))


class Bicycle:
    """Kinematic bicycle (HighwayEnv's model: 5 m wheelbase, slip angle beta)."""

    LENGTH = 5.0
    MAX_SPEED = 40.0

    def __init__(self, scenario: dict):
        self.x, self.y = 0.0, float(scenario.get("initial_lateral_metres", 0.0))
        self.heading = float(scenario.get("initial_heading_radians", 0.0))
        self.speed = 0.6 * float(scenario["target_speed_mps"])

    def step(self, steering: float, acceleration: float, dt: float) -> None:
        beta = math.atan(0.5 * math.tan(steering))
        self.x += self.speed * math.cos(self.heading + beta) * dt
        self.y += self.speed * math.sin(self.heading + beta) * dt
        self.heading += self.speed * math.sin(beta) / (self.LENGTH / 2) * dt
        self.speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, self.speed + acceleration * dt))

    def state(self) -> dict:
        error = math.atan2(math.sin(self.heading), math.cos(self.heading))
        return {"longitudinal_metres": self.x, "lateral_metres": self.y,
                "heading_error_radians": error, "speed_mps": self.speed, "collisions": 0}

    def observe(self) -> dict:
        """What the controller sees: the exact state."""
        return self.state()


def mirror_stepper(controller: dict):
    """features -> (steering, acceleration) by the in-process mirror; raises
    Rejected for a controller CSF would not admit."""
    check(controller)
    return lambda features: act(controller, features)


def _event(run_id: str, **payload) -> str:
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return json.dumps({"schema_version": 1, "recorded_at": stamp, **payload}, sort_keys=True)


def run_episode(job: dict, output: Path, run_id: str, plant_factory=None, stepper_factory=None,
                simulator: dict | None = None) -> dict:
    """One episode into `output`. plant_factory(scenario) builds the vehicle
    (default: the kinematic Bicycle); stepper_factory(controller) returns
    features -> (steering, acceleration) or raises Rejected (default: the
    mirror); `simulator` is what the manifest records (default: describe())."""
    scenario, goal = job["scenario"], float(job["goal_metres"])
    controller = normalize(job.get("controller"))
    output.mkdir(parents=True, exist_ok=False)
    summary = {"id": job["id"], "status": "rejected", "steps_completed": 0,
               "simulation_seconds": 0.0, "termination": "", "reason": ""}
    events = [_event(run_id, definition={"name": name}) for name in METRICS]
    trace = []
    try:
        step = (stepper_factory or mirror_stepper)(controller)
    except Rejected as error:
        summary["reason"] = f"controller rejected: {error}"
    else:
        plant = (plant_factory or Bicycle)(scenario)
        dt, half = scenario["tick_milliseconds"] / 1000, scenario["lane_half_width_metres"]
        target, fallbacks, termination = scenario["target_speed_mps"], 0, "step_budget"
        for tick in range(int(scenario["steps"])):
            s = plant.observe()
            features = [fixed_point(s["lateral_metres"] / half), fixed_point(s["heading_error_radians"] / 0.5),
                        fixed_point((target - s["speed_mps"]) / 10), 0]
            steering, acceleration = step(features)
            plant.step(steering / 1000 * 0.5, acceleration / 1000 * 3, dt)
            after, completed = plant.state(), tick + 1
            trace.append(json.dumps({"step": completed, "simulation_seconds": completed * dt,
                                     "observation": features, "state": after,
                                     "action": {"steering": steering, "acceleration": acceleration,
                                                "fallback": False}}, sort_keys=True))
            for name, value in (("simulation_steps_completed", completed),
                                ("scenario_longitudinal_metres", after["longitudinal_metres"]),
                                ("scenario_lateral_metres", after["lateral_metres"]),
                                ("scenario_heading_error_radians", after["heading_error_radians"]),
                                ("simulator_speed_mps", after["speed_mps"]),
                                ("scenario_collisions", 0), ("runtime_fallbacks", fallbacks)):
                events.append(_event(run_id, measurement={"metric": name, "value": value, "step": completed,
                                                          "run_id": run_id, "candidate_id": "fake",
                                                          "split": "native"}))
            summary["steps_completed"] = completed
            if abs(after["lateral_metres"]) > half:
                termination = "lane_departure"
            elif after["longitudinal_metres"] >= goal:
                termination = "goal_reached"
            if termination != "step_budget":
                break
        summary.update(status="completed", termination=termination,
                       simulation_seconds=summary["steps_completed"] * dt)
    events.append(_event(run_id, status={"run_id": run_id, "phase": summary["status"],
                                         "message": summary["termination"] or summary["reason"]}))
    (output / "events.jsonl").write_text("\n".join(events) + "\n")
    (output / "trace.jsonl").write_text("\n".join(trace) + ("\n" if trace else ""))
    (output / "manifest.json").write_text(json.dumps({
        "format": "csf-scenario-episode-v1", "status": summary["status"], "run_id": run_id,
        "episode": job["id"], "simulator": simulator or describe(), "scenario": scenario, "goal_metres": goal,
        "controller": job.get("controller"), "steps_completed": summary["steps_completed"],
        "simulation_seconds": summary["simulation_seconds"], "termination": summary["termination"],
        "reason": summary["reason"], "numeric_profile": 1}, indent=1, sort_keys=True))
    return summary


def describe() -> dict:
    return {"name": "fake", "version": "1", "gpu": False,
            "note": "dependency-free kinematic stand-in; not evidence about a simulator"}


def run_batch(jobs: list[dict], output: Path, run_id: str, crash_ids: frozenset = frozenset()) -> list[dict]:
    """crash_ids simulate a simulator crash at that episode (tests of the Infra guard)."""
    output.mkdir(parents=True, exist_ok=True)
    results, broken = [], ""
    for job in jobs:
        if broken or job["id"] in crash_ids:
            broken = broken or f"simulated simulator crash at {job['id']}"
            results.append({"id": job["id"], "status": "infra", "reason": broken})
            continue
        results.append(run_episode(job, output / job["id"], run_id))
    (output / "batch.json").write_text(json.dumps({"format": "csf-scenario-batch-v1", "run_id": run_id,
                                                   "simulator": describe(), "episodes": results},
                                                  indent=1, sort_keys=True))
    return results
