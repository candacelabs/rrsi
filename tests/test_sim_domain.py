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
"""Simulation domain: scenarios, CSF admission mirror, oracles, runner, adapter.

    python3 -m pytest -q tests/test_sim_domain.py

Runs on the dependency-free fake backend. With RRSI_SIM_CSF_RUNTIME set to the
CSF Go runtime, the fake controller semantics are also compared with it.
"""

import hashlib
import http.server
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "domains" / "sim"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rrsi.domain import load_domain as _load_domain  # noqa: E402

# Every domain names its modules harness/render/briefs/run_tasks; one pytest
# process holds several domains, so this file imports the sim modules and then
# returns sys.path and sys.modules to their previous state.
_SHARED = ("harness", "render", "briefs", "run_tasks", "backends", "oracles", "fake_backend", "scenarios",
           "heldout_vehicle", "heldout_physics", "phys_worker", "agreement")
_SAVED = {n: m for n, m in sys.modules.items() if n.split(".")[0] in _SHARED}
for _name in _SAVED:
    del sys.modules[_name]
_PATH, _MODULES = list(sys.path), set(sys.modules)
sys.path[:0] = [str(SIM), str(SIM / "bench"), str(SIM / "data"), str(SIM / "grader")]
import agreement                     # noqa: E402
import backends                      # noqa: E402
import fake_backend                  # noqa: E402
import heldout_physics               # noqa: E402
import heldout_vehicle               # noqa: E402
import oracles                      # noqa: E402
import run_tasks                    # noqa: E402
import scenarios                    # noqa: E402
from harness.agent import run_agent  # noqa: E402
sys.path[:] = _PATH
for _name in set(sys.modules) - _MODULES:
    del sys.modules[_name]
sys.modules.update(_SAVED)


def load_domain(name):
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] in _SHARED}
    for shared in saved:
        del sys.modules[shared]
    path, modules = list(sys.path), set(sys.modules)
    try:
        return _load_domain(name)
    finally:
        sys.path[:] = path
        for extra in set(sys.modules) - modules:
            if not extra.startswith("domains."):
                del sys.modules[extra]
        sys.modules.update(saved)


def scale(index, gain):
    return {"opcode": "OPCODE_SCALE", "value": gain,
            "arguments": [{"opcode": "OPCODE_INPUT", "input_index": index}]}


def clamp(expression):
    return {"opcode": "OPCODE_CLAMP", "lower": -1000, "upper": 1000, "arguments": [expression]}


def controller(lateral=-250, heading=-500, speed=500, name="pd"):
    """CSF's training-harness baseline shape: steering = a*in0 + b*in1, accel = c*in2."""
    return {"schema_version": 1, "name": name,
            "steering": clamp({"opcode": "OPCODE_ADD", "arguments": [scale(0, lateral), scale(1, heading)]}),
            "acceleration": clamp(scale(2, speed))}


def test_splits_are_disjoint_and_in_the_straight_profile():
    assert len(scenarios.EVOLVE) == 24 and len(scenarios.HELDOUT_IDS) == 8
    assert not set(scenarios.EVOLVE) & set(scenarios.HELDOUT_IDS)
    for task in scenarios.PRACTICE + scenarios.HELDOUT:
        s = task["scenario"]
        assert s["schema_version"] == 1 and "road_curvature_per_metre" not in s and "faults" not in s
        assert 4 <= s["target_speed_mps"] <= 20 and abs(s["initial_lateral_metres"]) < s["lane_half_width_metres"]
        assert abs(s["initial_heading_radians"]) <= 0.3 and 1 <= task["goal_metres"] <= 2000
        assert str(task["goal_metres"]) in scenarios.brief(task)
    assert scenarios.make(100) == scenarios.make(100), "scenarios must be reproducible from the seed"
    assert len({t["scenario"]["target_speed_mps"] for t in scenarios.PRACTICE}) > 4


def test_admission_mirror_accepts_the_baseline_and_rejects_malformed_trees():
    fake_backend.check(controller())
    bad = [
        {"schema_version": 1, "name": "x", "steering": {"opcode": "OPCODE_ADD", "arguments": []},
         "acceleration": {"opcode": "OPCODE_CONSTANT"}},
        {**controller(), "extra": 1},
        {**controller(), "steering": {"opcode": "OPCODE_INPUT", "input_index": 4}},
        {**controller(), "steering": {"opcode": "OPCODE_ADD", "value": 3,
                                      "arguments": [scale(0, 1), scale(1, 1)]}},
        {**controller(), "acceleration": {"opcode": "OPCODE_CLAMP", "lower": 5, "upper": 1,
                                          "arguments": [scale(2, 1)]}},
        {**controller(), "schema_version": 2},
    ]
    for c in bad:
        with pytest.raises(fake_backend.Rejected):
            fake_backend.check(c)
    assert backends.check(bad[0])[0] is False


def test_scale_truncates_toward_zero_like_the_go_runtime():
    c = {"schema_version": 1, "name": "s", "steering": scale(0, 1), "acceleration": scale(0, -1)}
    assert fake_backend.act(c, [1999, 0, 0, 0]) == (1, -1)
    assert fake_backend.act(c, [-1999, 0, 0, 0]) == (-1, 1)


def test_a_null_field_reads_as_its_default_like_protojson(tmp_path):
    # The Go runtime admits {"value": null}; the mirror once rejected it.
    nulled = controller()
    nulled["steering"]["value"] = None
    nulled["acceleration"]["arguments"][0]["value"] = None
    fake_backend.check(nulled)
    assert fake_backend.normalize(nulled)["steering"] == controller()["steering"]
    zero_gain = controller(speed=0)
    for features in ([500, -300, 200, 0], [-1999, 40, -700, 0]):
        assert fake_backend.act(fake_backend.normalize(nulled), features) == fake_backend.act(zero_gain, features)
    task = scenarios.PRACTICE[0]
    job = {"id": "n", "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": nulled}
    assert fake_backend.run_episode(job, tmp_path / "n", "r")["status"] == "completed"
    manifest = json.loads((tmp_path / "n" / "manifest.json").read_text())
    assert manifest["controller"] == nulled, "the manifest keeps the submitted controller"


def test_the_preview_plant_is_highway_envs_kinematic_bicycle(tmp_path):
    """The fake (preview) plant equals an inline copy of HighwayEnv's
    kinematic Vehicle.step: 5 m length, slip angle beta, one Euler step."""
    template = controller(-300, -600, 800)
    for task in scenarios.PRACTICE[:3]:
        job = {"id": task["id"], "scenario": task["scenario"], "goal_metres": task["goal_metres"],
               "controller": template}
        fake_backend.run_episode(job, tmp_path / task["id"], "r")
        rows = [json.loads(line) for line in (tmp_path / task["id"] / "trace.jsonl").read_text().splitlines()]
        s = task["scenario"]
        x, y, heading = 0.0, s["initial_lateral_metres"], s["initial_heading_radians"]
        speed, dt = 0.6 * s["target_speed_mps"], s["tick_milliseconds"] / 1000
        for row in rows:
            delta = row["action"]["steering"] / 1000 * 0.5
            beta = math.atan(0.5 * math.tan(delta))
            x += speed * math.cos(heading + beta) * dt
            y += speed * math.sin(heading + beta) * dt
            heading += speed * math.sin(beta) / (5.0 / 2) * dt
            speed += row["action"]["acceleration"] / 1000 * 3 * dt
            state = row["state"]
            assert abs(state["longitudinal_metres"] - x) < 1e-12 and abs(state["lateral_metres"] - y) < 1e-12
            assert abs(state["heading_error_radians"] - heading) < 1e-12 and abs(state["speed_mps"] - speed) < 1e-12
        assert len(rows) > 50


def grade_one(c, task=None, tmp=None):
    task = task or scenarios.PRACTICE[0]
    job = {"id": "e", "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": c}
    fake_backend.run_batch([job], tmp, "r")
    return oracles.grade(tmp / "e", task)


def test_oracles_separate_good_wrong_signed_and_rejected_controllers(tmp_path):
    good = grade_one(controller(), tmp=tmp_path / "good")
    assert good["status"] == "ok" and good["oracles"]["reached_goal"] and good["oracles"]["no_collision"]
    wrong = grade_one(controller(lateral=400, heading=600), tmp=tmp_path / "wrong")
    assert wrong["termination"] == "lane_departure" and not wrong["oracles"]["no_collision"]
    assert wrong["reward"] < good["reward"]
    slow = grade_one(controller(speed=0), tmp=tmp_path / "slow")
    assert not slow["oracles"]["reached_goal"]
    rejected = grade_one({"schema_version": 1, "name": "r"}, tmp=tmp_path / "rej")
    assert rejected["status"] == "rejected" and rejected["reward"] == 0.0


def test_events_are_csf_research_event_rows(tmp_path):
    grade_one(controller(), tmp=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "e" / "events.jsonl").read_text().splitlines()]
    measured = {r["measurement"]["metric"] for r in rows if "measurement" in r}
    assert measured == set(fake_backend.METRICS)
    assert all(r["schema_version"] == 1 and r["recorded_at"].endswith("Z") for r in rows)
    manifest = json.loads((tmp_path / "e" / "manifest.json").read_text())
    assert manifest["format"] == "csf-scenario-episode-v1" and manifest["status"] == "completed"


class ScriptedPolicy:
    """Stands in for domains/toy/policy.py: same attributes, scripted replies."""

    MODEL = "scripted"

    class PolicyError(RuntimeError):
        pass

    def __init__(self, replies):
        self.replies = replies

    def chat(self, messages, json_mode=False):
        brief = messages[1]["content"]
        reply = self.replies(brief)
        if isinstance(reply, Exception):
            raise reply
        return reply, 100


def submit(c):
    return json.dumps({"action": "submit", "controller": c})


def run_job(tmp_path, ids, k, policy, backend, job="j"):
    runs = tmp_path / "runs"
    surrogate = backends.FakeBackend()
    work = [(t, i) for t in ids for i in range(k)]
    for t, i in work:
        run_tasks.propose(runs, job, t, i, policy, surrogate, run_agent)
    pending = [(t, i) for t, i in work
               if (runs / "jobs" / job / t / f"t{i}" / "controller.json").is_file()
               and not (runs / "jobs" / job / t / f"t{i}" / "episode" / "manifest.json").is_file()]
    run_tasks.execute(runs, job, pending, backend)
    return runs, [run_tasks.grade_trial(runs / "jobs" / job / t / f"t{i}", scenarios.BY_ID[t], backend.name)
                  for t, i in work]


def test_runner_end_to_end_with_receipts_and_scoring(tmp_path):
    ids = scenarios.EVOLVE[:3]
    skipped = scenarios.brief(scenarios.BY_ID[ids[0]])
    policy = ScriptedPolicy(lambda brief: submit(controller()) if skipped not in brief else "no json")
    runs, grades = run_job(tmp_path, ids, 2, policy, backends.FakeBackend())
    assert len(grades) == 6 and "infra" not in grades
    tdir = runs / "jobs" / "j" / ids[1] / "t0"
    receipt = json.loads((tdir / "receipt.json").read_text())
    assert receipt["evidence_sha256"]["events.jsonl"] == run_tasks.sha256_file(tdir / "episode" / "events.jsonl")
    assert receipt["controller_sha256"] == run_tasks.sha256_file(tdir / "controller.json")
    dom = load_domain("sim")
    per, extra = dom.score(runs, "j", ids, 2)
    assert set(per) == set(ids) and extra["infra_rate"] == 0.0 and extra["sim_seconds"] > 0
    # The scripted policy never submits on the first scenario.
    assert per[ids[0]].rewards == [0.0, 0.0] and extra["no_submission_rate"] == pytest.approx(2 / 6)
    assert all(c is not None and c > 100 for c in per[ids[1]].tokens), "cost = tokens + simulated seconds"
    rec = dom.load_trial(runs, "j", ids[1], 0)
    text = dom.render_trace(rec)
    assert "FAILURE CLASS" in text and "sampled trace" in text and "def grade" not in text
    assert ids[1] in dom.task_row(ids[1], rec, per[ids[1]])


def test_simulator_crash_is_infra_never_a_controller_failure(tmp_path):
    ids = scenarios.EVOLVE[:2]
    policy = ScriptedPolicy(lambda brief: submit(controller()))
    crash = backends.FakeBackend(crash_ids=[f"{ids[0]}--t0"])
    runs, grades = run_job(tmp_path, ids, 1, policy, crash)
    assert grades == ["infra", "infra"]
    assert not (runs / "jobs" / "j" / ids[0] / "t0" / "verdict.json").exists()
    dom = load_domain("sim")
    per, extra = dom.score(runs, "j", ids, 1)
    assert extra["infra_rate"] == 1.0 and all(tr.missing == 1 for tr in per.values())
    assert dom.guards(type("E", (), {"extra": {}})(), type("E", (), {"extra": extra})())
    # The retry re-executes the kept proposals without asking the policy again.
    calls = []
    retry = ScriptedPolicy(lambda brief: calls.append(brief) or submit(controller()))
    runs, grades = run_job(tmp_path, ids, 1, retry, backends.FakeBackend())
    assert not calls and set(grades) <= {"graded", "passed"}


def test_policy_endpoint_failure_is_infra_and_retried(tmp_path):
    ids = scenarios.EVOLVE[:1]
    down = ScriptedPolicy(lambda brief: ScriptedPolicy.PolicyError("endpoint down"))
    runs, grades = run_job(tmp_path, ids, 1, down, backends.FakeBackend())
    assert grades == ["infra"]
    runs, grades = run_job(tmp_path, ids, 1, ScriptedPolicy(lambda b: submit(controller())), backends.FakeBackend())
    assert grades[0] in ("graded", "passed")


def test_starting_harness_checks_then_submits():
    replies = iter([json.dumps({"action": "check", "controller": {"name": "x"}}),
                    "prose", submit(controller())])
    seen = []
    out = run_agent("brief", lambda m: (next(replies), 7),
                    lambda c: (seen.append(c), {"ok": False, "error": "schema_version must be 1"})[1],
                    lambda c: {}, 5)
    assert out["controller"] == controller() and out["tokens"] == 21 and seen == [{"name": "x"}]
    assert any("rejected: schema_version" in m["content"] for m in out["messages"])


def test_rollout_tool_uses_the_surrogate_not_the_graded_backend(tmp_path):
    task = scenarios.PRACTICE[2]
    check, rollout = run_tasks.make_tools(task, backends.FakeBackend(), tmp_path)
    verdict = rollout(controller())
    assert verdict["backend"] == "fake" and set(verdict["oracles"]) == set(oracles.ORACLES)
    assert check({"name": "x"})["ok"] is False


def test_critic_denylist_flags_seeds_and_grader_access():
    import re
    dom = load_domain("sim")
    for text in ("if 'straight-103' in brief:", "open('oracles.py')", "seed == 104", "if seed in (100, 101):",
                 "import subprocess"):
        assert any(re.search(p, text) for p, _ in dom.critic_patterns), text
    for innocent in ("gain = -250  # proportional term",
                     "use about -300 to -600 on input 0 and 100 to 120 on input 2"):
        assert not any(re.search(p, innocent) for p, _ in dom.critic_patterns), innocent
    # Held-out physics: the harness runs in process, so reaching for the
    # grader, the salt, the environment, files or interpreter internals is
    # forbidden by the critic (not prevented by the runtime).
    hidden, _ = dom.critic_patterns[-1]
    for text in ('os.environ["RRSI_SIM_PHYS_SALT_FILE"]', "open(p).read()", "import heldout_physics",
                 "from grader import phys_worker", "rollout.__closure__", "import oracles", "oracles.grade = f",
                 'backends.get("phys")', "import inspect", 'sys.modules["x"]', "Path(p).read_text()",
                 "threading.Thread(target=f)", "os.getenv('HOME')", "import importlib", "fake_backend.Bicycle"):
        assert re.search(hidden, text), text
    prose = ("Surrogate rollout: ... If any oracle failed or the car left the lane, fix the gains/signs and "
             "submit again", "Submit only if all oracles pass and the car ends centred",
             "inspect the rollout verdict before submitting", "checked by the grader.",
             "the car may respond late; prefer moderate gains")
    h0 = [p.read_text() for p in sorted((SIM / "harness").glob("*.py"))]
    for innocent in (*prose, *h0):
        assert not re.search(hidden, innocent), innocent[:80]


@pytest.mark.skipif(not os.environ.get("RRSI_SIM_CSF_RUNTIME"), reason="needs the CSF Go runtime")
def test_fake_semantics_match_the_csf_go_runtime():
    rng = random.Random(5)
    for _ in range(40):
        c = controller(rng.randint(-3000, 3000), rng.randint(-3000, 3000), rng.randint(-3000, 3000))
        features = [rng.randint(-10000, 10000) for _ in range(3)] + [0]
        request = {"kind": "REQUEST_KIND_EVALUATE", "observation": {"features": features}}
        compiled = subprocess.run([os.environ["RRSI_SIM_CSF_RUNTIME"]], text=True, capture_output=True,
                                  input=json.dumps({"kind": "REQUEST_KIND_COMPILE", "controller": c}) + "\n")
        request["program"] = json.loads(compiled.stdout)["program"]
        out = subprocess.run([os.environ["RRSI_SIM_CSF_RUNTIME"]], text=True, capture_output=True,
                             input=json.dumps(request) + "\n")
        action = json.loads(out.stdout)["action"]
        assert (int(action.get("steering", 0)), int(action.get("acceleration", 0))) == fake_backend.act(c, features)
    assert backends._go_compile({"schema_version": 1, "name": "r"})[0] is False
    nulled = controller()
    nulled["steering"]["value"] = None
    assert backends._go_compile(nulled)[0] is True, "ProtoJSON reads null as the default"


def test_any_command_honouring_the_contract_is_an_backend(tmp_path):
    script = tmp_path / "backend.py"
    script.write_text(
        "import argparse, json, sys\n"
        f"sys.path.insert(0, {str(SIM / 'bench')!r})\n"
        "import fake_backend\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--jobs'); ap.add_argument('--output')\n"
        "ap.add_argument('--run-id'); a = ap.parse_args()\n"
        "from pathlib import Path\n"
        "jobs = json.loads(Path(a.jobs).read_text())['episodes']\n"
        "fake_backend.run_batch(jobs, Path(a.output), a.run_id)\n")
    task = scenarios.PRACTICE[0]
    jobs = [{"id": "e", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
             "controller": controller()}]
    facts = backends.CommandBackend("command", [sys.executable, str(script)]).execute(jobs, tmp_path / "out", "r1")
    assert facts["exit"] == 0 and oracles.grade(tmp_path / "out" / "e", task)["status"] == "ok"
    with pytest.raises(backends.InfraError, match="without batch.json"):
        backends.CommandBackend("broken", [sys.executable, "-c", "raise SystemExit(3)"]).execute(
            jobs, tmp_path / "out2", "r2")
    with pytest.raises(backends.InfraError):
        backends.CommandBackend("none", [])


# ---- held-out physics (the `phys` simulator backend) ----------------------

# A fixture, not a secret: campaign salts live in an owner-only file outside
# the repository and are never committed.
FIXTURE_SALT = hashlib.sha256(b"rrsi-sim-test-salt-not-secret").digest()
PINNED_PHYSICS_SHA256 = "9a61ac9998799b594cb0e056fdb00c69cb27f07576603274784ea70b66efe883"
REFERENCE = controller(-250, -500, 500, name="csf-reference")
AGGRESSIVE = controller(-1500, -800, 800, name="aggressive")


def salt_file(tmp_path, salt=FIXTURE_SALT, mode=0o600, name="salt"):
    path = tmp_path / name
    path.write_text(salt.hex() + "\n")
    path.chmod(mode)
    return path


def phys_jobs(ids, trials=(0,), c=REFERENCE):
    return [{"id": f"{t}--t{i}", "scenario": scenarios.BY_ID[t]["scenario"],
             "goal_metres": scenarios.BY_ID[t]["goal_metres"], "controller": c} for t in ids for i in trials]


def phys_backend(monkeypatch, path, runtime=""):
    monkeypatch.setattr(backends, "_PHYS_SALT_FILE", str(path))
    monkeypatch.setattr(backends, "CSF_RUNTIME", runtime)
    return backends.get("phys")


def graded_reward(task_id, trial, c, salt=FIXTURE_SALT, scale=1.0, tmp=None):
    """One graded episode in process (the worker's loop, mirror runtime)."""
    task = scenarios.BY_ID[task_id]
    physics, noise_seed = heldout_physics.draw(salt, f"{task_id}--t{trial}", scale)
    job = {"id": "g", "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": c}
    out = tmp / f"{task_id}-{trial}-{c['name']}"
    fake_backend.run_episode(job, out, "r", plant_factory=lambda sc: heldout_vehicle.HeldoutVehicle(
        sc, physics, random.Random(noise_seed)))
    return oracles.grade(out, task)["reward"], out


def test_the_physics_draw_is_pinned_and_keyed_by_episode_id():
    physics, noise_seed = heldout_physics.draw(FIXTURE_SALT, "straight-100--t0")
    assert heldout_physics.physics_sha256(physics, noise_seed) == PINNED_PHYSICS_SHA256, "physics-v1 drifted"
    assert heldout_physics.draw(FIXTURE_SALT, "straight-100--t0") == (physics, noise_seed)
    hashes = {heldout_physics.physics_sha256(*heldout_physics.draw(FIXTURE_SALT, f"{t}--t{i}"))
              for t in scenarios.EVOLVE + scenarios.HELDOUT_IDS for i in range(3)}
    assert len(hashes) == 96, "every episode id gets its own vehicle"
    other = heldout_physics.draw(b"another-fixture-salt-of-32-bytes", "straight-100--t0")
    assert heldout_physics.physics_sha256(*other) != PINNED_PHYSICS_SHA256
    assert physics != heldout_vehicle.NOMINAL


def test_scale_zero_is_nominal_with_the_same_noise_seed():
    physics, noise_seed = heldout_physics.draw(FIXTURE_SALT, "straight-105--t2", scale=0)
    assert physics == heldout_vehicle.NOMINAL
    assert noise_seed == heldout_physics.draw(FIXTURE_SALT, "straight-105--t2")[1]
    assert [name for name, *_ in heldout_physics.RANGES][:3] == ["mass_kg", "inertia_factor", "front_axle_metres"]
    assert len(heldout_physics.RANGES) == 21


def test_graded_dynamics_differ_from_the_preview_for_a_fixed_draw(tmp_path):
    task = scenarios.PRACTICE[4]
    job = {"id": "p", "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": REFERENCE}
    fake_backend.run_episode(job, tmp_path / "preview", "r")
    preview = [json.loads(x)["state"] for x in (tmp_path / "preview" / "trace.jsonl").read_text().splitlines()]
    for scale in (0.0, 1.0):
        _, out = graded_reward(task["id"], 0, REFERENCE, scale=scale, tmp=tmp_path / f"s{scale}")
        graded = [json.loads(x)["state"] for x in (out / "trace.jsonl").read_text().splitlines()]
        gap = max(abs(a["lateral_metres"] - b["lateral_metres"]) for a, b in zip(preview, graded))
        assert gap > 1e-3, f"scale {scale}: the graded vehicle must not be the preview's"
    # The same draw replays the same graded episode.
    first = (graded_reward(task["id"], 1, REFERENCE, tmp=tmp_path / "a")[1] / "trace.jsonl").read_text()
    again = (graded_reward(task["id"], 1, REFERENCE, tmp=tmp_path / "b")[1] / "trace.jsonl").read_text()
    assert first == again


def test_phys_backend_pairs_episodes_by_id_not_by_run_or_order(tmp_path, monkeypatch):
    backend = phys_backend(monkeypatch, salt_file(tmp_path))
    jobs = phys_jobs(scenarios.EVOLVE[:3], trials=(0, 1)) + phys_jobs(scenarios.HELDOUT_IDS[:1])
    backend.execute(jobs, tmp_path / "one", "run-one")
    backend.execute(list(reversed(jobs)), tmp_path / "two", "run-two-later")
    for job in jobs:
        a, b = tmp_path / "one" / job["id"], tmp_path / "two" / job["id"]
        assert (a / "trace.jsonl").read_bytes() == (b / "trace.jsonl").read_bytes()
        ma, mb = (json.loads((d / "manifest.json").read_text()) for d in (a, b))
        assert ma["status"] == "completed" and ma["simulator"]["physics_sha256"] == mb["simulator"]["physics_sha256"]
        assert ma["simulator"]["distribution"] == "physics-v1" and ma["simulator"]["name"] == "phys"
    shas = {json.loads((tmp_path / "one" / j["id"] / "manifest.json").read_text())["simulator"]["physics_sha256"]
            for j in jobs}
    assert len(shas) == len(jobs)


def test_grader_output_never_holds_the_salt_or_the_drawn_values(tmp_path, monkeypatch):
    backend = phys_backend(monkeypatch, salt_file(tmp_path))
    jobs = phys_jobs(scenarios.EVOLVE[:4], trials=(0, 2)) + phys_jobs(scenarios.HELDOUT_IDS[:2], c=AGGRESSIVE)
    backend.execute(jobs, tmp_path / "out", "r")
    text = "\n".join(p.read_text() for p in sorted((tmp_path / "out").rglob("*")) if p.is_file())
    forbidden = [FIXTURE_SALT.hex(), "mass_kg", "tyre_friction", "cornering", "noise_seed", "steer_rate"]
    for job in jobs:
        physics, noise_seed = heldout_physics.draw(FIXTURE_SALT, job["id"])
        forbidden += [repr(physics.mass_kg), repr(physics.tyre_friction), repr(physics.front_cornering_n_per_rad),
                      repr(physics.steer_gain), str(noise_seed)]
    assert not [f for f in forbidden if f in text]
    manifest = json.loads((tmp_path / "out" / jobs[0]["id"] / "manifest.json").read_text())
    assert manifest["simulator"]["salt_id"] == heldout_physics.salt_id(FIXTURE_SALT)
    assert len(manifest["simulator"]["physics_sha256"]) == 64


def test_physics_v1_is_fair_to_the_reference_and_punishes_aggressive_gains(tmp_path):
    ids = scenarios.EVOLVE + scenarios.HELDOUT_IDS
    reference = [graded_reward(t, i, REFERENCE, tmp=tmp_path)[0] for t in ids for i in range(5)]
    aggressive = [graded_reward(t, i, AGGRESSIVE, tmp=tmp_path)[0] for t in ids for i in range(5)]
    assert sum(reference) / len(reference) >= 0.98
    assert sum(aggressive) / len(aggressive) <= sum(reference) / len(reference) - 0.2


def test_phys_backend_refuses_a_missing_or_weak_salt(tmp_path, monkeypatch):
    for path, why in ((tmp_path / "absent", "unreadable"),
                      (salt_file(tmp_path, mode=0o644, name="open"), "owner"),
                      (salt_file(tmp_path, salt=b"short", name="short"), "64 hex")):
        monkeypatch.setattr(backends, "_PHYS_SALT_FILE", str(path))
        with pytest.raises(SystemExit, match=why):
            backends.get("phys")
    monkeypatch.setattr(backends, "_PHYS_SALT_FILE", "")
    with pytest.raises(SystemExit, match="unset"):
        backends.get("phys")
    # A worker started without the salt never writes a verdict-bearing batch.
    worker = backends.CommandBackend("phys", [sys.executable, str(SIM / "grader" / "phys_worker.py")],
                                     {"RRSI_SIM_PHYS_SALT_FILE": ""})
    with pytest.raises(backends.InfraError, match="without batch.json"):
        worker.execute(phys_jobs(scenarios.EVOLVE[:1]), tmp_path / "out", "r")
    assert not (tmp_path / "out" / f"{scenarios.EVOLVE[0]}--t0").exists()


def test_phys_backend_refuses_episodes_that_are_not_graded_trials(tmp_path, monkeypatch):
    backend = phys_backend(monkeypatch, salt_file(tmp_path))
    task = scenarios.PRACTICE[0]
    jobs = [{"id": "rollout", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
             "controller": REFERENCE}, *phys_jobs(scenarios.EVOLVE[:1])]
    backend.execute(jobs, tmp_path / "out", "r")
    assert oracles.grade(tmp_path / "out" / "rollout", task)["status"] == "infra"
    assert oracles.grade(tmp_path / "out" / jobs[1]["id"], task)["status"] == "ok"


def test_the_preview_is_never_the_phys_grader(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "BACKEND", "phys")
    monkeypatch.delenv("RRSI_SIM_SURROGATE", raising=False)
    assert backends.surrogate_name() == "fake"
    monkeypatch.setenv("RRSI_SIM_SURROGATE", "phys")
    with pytest.raises(SystemExit, match="cannot serve as the rollout preview"):
        backends.surrogate_name()
    monkeypatch.setenv("RRSI_SIM_SURROGATE", "none")
    assert backends.surrogate_name() == "none"
    _, rollout = run_tasks.make_tools(scenarios.PRACTICE[0], backends.get("none"), tmp_path)
    assert rollout(REFERENCE) == {"error": "surrogate unavailable: preview disabled"}


@pytest.mark.skipif(not os.environ.get("RRSI_SIM_CSF_RUNTIME"), reason="needs the CSF Go runtime")
def test_phys_go_runtime_matches_the_mirror(tmp_path, monkeypatch):
    path = salt_file(tmp_path)
    jobs = phys_jobs(scenarios.EVOLVE[:3], trials=(0,)) + phys_jobs(scenarios.HELDOUT_IDS[:3], c=AGGRESSIVE)
    phys_backend(monkeypatch, path).execute(jobs, tmp_path / "mirror", "r")
    phys_backend(monkeypatch, path, os.environ["RRSI_SIM_CSF_RUNTIME"]).execute(jobs, tmp_path / "go", "r")
    for job in jobs:
        a, b = tmp_path / "mirror" / job["id"], tmp_path / "go" / job["id"]
        assert (a / "trace.jsonl").read_bytes() == (b / "trace.jsonl").read_bytes()
        manifest = json.loads((b / "manifest.json").read_text())
        assert manifest["simulator"]["controller_runtime"] == "csf-go" and manifest["controller_hash"]


PROBE_HARNESS = """import importlib.util, json, os, sys


def run_agent(brief, chat, check, rollout, max_steps):
    text, tokens = chat([{"role": "system", "content": "probe"}, {"role": "user", "content": brief}])
    seen = {"phys_env": sorted(k for k in os.environ if k.startswith("RRSI_SIM_PHYS_")),
            "grader_importable": [m for m in ("heldout_physics", "heldout_vehicle", "phys_worker")
                                  if importlib.util.find_spec(m) is not None],
            "grader_loaded": [m for m in ("heldout_physics", "heldout_vehicle", "phys_worker")
                              if m in sys.modules],
            "salt_in_backends": getattr(sys.modules.get("backends"), "_PHYS_SALT_FILE", None),
            "preview": rollout(json.loads(text)["controller"]).get("backend")}
    return {"controller": json.loads(text)["controller"], "tokens": tokens,
            "messages": [{"role": "assistant", "content": json.dumps(seen)}]}
"""


def policy_server(reply):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"choices": [{"message": {"content": reply}}], "usage": {"total_tokens": 50}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def checkout_copy(tmp_path, harness=PROBE_HARNESS):
    """A worktree-shaped copy (the runner checks out the evaluated commit)."""
    root = tmp_path / "wt"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(SIM, root / "domains" / "sim", ignore=ignore)
    shutil.copytree(ROOT / "rrsi", root / "rrsi", ignore=ignore)
    (root / "domains" / "toy").mkdir()
    shutil.copy(ROOT / "domains" / "toy" / "policy.py", root / "domains" / "toy" / "policy.py")
    (root / "domains" / "sim" / "harness" / "agent.py").write_text(harness)
    return root


def test_the_agent_never_shares_a_process_with_the_phys_grader(tmp_path, monkeypatch):
    root = checkout_copy(tmp_path)
    server, calls = policy_server(submit(REFERENCE))
    path = salt_file(tmp_path)
    try:
        monkeypatch.setenv("RRSI_POLICY_BACKEND", "openai")
        monkeypatch.setenv("RRSI_POLICY_MODEL", "scripted")
        monkeypatch.setenv("RRSI_POLICY_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv("RRSI_SIM_BACKEND", "phys")
        monkeypatch.setenv("RRSI_SIM_PHYS_SALT_FILE", str(path))
        monkeypatch.delenv("RRSI_SIM_SURROGATE", raising=False)
        dom = load_domain("sim")
        ids = scenarios.EVOLVE[:2]
        dom.run(root, tmp_path / "runs", "j", ids, 1)
        assert len(calls) == 2
        for t in ids:
            tdir = tmp_path / "runs" / "jobs" / "j" / t / "t0"
            seen = json.loads(json.loads((tdir / "traj.json").read_text())["messages"][0]["content"])
            assert seen == {"phys_env": [], "grader_importable": [], "grader_loaded": [],
                            "salt_in_backends": "", "preview": "fake"}
            verdict = json.loads((tdir / "verdict.json").read_text())
            manifest = json.loads((tdir / "episode" / "manifest.json").read_text())
            assert verdict["backend"] == "phys" and verdict["status"] == "ok"
            assert manifest["simulator"]["name"] == "phys" and manifest["simulator"]["distribution"] == "physics-v1"
        # Without a usable salt the evaluation stops before any policy call.
        monkeypatch.setenv("RRSI_SIM_PHYS_SALT_FILE", str(tmp_path / "absent"))
        with pytest.raises(SystemExit, match="phys simulator backend"):
            dom.run(root, tmp_path / "runs", "j2", ids, 1)
        assert len(calls) == 2
    finally:
        server.shutdown()


def test_a_runner_without_the_phase_split_fails_loudly(tmp_path, monkeypatch):
    root = checkout_copy(tmp_path)
    (root / "domains" / "sim" / "bench" / "run_tasks.py").write_text(
        "import argparse\nap = argparse.ArgumentParser()\n"
        "for f in ('--runs', '--job', '--ids', '--n', '--backend'):\n    ap.add_argument(f)\nap.parse_args()\n")
    monkeypatch.setenv("RRSI_SIM_BACKEND", "fake")
    with pytest.raises(SystemExit, match="predates the propose/grade split"):
        load_domain("sim").run(root, tmp_path / "runs", "j", scenarios.EVOLVE[:1], 1)


def test_evidence_the_agent_leaves_behind_is_discarded(tmp_path):
    runs, task = tmp_path / "runs", scenarios.PRACTICE[0]
    tdir = runs / "jobs" / "j" / task["id"] / "t0"

    def forger(brief, chat, check, rollout, max_steps):
        (tdir / "verdict.json").write_text(json.dumps({"status": "ok", "reward": 1.0}))
        (tdir / "episode").mkdir()
        (tdir / "episode" / "manifest.json").write_text("{}")
        return {"controller": controller(lateral=400, heading=600), "messages": [], "tokens": 1}

    policy = ScriptedPolicy(lambda brief: submit(controller()))
    run_tasks.propose(runs, "j", task["id"], 0, policy, backends.FakeBackend(), forger)
    assert not (tdir / "verdict.json").exists() and not (tdir / "episode").exists()
    run_tasks.execute(runs, "j", [(task["id"], 0)], backends.FakeBackend())
    run_tasks.grade_trial(tdir, task, "fake")
    assert json.loads((tdir / "verdict.json").read_text())["reward"] < 1.0


class SerialPool:
    def map(self, fn, items):
        return [fn(x) for x in items]


def test_agreement_replays_stored_trials_on_preview_and_graded_vehicle(tmp_path):
    ids = scenarios.EVOLVE[:3]
    policy = ScriptedPolicy(lambda brief: submit(controller(-1500, -800, 800)))
    runs, _ = run_job(tmp_path, ids, 2, policy, backends.FakeBackend(), job="stored")
    agreement._init(FIXTURE_SALT)
    report, rows = agreement.replay_jobs(SerialPool(), runs, ["stored"])
    r = report["stored"]
    assert r["n"] == 6 and r["preview_equals_stored"] == 1.0, "the fake backend is the preview"
    assert r["preview_S"] == r["stored_S"] and r["graded_S"] is not None
    assert {(row["task"], row["trial"]) for row in rows} == {(t, i) for t in ids for i in range(2)}
    expected = graded_reward(ids[0], 1, controller(-1500, -800, 800), tmp=tmp_path / "direct")[0]
    assert [row["graded"] for row in rows if (row["task"], row["trial"]) == (ids[0], 1)] == [expected]
    arms = agreement.arms_report(rows, {"A": ["stored"], "B": ["stored"]}, "A")
    assert arms["B"]["paired_graded_vs_A"]["mean"] == 0.0
