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

Runs on the dependency-free fake engine. With RRSI_SIM_CSF_RUNTIME set to the
CSF Go runtime, the fake controller semantics are also compared with it.
"""

import json
import os
import random
import subprocess
import sys
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
_SHARED = ("harness", "render", "briefs", "run_tasks", "engines", "oracles", "fake_engine", "scenarios")
_SAVED = {n: m for n, m in sys.modules.items() if n.split(".")[0] in _SHARED}
for _name in _SAVED:
    del sys.modules[_name]
_PATH, _MODULES = list(sys.path), set(sys.modules)
sys.path[:0] = [str(SIM), str(SIM / "bench"), str(SIM / "data")]
import engines                      # noqa: E402
import fake_engine                  # noqa: E402
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
    fake_engine.check(controller())
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
        with pytest.raises(fake_engine.Rejected):
            fake_engine.check(c)
    assert engines.check(bad[0])[0] is False


def test_scale_truncates_toward_zero_like_the_go_runtime():
    c = {"schema_version": 1, "name": "s", "steering": scale(0, 1), "acceleration": scale(0, -1)}
    assert fake_engine.act(c, [1999, 0, 0, 0]) == (1, -1)
    assert fake_engine.act(c, [-1999, 0, 0, 0]) == (-1, 1)


def grade_one(c, task=None, tmp=None):
    task = task or scenarios.PRACTICE[0]
    job = {"id": "e", "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": c}
    fake_engine.run_batch([job], tmp, "r")
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
    assert measured == set(fake_engine.METRICS)
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


def run_job(tmp_path, ids, k, policy, engine, job="j"):
    runs = tmp_path / "runs"
    surrogate = engines.FakeEngine()
    work = [(t, i) for t in ids for i in range(k)]
    for t, i in work:
        run_tasks.propose(runs, job, t, i, policy, surrogate)
    pending = [(t, i) for t, i in work
               if (runs / "jobs" / job / t / f"t{i}" / "controller.json").is_file()
               and not (runs / "jobs" / job / t / f"t{i}" / "episode" / "manifest.json").is_file()]
    run_tasks.execute(runs, job, pending, engine)
    return runs, [run_tasks.grade_trial(runs / "jobs" / job / t / f"t{i}", scenarios.BY_ID[t], engine.name)
                  for t, i in work]


def test_runner_end_to_end_with_receipts_and_scoring(tmp_path):
    ids = scenarios.EVOLVE[:3]
    skipped = scenarios.brief(scenarios.BY_ID[ids[0]])
    policy = ScriptedPolicy(lambda brief: submit(controller()) if skipped not in brief else "no json")
    runs, grades = run_job(tmp_path, ids, 2, policy, engines.FakeEngine())
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
    crash = engines.FakeEngine(crash_ids=[f"{ids[0]}--t0"])
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
    runs, grades = run_job(tmp_path, ids, 1, retry, engines.FakeEngine())
    assert not calls and set(grades) <= {"graded", "passed"}


def test_policy_endpoint_failure_is_infra_and_retried(tmp_path):
    ids = scenarios.EVOLVE[:1]
    down = ScriptedPolicy(lambda brief: ScriptedPolicy.PolicyError("endpoint down"))
    runs, grades = run_job(tmp_path, ids, 1, down, engines.FakeEngine())
    assert grades == ["infra"]
    runs, grades = run_job(tmp_path, ids, 1, ScriptedPolicy(lambda b: submit(controller())), engines.FakeEngine())
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


def test_rollout_tool_uses_the_surrogate_not_the_graded_engine(tmp_path):
    task = scenarios.PRACTICE[2]
    check, rollout = run_tasks.make_tools(task, engines.FakeEngine(), tmp_path)
    verdict = rollout(controller())
    assert verdict["engine"] == "fake" and set(verdict["oracles"]) == set(oracles.ORACLES)
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
        assert (int(action.get("steering", 0)), int(action.get("acceleration", 0))) == fake_engine.act(c, features)
    assert engines._go_compile({"schema_version": 1, "name": "r"})[0] is False


def test_any_command_honouring_the_contract_is_an_engine(tmp_path):
    script = tmp_path / "engine.py"
    script.write_text(
        "import argparse, json, sys\n"
        f"sys.path.insert(0, {str(SIM / 'bench')!r})\n"
        "import fake_engine\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--jobs'); ap.add_argument('--output')\n"
        "ap.add_argument('--run-id'); a = ap.parse_args()\n"
        "from pathlib import Path\n"
        "jobs = json.loads(Path(a.jobs).read_text())['episodes']\n"
        "fake_engine.run_batch(jobs, Path(a.output), a.run_id)\n")
    task = scenarios.PRACTICE[0]
    jobs = [{"id": "e", "scenario": task["scenario"], "goal_metres": task["goal_metres"],
             "controller": controller()}]
    facts = engines.CommandEngine("command", [sys.executable, str(script)]).execute(jobs, tmp_path / "out", "r1")
    assert facts["exit"] == 0 and oracles.grade(tmp_path / "out" / "e", task)["status"] == "ok"
    with pytest.raises(engines.InfraError, match="without batch.json"):
        engines.CommandEngine("broken", [sys.executable, "-c", "raise SystemExit(3)"]).execute(
            jobs, tmp_path / "out2", "r2")
    with pytest.raises(engines.InfraError):
        engines.CommandEngine("none", [])
