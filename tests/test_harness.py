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
"""Harness miner, LLM stage: synthetic episodes and a fake model only.

    python3 -m pytest -q tests/test_harness.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi.harness import mine as M  # noqa: E402
from rrsi.harness.llm import LLMError, parse_json  # noqa: E402


def episode(i, session, project, signals, text="Exit code 1"):
    return {"id": f"ep{i:04d}", "project": project, "session_id": session, "agent_id": None,
            "subagent": False, "file": f"{project}/{session}.jsonl", "start": f"2026-01-0{1 + i % 5}T00:00:00Z",
            "end": "2026-01-01T00:00:01Z", "start_event": i, "end_event": i,
            "signals": signals, "user_turn": "make the widget build",
            "context": [{"i": i, "ts": "t", "kind": "tool_result", "tool": "Bash", "is_error": True,
                         "signals": list(signals), "text": text}],
            "counts": {"span_events": 1, "tool_calls": 0, "tool_errors": 1, "human_turns": 0, "session_events": 9}}


class FakeModel:
    """Labels by the first signal; clusters hook/permission together; writes a fixed task."""

    def __init__(self):
        self.calls = {"label": 0, "cluster": 0, "task": 0}
        self.prompts = []

    def __call__(self, system, prompt, schema, out):
        self.prompts.append(prompt)
        if "labels" in schema["properties"]:
            self.calls["label"] += 1
            ids = [l.split("]")[0][1:] for l in prompt.splitlines() if l.startswith("[ep")]
            return {"labels": [{"id": i, "pattern": "Noise" if i.endswith("9") else
                                ("Hook Timeout!" if int(i[2:]) % 2 else "permission blocked"),
                                "summary": f"summary of {i}", "harness_fixable": True} for i in ids]}
        if "clusters" in schema["properties"]:
            self.calls["cluster"] += 1
            return {"clusters": [{"key": "Blocked Tools", "title": "Tool calls blocked",
                                  "patterns": ["hook_timeout", "permission_blocked", "made_up"]}]}
        self.calls["task"] += 1
        return {"title": "Stop blocked tool calls", "struggle_pattern": "p", "root_cause_hypothesis": "h",
                "proposed_fix": {"kind": "tool_cli_fix", "change": "c | d"}, "acceptance_check": "a",
                "priority": "P1", "exam_candidate": {"checkable": True, "before": "b", "after": "a", "check": "k"}}


def corpus():
    return [episode(i, f"s{i % 3}", f"proj{i % 2}", {"tool_error": 1} if i % 2 else {"hook_timeout": 2})
            for i in range(20)]


class HarnessMinerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "harness"
        self.out.mkdir()
        eps = corpus()
        (self.out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        (self.out / "traces-summary.json").write_text(json.dumps({
            "transcripts": 3, "processed": 3, "skipped_unchanged": 0, "sessions": 3, "projects": 2,
            "events": 99, "episodes": len(eps), "episodes_per_signal": {"tool_error": 10, "hook_timeout": 10},
            "hits_per_signal": {"tool_error": 10, "hook_timeout": 20}, "sessions_with_episodes": 3, "seconds": 0.1}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_mine(self, fake):
        return M.mine(out=self.out, skip_traces=True, complete=fake, batch=6, jobs=2, top=5, log=lambda _: None)

    def test_end_to_end_counts_are_measured_not_generated(self):
        fake = FakeModel()
        run = self.run_mine(fake)
        self.assertEqual(fake.calls, {"label": 4, "cluster": 1, "task": 1})
        task = json.loads((self.out / "tasks" / "blocked_tools.json").read_text())
        ev = task["evidence"]
        # 20 episodes, ep0009 and ep0019 labelled noise.
        self.assertEqual(ev["episodes"], 18)
        self.assertEqual(ev["sessions"], 3)
        self.assertEqual(ev["projects"], 2)
        self.assertEqual(ev["signals"], {"tool_error": 8, "hook_timeout": 10})
        self.assertEqual(len(ev["episode_ids"]), 18)
        self.assertEqual(run["tasks"], 1)
        index = json.loads((self.out / "tasks" / "index.json").read_text())
        self.assertEqual(index[0]["episodes"], 18)
        exam = (self.out / "exam_candidates.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(exam[0])["task"], "blocked_tools")
        report = (self.out / "REPORT.md").read_text()
        self.assertIn("| 1 | Stop blocked tool calls | 18 | 3 | 2 |", report)
        self.assertIn("c \\| d", report)
        self.assertIn("| hook_timeout | 10 | 20 |", report)

    def test_rerun_is_cached(self):
        self.run_mine(FakeModel())
        fake = FakeModel()
        self.run_mine(fake)
        self.assertEqual(fake.calls, {"label": 0, "cluster": 0, "task": 0})

    def test_failed_batches_are_counted_and_retried_next_run(self):
        calls = {"n": 0}
        good = FakeModel()

        def flaky(system, prompt, schema, out):
            if "labels" in schema["properties"]:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise LLMError("boom")
            return good(system, prompt, schema, out)

        run = M.mine(out=self.out, skip_traces=True, complete=flaky, batch=6, jobs=1, top=5, log=lambda _: None)
        self.assertEqual(run["failed_calls"], 1)
        fake = FakeModel()
        self.run_mine(fake)
        self.assertEqual(fake.calls["label"], 1)

    def test_an_auth_failure_stops_the_run(self):
        from rrsi.harness.llm import LLMAuthError

        def logged_out(system, prompt, schema, out):
            raise LLMAuthError("Failed to authenticate")

        with self.assertRaises(LLMAuthError):
            M.mine(out=self.out, skip_traces=True, complete=logged_out, batch=6, jobs=2, log=lambda _: None)

    def test_prompts_are_redacted(self):
        eps = [episode(0, "s", "p", {"tool_error": 1},
                       text="ssh user@example.invalid at 100.64.1.2 token=abc123 in /home/someone/x " + "a" * 40)]
        (self.out / "episodes.jsonl").write_text(json.dumps(eps[0]) + "\n")
        fake = FakeModel()
        self.run_mine(fake)
        sent = "\n".join(fake.prompts)
        for leaked in ("example.invalid", "100.64.1.2", "abc123", "/home/someone", "a" * 40):
            self.assertNotIn(leaked, sent)
        self.assertIn("<email>", sent)

    def test_output_inside_a_work_tree_is_refused(self):
        repo = Path(self.tmp.name) / "repo"
        (repo / ".git").mkdir(parents=True)
        with self.assertRaises(M.PrivacyError):
            M.mine(out=repo / "private", skip_traces=True, complete=FakeModel(), log=lambda _: None)
        self.assertFalse((repo / "private").exists())

    def test_digest_keeps_signal_events_and_truncates(self):
        e = episode(1, "s", "p", {"retry": 1}, text="x" * 1000)
        d = M.digest(e, width=50)
        self.assertIn("*retry", d)
        self.assertLess(max(len(l) for l in d.splitlines()), 120)

    def test_parse_json_accepts_fenced_and_bare(self):
        self.assertEqual(parse_json('text ```json\n{"a": 1}\n``` more'), {"a": 1})
        self.assertEqual(parse_json('noise {"a": {"b": 2}} tail'), {"a": {"b": 2}})
        # literal newlines in strings and prose after the object (seen from CLI backends)
        self.assertEqual(parse_json('{"a": "x\ny"} then } more'), {"a": "x\ny"})
        with self.assertRaises(LLMError):
            parse_json("no json here")

    def test_cli_parses(self):
        from rrsi import __main__ as cli
        with self.assertRaises(SystemExit):
            cli.main(["harness", "mine", "--backend", "nope"])


if __name__ == "__main__":
    unittest.main()
