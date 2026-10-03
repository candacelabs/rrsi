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
    """Labels by the first signal (an alias for odd ids, a canonical id for even
    ones); clusters hook/permission together where a mode still clusters;
    writes a fixed task."""

    def __init__(self):
        self.calls = {"label": 0, "cluster": 0, "task": 0}
        self.prompts = []
        self.schemas = []

    def __call__(self, system, prompt, schema, out):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if "labels" in schema["properties"]:
            self.calls["label"] += 1
            ids = [l.split("]")[0][1:] for l in prompt.splitlines() if l.startswith("[ep")]
            return {"labels": [{"id": i, "pattern": "noise" if i.endswith("9") else
                                ("hook_timeout_blocks_writes" if int(i[2:]) % 2 else "permission_denied_tool_use"),
                                "new_pattern": "", "summary": f"summary of {i}", "harness_fixable": True} for i in ids]}
        if "clusters" in schema["properties"]:
            self.calls["cluster"] += 1
            return {"clusters": [{"key": "Blocked Tools", "title": "Tool calls blocked",
                                  "patterns": ["hook_timeout_blocks_writes", "permission_denied_tool_use", "made_up"]}]}
        self.calls["task"] += 1
        extra = ({"trigger_rule": {"when": "w", "owner_resolution": "o", "payload": "p",
                                   "rule": "when two sessions edit one file, message the claimant"}}
                 if "trigger_rule" in schema["properties"] else {})
        return {**extra, "title": "Stop blocked tool calls", "struggle_pattern": "p", "root_cause_hypothesis": "h",
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
        # traces mode labels against the canonical vocabulary: the ids are the
        # clusters, so no cluster call, and one task per id.
        self.assertEqual(fake.calls, {"label": 4, "cluster": 0, "task": 2})
        self.assertIn("hook_timeout_blocks_tools", fake.schemas[0]["properties"]["labels"]["items"]["properties"]["pattern"]["enum"])
        task = json.loads((self.out / "tasks" / "permission_denied_tool_use.json").read_text())
        ev = task["evidence"]
        # 20 episodes: even ids (all in proj0) are permission denials, odd ids (proj1) hook timeouts,
        # ep0009 and ep0019 noise.
        self.assertEqual((ev["episodes"], ev["sessions"], ev["projects"]), (10, 3, 1))
        self.assertEqual(ev["signals"], {"hook_timeout": 10})
        self.assertEqual(len(ev["episode_ids"]), 10)
        hook = json.loads((self.out / "tasks" / "hook_timeout_blocks_tools.json").read_text())["evidence"]
        self.assertEqual((hook["episodes"], hook["signals"]), (8, {"tool_error": 8}))
        self.assertEqual(run["tasks"], 2)
        index = json.loads((self.out / "tasks" / "index.json").read_text())
        self.assertEqual([i["episodes"] for i in index], [10, 8])
        exam = (self.out / "exam_candidates.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(exam[0])["task"], "permission_denied_tool_use")
        report = (self.out / "REPORT.md").read_text()
        self.assertIn("| 1 | Stop blocked tool calls | 10 | 3 | 1 |", report)
        self.assertIn("c \\| d", report)
        self.assertIn("| hook_timeout | 10 | 20 |", report)
        # The alias the model answered is kept as `raw`; the record carries its canonical id.
        recs = [json.loads(l) for l in (self.out / "labels.jsonl").read_text().splitlines()]
        odd = next(r for r in recs if r["id"] == "ep0001")
        self.assertEqual((odd["raw"], odd["pattern"], odd["canonical"]),
                         ("hook_timeout_blocks_writes", "hook_timeout_blocks_tools", True))

    def test_handoffs_mode_carries_trigger_rules_and_peer_outcomes(self):
        out = Path(self.tmp.name) / "handoffs"
        out.mkdir()
        eps = corpus()
        eps[0]["extra"] = {"outcomes": [{"event": 0, "from": "x", "coordinator": False, "retraction": True,
                                         "acted": True, "replied": False}]}
        eps[1]["extra"] = {"other_session": "abcdef0123456789", "other_project": "proj9"}
        (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        (out / "handoffs-summary.json").write_text(json.dumps({
            "transcripts": 3, "processed": 3, "skipped_unchanged": 0, "sessions": 3, "projects": 2, "events": 99,
            "episodes": len(eps), "episodes_per_signal": {"peer_message": 20}, "hits_per_signal": {"peer_message": 25},
            "sessions_with_episodes": 3, "seconds": 0.1,
            "peer": {"messages": 4, "from_peers": 3, "from_coordinator": 1, "acted": 3, "replied": 2,
                     "retractions": 1, "senders": 2, "message_out_calls": 5, "message_out_failed": 0}}))
        fake = FakeModel()
        run = M.mine(miner="handoffs", out=out, skip_traces=True, complete=fake, batch=6, jobs=2, top=5,
                     log=lambda _: None)
        self.assertEqual(run["miner"], "handoffs")
        self.assertEqual(run["top"][0]["trigger_rule"], "when two sessions edit one file, message the claimant")
        task = json.loads((out / "tasks" / "blocked_tools.json").read_text())
        self.assertEqual(task["trigger_rule"]["owner_resolution"], "o")
        report = (out / "REPORT.md").read_text()
        self.assertIn("# Agent handoffs: report", report)
        self.assertIn("| Receiver acted | 3 (75%) |", report)
        self.assertIn("Trigger rule |", report)
        sent = "\n".join(fake.prompts)
        self.assertIn("acted=true replied=false retraction=true", sent)
        self.assertIn("other session: abcdef01", sent)

    def test_pr_gap_mode_reports_measured_rules_and_runs_the_miner_with_github(self):
        out = Path(self.tmp.name) / "pr-gap"
        out.mkdir()
        eps = corpus()
        eps[0]["extra"] = {"outcome": "pushed_no_pr", "commits": 3, "commits_before_first_push": 2,
                           "unpushed_at_end": 1, "commit_to_push_secs": 1440, "push_to_pr_secs": None, "brief": "defers"}
        eps[1]["extra"] = {"phrases": ["don't open a separate pr"]}
        (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        dist = {"n": 2, "median": 1.0, "p90": 2.0, "max": 3.0}
        pop = {"runs": 5, "active_runs": 4, "outcomes": {"pr_opened": 2, "pushed_no_pr": 1, "never_pushed": 1},
               "gap_runs": 2, "github_said_no_pr_at_push": 1, "commit_to_push_min": dist, "push_to_pr_min": dist,
               "commits_before_first_push": dist, "runs_with_unpushed_commits_at_end": 1}
        rule = {"minutes": 5, "rule": "when an agent's first commit is 5 minutes old with no push, open a draft PR",
                "fires": 3, "gaps_caught": 2, "nags": 1, "gaps_missed": 0, "score": 1}
        summary = {"transcripts": 5, "processed": 5, "skipped_unchanged": 0, "sessions": 2, "projects": 1, "events": 50,
                   "episodes": len(eps), "episodes_per_signal": {"pushed_no_pr": 20}, "hits_per_signal": {"pushed_no_pr": 20},
                   "sessions_with_episodes": 2, "seconds": 0.1, "all": pop, "main_sessions": pop, "subagents": pop,
                   "subagent_gap_by_brief": {"defers": [2, 2], "early": [2, 0]},
                   "github": {"joined": True, "lookups": 3, "failed": 0, "pr_existing": 1,
                              "pushed_no_pr": {"confirmed_no_pr": 1}},
                   "rules": [rule], "top_rule": rule}
        (out / "pr-gap-summary.json").write_text(json.dumps(summary))
        fake = FakeModel()
        run = M.mine(miner="pr-gap", out=out, skip_traces=True, complete=fake, batch=6, jobs=2, top=5,
                     log=lambda _: None)
        self.assertEqual(run["miner"], "pr-gap")
        self.assertIn("trigger_rule", run["top"][0])
        report = (out / "REPORT.md").read_text()
        self.assertIn("# Agents without a PR: report", report)
        self.assertIn("| subagents | 4 | 2 | 0 | 0 | 1 | 1 | 2 | 1 |", report)
        self.assertIn("| defers | 2 | 2 | 100% |", report)
        self.assertIn("| 5 | 3 | 2 | 1 | 0 | 1 |", report)
        self.assertIn("**Top rule:** when an agent's first commit is 5 minutes old", report)
        self.assertIn("1 confirmed with no PR", report)
        sent = "\n".join(fake.prompts)
        self.assertIn("run: outcome=pushed_no_pr commits=3 before_first_push=2", sent)
        self.assertIn("phrases: don't open a separate pr", sent)
        self.assertIn("always have a PR", M.MODES["pr-gap"].label_system)
        self.assertEqual(M.MODES["pr-gap"].miner_args, ("--github",))

    def test_run_miner_passes_mode_arguments(self):
        seen = {}

        class Done:
            stdout = "{}"

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return Done()

        orig_run, orig_bin = M.subprocess.run, M.rust_binary
        M.subprocess.run, M.rust_binary = fake_run, (lambda build=True: Path("/x/rrsi-mine"))
        try:
            M.run_miner("pr-gap", Path("/o"), None, "", 8, ["skip"], ("--github",))
        finally:
            M.subprocess.run, M.rust_binary = orig_run, orig_bin
        self.assertEqual(seen["cmd"], ["/x/rrsi-mine", "pr-gap", "--out", "/o", "--jobs", "8", "--since", "",
                                       "--github", "--exclude", "skip"])

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

    def test_a_network_timeout_is_retried_not_an_auth_failure(self):
        # The Copilot CLI's generic troubleshooting text says "re-authenticate" even when
        # the model catalog merely timed out behind a flaky proxy; that must not end the run.
        from rrsi.harness import llm
        from rrsi.harness.llm import LLMAuthError
        timeout = ("copilot rc=1: Error: Failed to load models\n\nError: Model catalog request timed out after 30000ms\n"
                   "  • Start 'copilot' and run the '/login' command to re-authenticate")
        with self.assertRaises(LLMError) as ctx:
            llm._raise(timeout)
        self.assertNotIsInstance(ctx.exception, LLMAuthError)
        with self.assertRaises(LLMAuthError):
            llm._raise("copilot rc=1: Error: Failed to authenticate. Not logged in.")
        dns = ("copilot rc=1: onnection: dns error: error resolving DNS: failed to lookup address information: "
               "Temporary failure in name resolution [ENOTFOUND]\n\nCopilot could not retrieve the list of available "
               "models.\n\nTo resolve this, try the following:\n  • Start 'copilot' and run the '/login' command to "
               "re-authenticate")
        with self.assertRaises(LLMError) as ctx:
            llm._raise(dns)
        self.assertNotIsInstance(ctx.exception, LLMAuthError)
        self.assertTrue(llm.is_transient(dns))
        calls = {"n": 0}

        def flaky(backend, model, system, prompt, schema, cwd, effort):
            calls["n"] += 1
            if calls["n"] < 3:
                llm._raise(timeout)
            return {"labels": []}

        old = (M.complete_json, M.BACKOFF_SECONDS)
        M.complete_json, M.BACKOFF_SECONDS = flaky, 0
        try:
            self.assertEqual(M.completer("copilot", "m", "low")("s", "p", {}, self.out), {"labels": []})
        finally:
            M.complete_json, M.BACKOFF_SECONDS = old
        self.assertEqual(calls["n"], 3)

    def test_an_auth_failure_stops_the_run(self):
        from rrsi.harness.llm import LLMAuthError

        def logged_out(system, prompt, schema, out):
            raise LLMAuthError("Failed to authenticate")

        with self.assertRaises(LLMAuthError):
            M.mine(out=self.out, skip_traces=True, complete=logged_out, batch=6, jobs=2, log=lambda _: None)

    def test_prompts_are_redacted(self):
        eps = [episode(0, "s", "p", {"tool_error": 1},
                       text="ssh user@example.invalid at 203.0.113.7 token=abc123 in /home/someone/x " + "a" * 40
                            + " via https://proxy.corp.example.com:8080/v1 and gpu-01.lab.internal, not tests/test_x.py")]
        (self.out / "episodes.jsonl").write_text(json.dumps(eps[0]) + "\n")
        fake = FakeModel()
        self.run_mine(fake)
        sent = "\n".join(fake.prompts)
        for leaked in ("example.invalid", "203.0.113.7", "abc123", "/home/someone", "a" * 40,
                       "corp.example.com", "gpu-01.lab.internal"):
            self.assertNotIn(leaked, sent)
        self.assertIn("<email>", sent)
        self.assertIn("<host>", sent)
        self.assertIn("test_x.py", sent, "file names are not hostnames")

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
