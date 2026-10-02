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
"""The compounding measurement: canonical vocabulary, day boundary, host
attribution, daily records. Synthetic facts, episodes and harness runs only.

    python3 -m pytest -q tests/test_measure.py
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi.harness import measure as X  # noqa: E402
from rrsi.harness import mine as M  # noqa: E402
from rrsi.harness.patterns import NEW_PATTERN, NOISE, Pattern, PatternId, Vocabulary, VocabularyError  # noqa: E402

SENTINEL = "PRIVATE-SENTINEL-TEXT-NEVER-IN-OUTPUT"
# 2026-10-01 is PDT (UTC-7): the LA day ends at 07:00Z on Oct 2.
NOW_OPEN = datetime(2026, 10, 2, 6, 59, 59, tzinfo=timezone.utc)
NOW_CLOSED = datetime(2026, 10, 2, 7, 0, 0, tzinfo=timezone.utc)


def vocab(*patterns: tuple[str, str, tuple]) -> Vocabulary:
    return Vocabulary([Pattern(PatternId(i), t, a) for i, t, a in patterns], 1, "test")


def episode(i: int, session: str, start: str) -> dict:
    return {"id": f"ep{i:04d}", "project": "proj", "session_id": session, "agent_id": None, "subagent": False,
            "file": f"proj/{session}.jsonl", "start": start, "end": start, "start_event": i, "end_event": i,
            "signals": {"tool_error": 1}, "user_turn": SENTINEL,
            "context": [{"i": i, "ts": start, "kind": "tool_result", "tool": "Bash", "is_error": True,
                         "signals": ["tool_error"], "text": SENTINEL}],
            "counts": {"span_events": 1, "tool_calls": 0, "tool_errors": 1, "human_turns": 0, "session_events": 9}}


def facts(hours: dict, hook: int = 0, guard: int = 0) -> dict:
    return {"mtime": 1, "size": 1, "hash": "h", "since": "", "episodes": 0, "events": 1,
            "facts": {"version": 1, "tool_calls_by_utc_hour": hours, "hook_timeouts": hook, "guard_rejections": guard}}


class VocabularyTest(unittest.TestCase):
    def test_aliases_fold_to_ids_and_unknown_slugs_escape(self):
        v = vocab(("hook_timeout_blocks_tools", "t", ("hook_timeout_blocks_writes", "desktop_hook_no_response")))
        self.assertEqual(v.fold("Hook Timeout Blocks Writes"), v.fold("hook_timeout_blocks_writes"))
        self.assertEqual((v.fold("hook_timeout_blocks_writes").pattern, v.fold("hook_timeout_blocks_writes").canonical),
                         ("hook_timeout_blocks_tools", True))
        self.assertEqual(v.fold("hook_timeout_blocks_tools").canonical, True)
        self.assertEqual((v.fold("noise").pattern, v.fold("noise").canonical), (NOISE, True))
        esc = v.fold("something nobody named")
        self.assertEqual((esc.pattern, esc.canonical), ("something_nobody_named", False))
        self.assertEqual(v.choices()[-2:], [NOISE, NEW_PATTERN])
        self.assertEqual(v.title("hook_timeout_blocks_tools"), "t")
        self.assertEqual(v.title("something_nobody_named"), "something nobody named")

    def test_ids_and_aliases_are_disjoint_and_unique(self):
        with self.assertRaises(VocabularyError):
            vocab(("a", "t", ("x",)), ("b", "t", ("x",)))
        with self.assertRaises(VocabularyError):
            vocab(("a", "t", ("b",)), ("b", "t", ()))
        with self.assertRaises(VocabularyError):
            vocab(("a", "t", ()), ("a", "t", ()))
        with self.assertRaises(VocabularyError):
            vocab((NOISE, "t", ()))

    def test_shipped_vocabulary_loads_and_is_a_fixed_choice(self):
        v = Vocabulary.load()
        self.assertGreaterEqual(len(v.ids), 30)
        self.assertGreaterEqual(len(v.alias_of), 300)
        self.assertEqual(v.fold("worktree_guard_rejects_complex_commands").pattern, "worktree_guard_rejects_complex_bash")
        self.assertEqual(v.fold("blocked_sleep_polling").pattern, "sleep_polling_blocked")
        self.assertTrue(all(s == s.lower() and " " not in s for s in [*v.ids, *v.alias_of]))
        self.assertIs(M.MODES["traces"].vocabulary.__class__, Vocabulary)
        self.assertIsNone(M.MODES["handoffs"].vocabulary)
        schema = M.label_schema(v)
        item = schema["properties"]["labels"]["items"]
        self.assertEqual(item["properties"]["pattern"]["enum"], v.choices())
        self.assertIn(NEW_PATTERN, item["required"])

    def test_old_free_slug_labels_are_folded_on_load_and_the_folding_is_counted(self):
        v = vocab(("hook_timeout_blocks_tools", "t", ("hook_timeout_blocks_writes",)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in [
                {"id": "a", "pattern": "hook_timeout_blocks_writes", "summary": "s", "harness_fixable": True},
                {"id": "b", "pattern": "hook_timeout_blocks_tools", "summary": "s", "harness_fixable": True},
                {"id": "c", "pattern": "odd_new_thing", "summary": "s", "harness_fixable": False},
                {"id": "d", "pattern": NOISE, "summary": "s", "harness_fixable": False}]) + "\n")
            raw, folded = M.load_labels(out), M.load_labels(out, v)
            self.assertEqual(folded["a"]["pattern"], "hook_timeout_blocks_tools")
            self.assertEqual((folded["a"]["raw"], folded["a"]["canonical"]), ("hook_timeout_blocks_writes", True))
            self.assertEqual((folded["c"]["pattern"], folded["c"]["canonical"]), ("odd_new_thing", False))
            self.assertEqual(X.folding_stats(raw, folded, v),
                             {"labels": 4, "raw_slugs": 3, "canonical_ids": 1, "aliases_folded": 1, "uncanonical": 1,
                              "labels_remapped": 1})

    def test_labeler_takes_a_constrained_choice_or_an_escape(self):
        v = vocab(("hook_timeout_blocks_tools", "t", ("hook_timeout_blocks_writes",)))
        eps = [episode(i, "s", "2026-10-01T10:00:00Z") for i in range(3)]

        def model(system, prompt, schema, out):
            self.assertIn("- hook_timeout_blocks_tools: t", system)
            self.assertEqual(schema["properties"]["labels"]["items"]["properties"]["pattern"]["enum"], v.choices())
            return {"labels": [
                {"id": "ep0000", "pattern": "hook_timeout_blocks_tools", "new_pattern": "", "summary": "s", "harness_fixable": True},
                {"id": "ep0001", "pattern": NEW_PATTERN, "new_pattern": "hook_timeout_blocks_writes", "summary": "s", "harness_fixable": True},
                {"id": "ep0002", "pattern": NEW_PATTERN, "new_pattern": "Brand New Cause", "summary": "s", "harness_fixable": True}]}

        with tempfile.TemporaryDirectory() as tmp:
            labels, failed = M.label(Path(tmp), eps, model, 10, 1, lambda _: None, None, v)
            self.assertEqual(failed, 0)
            self.assertEqual([(labels[e["id"]]["pattern"], labels[e["id"]]["canonical"]) for e in eps],
                             [("hook_timeout_blocks_tools", True), ("hook_timeout_blocks_tools", True), ("brand_new_cause", False)])
            self.assertEqual(labels["ep0002"]["raw"], "brand_new_cause")


class DayBoundaryTest(unittest.TestCase):
    def test_days_close_at_2359_los_angeles_in_both_daylight_and_standard_time(self):
        self.assertEqual(X.la_day("2026-10-02T06:59:59.500Z"), "2026-10-01")  # PDT: 23:59:59.5
        self.assertEqual(X.la_day("2026-10-02T07:00:00Z"), "2026-10-02")
        self.assertEqual(X.la_day("2026-12-02T07:59:59Z"), "2026-12-01")  # PST: 23:59:59
        self.assertEqual(X.la_day("2026-12-02T08:00:00Z"), "2026-12-02")
        self.assertEqual(X.la_day("2026-10-01T12:00:00+02:00"), "2026-10-01")
        self.assertEqual(X.hour_day("2026-10-02T06"), "2026-10-01")
        self.assertEqual(X.hour_day("2026-10-02T07"), "2026-10-02")
        self.assertEqual(X.week_of("2026-10-01"), "2026-09-28")

    def test_a_day_is_open_until_its_last_second_has_passed(self):
        self.assertFalse(X.day_closed("2026-10-01", NOW_OPEN))
        self.assertTrue(X.day_closed("2026-10-01", NOW_CLOSED))
        self.assertTrue(X.day_closed("2026-12-01", datetime(2026, 12, 2, 8, 0, tzinfo=timezone.utc)))
        self.assertFalse(X.day_closed("2026-12-01", datetime(2026, 12, 2, 7, 59, tzinfo=timezone.utc)))


class HostAttributionTest(unittest.TestCase):
    def test_hosts_and_denominators(self):
        transcripts = [X.SessionCalls("a", {"2026-10-01T10": 5}, hook_timeouts=2),  # markers, but harness-launched
                       X.SessionCalls("b", {"2026-10-01T10": 7}, guard_rejections=1),
                       X.SessionCalls("c", {"2026-10-01T10": 9})]
        harness = [X.SessionCalls("a", {"2026-10-01T10": 6}, assignment="asg-1"),
                   X.SessionCalls("d", {"2026-10-01T11": 2}, assignment="asg-2")]  # no transcript at all
        s = X.attribute(transcripts, harness)
        self.assertEqual({k: v.host for k, v in s.items()},
                         {"a": "csf_harness", "b": "desktop_hosted", "c": "cli", "d": "csf_harness"})
        self.assertEqual((s["a"].tool_calls, s["a"].denominator, s["a"].tool_calls_transcript, s["a"].tool_calls_harness),
                         (6, "csf_harness", 5, 6))
        self.assertEqual((s["b"].tool_calls, s["b"].denominator, s["b"].tool_calls_harness), (7, "transcript", None))
        self.assertEqual((s["a"].hook_timeouts, s["b"].guard_rejections, s["d"].tool_calls), (2, 1, 2))

    def test_transcript_facts_fold_subagents_into_their_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "traces-state.json"
            state.write_text(json.dumps({
                "proj/ses-1.jsonl": facts({"2026-10-01T10": 3}, hook=1),
                "proj/ses-1/subagents/agent-1.jsonl": facts({"2026-10-01T10": 2, "2026-10-01T11": 1}, guard=1),
                "proj/ses-2.jsonl": facts({}),
                "legacy/old.jsonl": {"mtime": 1, "size": 1, "hash": "h", "since": "", "episodes": 0, "events": 1}}))
            got = {s.session_id: s for s in X.TranscriptCalls(state).sessions()}
            self.assertEqual(got["ses-1"].calls_by_utc_hour, {"2026-10-01T10": 5, "2026-10-01T11": 1})
            self.assertEqual((got["ses-1"].hook_timeouts, got["ses-1"].guard_rejections), (1, 1))
            self.assertEqual((got["ses-2"].tool_calls, got["old"].tool_calls), (0, 0))

    def test_harness_run_directories_count_tool_use_blocks_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "asg-1"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({"session_id": "ses-a", "agent_id": "x"}))
            lines = [
                {"time": "2026-10-02T06:59:00.000000000Z", "event": {"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "name": "Bash"}, {"type": "text", "text": SENTINEL}, {"type": "tool_use", "name": "Read"}]}}},
                {"time": "2026-10-02T07:00:00.000000000Z", "event": {"type": "assistant", "parent_tool_use_id": "t1",
                                                                     "message": {"content": [{"type": "tool_use", "name": "Edit"}]}}},
                {"time": "2026-10-02T07:00:01Z", "event": {"type": "user", "message": {"content": [{"type": "tool_result"}]}}},
                {"time": "2026-10-02T07:00:02Z", "msg": "session gate decision"},
            ]
            (run / "events.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\nnot json\n")
            (root / "bazel-disk-cache").mkdir()  # not a run directory
            (root / "asg-2").mkdir()
            (root / "asg-2" / "run.json").write_text("{}")  # no session id, no events
            got = list(X.HarnessCalls(root).sessions())
            self.assertEqual(len(got), 1)
            self.assertEqual((got[0].session_id, got[0].assignment), ("ses-a", "asg-1"))
            self.assertEqual(got[0].calls_by_utc_hour, {"2026-10-02T06": 2, "2026-10-02T07": 1})
            self.assertEqual(list(X.HarnessCalls(root / "missing").sessions()), [])


class StatsTest(unittest.TestCase):
    def test_poisson_interval(self):
        self.assertIsNone(X.poisson_ci(3, 0))
        lo, hi = X.poisson_ci(0, 1000)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 3.69, delta=0.05)  # exact Garwood upper bound for k=0: 3.689
        lo, hi = X.poisson_ci(100, 10000)
        self.assertAlmostEqual(lo, 8.14, delta=0.1)   # exact: 8.136
        self.assertAlmostEqual(hi, 12.16, delta=0.1)  # exact: 12.163
        self.assertEqual(X.rate(5, 2000), 2.5)
        self.assertIsNone(X.rate(5, 0))

    def test_trend_verdicts(self):
        halving = [(i, int(800 / 2 ** i), 100000) for i in range(5)]
        t = X.trend(halving)
        self.assertAlmostEqual(t.factor, 0.5, delta=0.02)
        self.assertLess(t.ci95[1], 1)
        self.assertTrue(t.verdict().startswith("Declining"))
        flat = [(i, 100 + (i % 2) * 3, 10000) for i in range(6)]
        t = X.trend(flat)
        self.assertLess(t.ci95[0], 1)
        self.assertGreater(t.ci95[1], 1)
        self.assertTrue(t.verdict().startswith("Flat"))
        self.assertTrue(X.trend([(i, 2 * i + 1, 1000) for i in range(6)]).verdict().startswith("Rising"))
        # Quiet weeks (no struggle or under ACTIVE_CALLS) do not enter; too few weeks means no estimate.
        t = X.trend([(0, 5, 1000), (1, 0, 1000), (2, 5, 100), (3, 4, 1000)])
        self.assertEqual((t.weeks, t.factor), (2, None))
        self.assertTrue(t.verdict().startswith("Not estimable"))


class DailyRecordTest(unittest.TestCase):
    """Three sessions: `ses-a` launched by the CSF harness (also has a
    transcript), `ses-b` Desktop-hosted (a guard rejection), `ses-c` plain
    CLI. Episodes on Oct 1 and at both sides of the Oct 1/2 LA boundary."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.out, self.csf = base / "out", base / "csf"
        self.out.mkdir()
        (self.csf / "asg-1").mkdir(parents=True)
        (self.csf / "asg-1" / "run.json").write_text(json.dumps({"session_id": "ses-a"}))
        (self.csf / "asg-1" / "events.jsonl").write_text("".join(json.dumps(
            {"time": f"2026-10-01T{h:02d}:00:00Z", "event": {"type": "assistant", "message": {"content": [
                {"type": "tool_use"}] * 100}}}) + "\n" for h in (10, 11)))
        (self.out / "traces-state.json").write_text(json.dumps({
            "proj/ses-a.jsonl": facts({"2026-10-01T10": 90, "2026-10-01T11": 90}, hook=3),
            "proj/ses-b.jsonl": facts({"2026-10-01T12": 500, "2026-10-02T06": 500, "2026-10-02T07": 1000}, guard=2),
            "proj/ses-c.jsonl": facts({"2026-10-01T12": 300})}))
        eps = [episode(0, "ses-a", "2026-10-01T10:30:00Z"), episode(1, "ses-a", "2026-10-01T10:40:00Z"),
               episode(2, "ses-b", "2026-10-01T12:00:00Z"), episode(3, "ses-b", "2026-10-02T06:59:59Z"),
               episode(4, "ses-b", "2026-10-02T07:00:00Z"), episode(5, "ses-c", "2026-10-01T12:30:00Z"),
               episode(6, "ses-c", "2026-10-01T12:31:00Z"), episode(7, "ses-c", "2026-10-01T12:32:00Z")]
        (self.out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        labels = [{"id": "ep0000", "pattern": "hook_timeout_blocks_writes", "summary": SENTINEL, "harness_fixable": True},
                  {"id": "ep0001", "pattern": "hook_timeout_blocks_tools", "summary": "s", "harness_fixable": True},
                  {"id": "ep0002", "pattern": "worktree_guard_rejects_complex_commands", "summary": "s", "harness_fixable": True},
                  {"id": "ep0003", "pattern": "worktree_guard_rejects_complex_bash", "summary": "s", "harness_fixable": True},
                  {"id": "ep0004", "pattern": "wrong_python_env", "summary": "s", "harness_fixable": True},
                  {"id": "ep0005", "pattern": NOISE, "summary": "s", "harness_fixable": False},
                  {"id": "ep0006", "pattern": "never_seen_before_cause", "summary": "s", "harness_fixable": True},
                  {"id": "ep0007", "pattern": "iterative_test_fix", "summary": "s", "harness_fixable": False}]
        (self.out / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
        self.schema = json.loads(X.SCHEMA_PATH.read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def run_measure(self, now=NOW_OPEN):
        return X.measure(out=self.out, csf_root=self.csf, skip_miner=True, skip_label=True, now=now, log=lambda _: None)

    def test_daily_records_validate_against_the_schema_and_count_by_host_and_pattern(self):
        got = self.run_measure()
        days = sorted(p.name for p in (self.out / "daily").iterdir())
        self.assertEqual(days, ["2026-10-01.json", "2026-10-02.json"])
        d1 = json.loads((self.out / "daily" / "2026-10-01.json").read_text())
        d2 = json.loads((self.out / "daily" / "2026-10-02.json").read_text())
        for d in (d1, d2):
            jsonschema.validate(d, self.schema)
        # Oct 1: ses-a counts by its harness record (200, not the transcript's 180); ses-b 1000 (500 + 500
        # in the 06Z hour, still Oct 1 in LA); ses-c 300.
        self.assertEqual((d1["tool_calls"], d1["sessions"]), (1500, 3))
        # Episodes: 7 on Oct 1 (ep0003 at 06:59:59Z is still Oct 1); ep0005 noise, ep0007 not fixable.
        self.assertEqual((d1["episodes"], d1["labelled"], d1["noise"], d1["struggles"]), (7, 7, 1, 5))
        self.assertEqual(d1["rate_per_1k"], round(1000 * 5 / 1500, 3))
        self.assertEqual(d1["hosts"]["csf_harness"], {"sessions": 1, "tool_calls": 200, "struggles": 2,
                                                       "rate_per_1k": 10.0, "ci95": X.poisson_ci(2, 200)})
        self.assertEqual((d1["hosts"]["desktop_hosted"]["tool_calls"], d1["hosts"]["desktop_hosted"]["struggles"]), (1000, 2))
        self.assertEqual((d1["hosts"]["cli"]["tool_calls"], d1["hosts"]["cli"]["struggles"]), (300, 1))
        self.assertEqual(d1["patterns"]["hook_timeout_blocks_tools"], {"episodes": 2, "harness_fixable": 2, "hosts": {"csf_harness": 2}})
        self.assertEqual(d1["patterns"]["worktree_guard_rejects_complex_bash"]["hosts"], {"desktop_hosted": 2})
        self.assertEqual(d1["patterns"]["iterative_new_test_fixes"], {"episodes": 1, "harness_fixable": 0, "hosts": {"cli": 1}})
        self.assertEqual(d1["uncanonical"], {"never_seen_before_cause": 1})
        self.assertFalse(d1["closed"])
        # Oct 2: one episode (ep0004 at 07:00:00Z), 1000 calls, open.
        self.assertEqual((d2["tool_calls"], d2["episodes"], d2["struggles"], d2["closed"]), (1000, 1, 1, False))
        self.assertEqual(d2["patterns"], {"host_toolchain_missing": {"episodes": 1, "harness_fixable": 1, "hosts": {"desktop_hosted": 1}}})
        self.assertEqual(got["receipt"]["sessions"], {"csf_harness": 1, "desktop_hosted": 1, "cli": 1})
        self.assertEqual(got["receipt"]["days"], {"written": 2, "closed": 0, "open": 2})
        self.assertEqual(got["receipt"]["labels"]["aliases_folded"], 4)
        self.assertEqual(got["receipt"]["labels"]["uncanonical"], 1)
        self.assertEqual(got["receipt"]["outcome"], "ok")

    def test_a_closed_day_stays_closed_and_an_open_day_is_recomputed(self):
        self.run_measure(NOW_CLOSED)
        d1 = json.loads((self.out / "daily" / "2026-10-01.json").read_text())
        d2 = json.loads((self.out / "daily" / "2026-10-02.json").read_text())
        self.assertEqual((d1["closed"], d2["closed"]), (True, False))
        series = [json.loads(l) for l in (self.out / "series.jsonl").read_text().splitlines()]
        rate_all = [s for s in series if s["series"] == X.SERIES_RATE and s["host"] == "all"]
        self.assertEqual([(s["day"], s["closed"], s["struggles"]) for s in rate_all],
                         [("2026-10-01", True, 5), ("2026-10-02", False, 1)])
        self.assertTrue(all(s["unit"] == "episodes per 1000 tool calls" for s in rate_all))
        eps = [s for s in series if s["series"] == X.SERIES_EPISODES]
        self.assertIn({"series": X.SERIES_EPISODES, "unit": "episodes", "day": "2026-10-01", "closed": True,
                       "computed_at": d1["computed_at"], "host": "csf_harness", "pattern": "hook_timeout_blocks_tools",
                       "episodes": 2}, eps)
        self.assertTrue(all(set(s) <= {"series", "unit", "day", "closed", "computed_at", "host", "pattern", "episodes"} for s in eps))

    def test_a_measured_zero_is_not_missing_data(self):
        # Oct 3: tool calls and no episodes (a measured zero); Oct 4: an episode and no tool calls (no denominator).
        state = json.loads((self.out / "traces-state.json").read_text())
        state["proj/ses-c.jsonl"]["facts"]["tool_calls_by_utc_hour"]["2026-10-03T12"] = 400
        (self.out / "traces-state.json").write_text(json.dumps(state))
        with (self.out / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(episode(8, "ses-c", "2026-10-04T12:00:00Z")) + "\n")
        with (self.out / "labels.jsonl").open("a") as f:
            f.write(json.dumps({"id": "ep0008", "pattern": "wrong_python_env", "summary": "s", "harness_fixable": True}) + "\n")
        self.run_measure(datetime(2026, 10, 9, tzinfo=timezone.utc))
        d3 = json.loads((self.out / "daily" / "2026-10-03.json").read_text())
        d4 = json.loads((self.out / "daily" / "2026-10-04.json").read_text())
        jsonschema.validate(d3, self.schema)
        jsonschema.validate(d4, self.schema)
        self.assertEqual((d3["tool_calls"], d3["struggles"], d3["rate_per_1k"]), (400, 0, 0.0))
        self.assertGreater(d3["ci95"][1], 0)
        self.assertEqual((d4["tool_calls"], d4["struggles"], d4["rate_per_1k"], d4["ci95"]), (0, 1, None, None))

    def test_unlabelled_episodes_are_counted_not_scored(self):
        with (self.out / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(episode(9, "ses-c", "2026-10-01T13:00:00Z")) + "\n")
        got = self.run_measure()
        d1 = json.loads((self.out / "daily" / "2026-10-01.json").read_text())
        self.assertEqual((d1["episodes"], d1["labelled"], d1["struggles"]), (8, 7, 5))
        self.assertEqual(got["receipt"]["labels"]["unlabelled"], 1)
        self.assertIn("Unlabelled episodes: 1", got["report"])

    def test_report_leads_with_the_verdict_and_quotes_no_transcript(self):
        got = self.run_measure()
        text = (self.out / "MEASURE.md").read_text()
        self.assertEqual(text, got["report"])
        self.assertTrue(text.splitlines()[2].startswith("**Verdict:**"), text.splitlines()[:3])
        self.assertIn("| 2026-09-28 | 3 | 2500 | 6 |", text)  # the one week: all calls, all struggles
        self.assertIn("| csf_harness | 1 | 200 | 2 | 10.0 |", text)
        self.assertIn("| desktop_hosted | 1 | 2000 | 3 | 1.5 |", text)  # both days, whole corpus
        self.assertIn("| cli | 1 | 300 | 1 |", text)
        self.assertIn("`hook_timeout_blocks_tools` | 2 | 1 |", text)
        self.assertIn("4 slugs folded through aliases, 1 left uncanonical", text)
        self.assertIn("harness records count 200 tool calls against 180 in the transcripts (180 in common)", text)
        for path in [self.out / "MEASURE.md", self.out / "series.jsonl", self.out / "measure-run.json",
                     *(self.out / "daily").iterdir()]:
            self.assertNotIn(SENTINEL, path.read_text(), path)
        receipt = json.loads((self.out / "measure-run.json").read_text())
        self.assertEqual(receipt["procedure"], "python -m rrsi harness measure")
        self.assertTrue(receipt["evidence"]["report"].endswith("MEASURE.md"))

    def test_output_inside_a_work_tree_is_refused(self):
        repo = Path(self.tmp.name) / "repo"
        (repo / ".git").mkdir(parents=True)
        with self.assertRaises(M.PrivacyError):
            X.measure(out=repo / "private", csf_root=self.csf, skip_miner=True, skip_label=True, log=lambda _: None)

    def test_cli_runs_the_measurement(self):
        from rrsi import __main__ as cli
        rc = cli.main(["harness", "measure", "--out", str(self.out), "--csf-root", str(self.csf), "--skip-miner",
                       "--skip-label", "--now", "2026-10-02T07:00:00Z", "--quiet"])
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads((self.out / "daily" / "2026-10-01.json").read_text())["closed"])


if __name__ == "__main__":
    unittest.main()
