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
"""LLM telemetry miner: recorded, anonymized Langfuse observations and LiteLLM
request-log rows (tests/fixtures/langfuse). The end-to-end test runs this
checkout's `rrsi-mine` when it is built.

    python3 -m pytest -q tests/test_langfuse_miner.py
"""

import json
import sys
import tempfile
import unittest
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi.harness import langfuse_miner as L  # noqa: E402
from rrsi.harness import mine as M  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "langfuse"
LIMITS = {"fast": 262144, "deep": 262144}
S = L.Signal


def load(name):
    return json.loads((FIX / name).read_text())


def projected_rows():
    return [L.project_litellm(r, "hash-self") for r in load("litellm_spend_logs.json")]


def projected_obs(name):
    return [L.project_langfuse(o) for o in load(name)]


def hits_of(rows):
    """request id -> sorted signal names, over every gateway session."""
    out = {}
    for rs in L.gateway_sessions(rows):
        for i, sigs in L.gateway_hits(rs, LIMITS).items():
            out[rs[i]["request_id"]] = sorted(s.value for s in sigs)
    return out


def req(rid, start, status="success", pt=1000, ct=10, first="", key="self", client="OpenAI", group="fast",
        code="", cls="", error="", retries=0, call="acompletion"):
    return {"request_id": rid, "start": start, "end": start, "first_token": first, "duration_ms": 0,
            "call_type": call, "model_group": group, "model": group, "status": status, "prompt_tokens": pt,
            "completion_tokens": ct, "client": client, "key": key, "error_code": code, "error_class": cls,
            "error": error, "attempted_retries": retries}


class LiteLLMProjection(unittest.TestCase):
    def test_rows_lose_every_caller_identity(self):
        for raw in load("litellm_spend_logs.json"):
            blob = json.dumps(L.project_litellm(raw, "hash-self"))
            md = raw["metadata"]
            for ident in (raw["api_key"], raw["user"], raw["requester_ip_address"], raw["team_id"],
                          raw["session_id"], raw["api_base"], md["user_api_key_alias"],
                          md["user_api_key_user_id"], md["user_api_key_team_alias"]):
                if ident != "":
                    self.assertNotIn(ident, blob)

    def test_key_labels_separate_callers_without_naming_them(self):
        self.assertEqual(L.key_label("hash-self", "mine", "hash-self"), L.SELF_KEY)
        a1, a2 = L.key_label("h1", "agents-key", "hash-self"), L.key_label("h2", "agents-key", "hash-self")
        b = L.key_label("h3", "other-key", "hash-self")
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)
        self.assertNotIn("agents", a1)
        # No alias: the key hash names the caller, still hashed again.
        self.assertNotIn("h4", L.key_label("h4", "", "hash-self")[2:])

    def test_error_text_is_scrubbed(self):
        auth = [r for r in projected_rows() if r["error_code"] == "401"][0]["error"]
        self.assertIn("user_id=<id>", auth)
        for leak in ("ab12", "10.11.12.13", "gw.example.corp"):
            self.assertNotIn(leak, auth)

    def test_client_is_the_user_agent_family(self):
        self.assertEqual(L.client_of(["User-Agent: OpenAI", "User-Agent: OpenAI/JS 6.1.0", "team:x"]), "OpenAI")
        self.assertEqual(L.client_of(["team:x"]), "")


class FailureKinds(unittest.TestCase):
    def test_first_matching_rule_wins(self):
        cases = [
            ("403", "ProxyException", "key not allowed to access model. Tried to access gateway-selftest",
             S.MODEL_ACCESS_DENIED),
            ("400", "BadRequestError", "Invalid model name passed in model=gpt-5-mini", S.MODEL_ACCESS_DENIED),
            ("400", "ContextWindowExceededError", "", S.CONTEXT_OVERFLOW),
            ("400", "BadRequestError", "This model's maximum context length is 262144 tokens", S.CONTEXT_OVERFLOW),
            ("401", "ProxyException", "Authentication Error", S.AUTH_ERROR),
            ("403", "ProxyException", "team budget blocked", S.AUTH_ERROR),
            ("429", "RateLimitError", "", S.RATE_LIMITED),
            ("408", "", "", S.LLM_TIMEOUT),
            ("500", "Timeout", "Request timed out", S.LLM_TIMEOUT),
            ("400", "BadRequestError", "tools.0.function.parameters is invalid", S.BAD_REQUEST),
            ("422", "", "", S.BAD_REQUEST),
            ("502", "", "", S.UPSTREAM_ERROR),
            ("", "APIConnectionError", "", S.UPSTREAM_ERROR),
            ("", "", "something new", S.LLM_ERROR),
        ]
        for code, cls, msg, want in cases:
            with self.subTest(code=code, cls=cls, msg=msg):
                self.assertEqual(L.failure_kind(code, cls, msg), want)


class GatewayDetectors(unittest.TestCase):
    def test_recorded_rows(self):
        self.assertEqual(hits_of(projected_rows()), {
            "chatcmpl-a2": ["context_overflow"],
            "chatcmpl-a3": ["context_pressure", "llm_retry"],
            "chatcmpl-a4": ["empty_completion"],
            "chatcmpl-a5": ["slow_first_token"],
            "chatcmpl-a6": ["gateway_retry"],
            "b1-selftest": ["model_access_denied"],
            "b2-selftest": ["llm_retry", "model_access_denied"],
            "b3-authfail": ["auth_error", "llm_retry"],
        })

    def test_sessions_split_by_stream_idle_gap_and_utc_midnight(self):
        rows = [req("r1", "2026-09-28T23:50:00+00:00"), req("r2", "2026-09-29T00:05:00+00:00"),
                req("r3", "2026-09-29T00:20:00+00:00"), req("r4", "2026-09-29T00:51:00+00:00"),
                req("r5", "2026-09-29T00:21:00+00:00", key="k-1"),
                req("r6", "2026-09-29T00:22:00+00:00", group="deep")]
        got = sorted([r["request_id"] for r in s] for s in L.gateway_sessions(rows))
        self.assertEqual(got, [["r1"], ["r2", "r3"], ["r4"], ["r5"], ["r6"]])

    def test_retry_needs_a_failed_twin_inside_the_window(self):
        fail = req("f", "2026-09-28T10:00:00+00:00", status="failure", pt=50000, code="500")
        cases = [("same size, 1 min later", req("x", "2026-09-28T10:01:00+00:00", pt=50100), True),
                 ("same size, 3 min later", req("x", "2026-09-28T10:03:00+00:00", pt=50000), False),
                 ("another conversation", req("x", "2026-09-28T10:01:00+00:00", pt=20000), False)]
        for name, nxt, want in cases:
            with self.subTest(name):
                got = L.gateway_hits([fail, nxt], LIMITS).get(1, [])
                self.assertEqual(S.LLM_RETRY in got, want)
        ok = req("o", "2026-09-28T10:00:00+00:00", pt=50000)
        self.assertNotIn(1, L.gateway_hits([ok, req("x", "2026-09-28T10:00:30+00:00", pt=50000)], LIMITS))

    def test_a_failures_only_stream_infers_no_retries(self):
        a = req("f1", "2026-09-28T10:00:00+00:00", status="failure", code="403", error="not allowed to access model")
        b = req("f2", "2026-09-28T10:01:00+00:00", status="failure", code="403", error="not allowed to access model")
        self.assertEqual(L.gateway_hits([a, b], LIMITS, complete=False),
                         {0: [S.MODEL_ACCESS_DENIED], 1: [S.MODEL_ACCESS_DENIED]})
        self.assertIn(S.LLM_RETRY, L.gateway_hits([a, b], LIMITS, complete=True)[1])

    def test_context_pressure_is_relative_to_the_model_limit(self):
        at = int(L.CONTEXT_PRESSURE * LIMITS["fast"]) + 1
        cases = [("at the threshold", req("x", "2026-09-28T10:00:00+00:00", pt=at), True),
                 ("below it", req("x", "2026-09-28T10:00:00+00:00", pt=at - 2), False),
                 ("unknown model", req("x", "2026-09-28T10:00:00+00:00", pt=at, group="other"), False)]
        for name, r, want in cases:
            with self.subTest(name):
                self.assertEqual(S.CONTEXT_PRESSURE in L.gateway_hits([r], LIMITS).get(0, []), want)

    def test_episodes_merge_one_burst_per_session(self):
        eps = L.litellm_episodes(projected_rows(), LIMITS, "litellm-b300")
        sig = {e["refs"][0]: e["signals"] for e in eps}
        self.assertEqual(sig, {
            "chatcmpl-a2": {"context_overflow": 1, "context_pressure": 1, "empty_completion": 1,
                            "gateway_retry": 1, "llm_retry": 1, "slow_first_token": 1},
            "b1-selftest": {"auth_error": 1, "llm_retry": 2, "model_access_denied": 2},
        })
        self.assertTrue(all(e["project"] == "litellm-b300" for e in eps))
        self.assertEqual({e["counts"]["tool_errors"] for e in eps}, {1, 3})


class Assembly(unittest.TestCase):
    def test_fnv64_reference_vectors(self):
        self.assertEqual(L.fnv64(b""), 0xCBF29CE484222325)
        self.assertEqual(L.fnv64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(L.fnv64(b"foobar"), 0x85944171F73967E8)

    def evs(self, n):
        return [L.Ev(ts=f"t{i}", kind="llm_request", text=f"e{i}", ref=f"r{i}", call=True) for i in range(n)]

    def test_hits_merge_within_the_rrsi_mine_gap(self):
        evs = self.evs(30)
        one = L.assemble(evs, {5: [S.LLM_ERROR], 5 + L.EPISODE_GAP: [S.LLM_RETRY]}, project="p", session_id="s", file="f")
        two = L.assemble(evs, {5: [S.LLM_ERROR], 6 + L.EPISODE_GAP: [S.LLM_RETRY]}, project="p", session_id="s", file="f")
        self.assertEqual(len(one), 1)
        self.assertEqual(len(two), 2)
        e = one[0]
        self.assertEqual([c["i"] for c in e["context"]],
                         list(range(5 - L.CONTEXT_BEFORE, 5 + L.EPISODE_GAP + L.CONTEXT_AFTER + 1)))
        self.assertEqual(e["id"], f"{L.fnv64(b'f#5'):016x}")
        self.assertEqual((e["refs"], e["examples"]), (["r5", "r11"], {"llm_error": "r5", "llm_retry": "r11"}))
        self.assertNotIn("agent_id", e)

    def test_long_episode_keeps_head_and_tail_context(self):
        evs = self.evs(80)
        hits = {i: [S.LLM_ERROR] for i in range(5, 61, 5)}
        (e,) = L.assemble(evs, hits, project="p", session_id="s", file="f")
        idx = [c["i"] for c in e["context"]]
        self.assertEqual(len(idx), L.CONTEXT_MAX)
        self.assertEqual(idx[0], 5 - L.CONTEXT_BEFORE)
        self.assertEqual(idx[-1], 60 + L.CONTEXT_AFTER)
        self.assertEqual(e["counts"]["span_events"], 56)
        self.assertEqual(len(e["refs"]), L.REFS_MAX)


class LangfuseSessions(unittest.TestCase):
    def test_a_session_exported_twice_counts_once(self):
        obs = L.dedupe(projected_obs("observations_bridge.json"))
        kinds = [L.kind_of(o) for o in obs]
        self.assertEqual(kinds.count("tool_use"), 3)
        self.assertEqual(kinds.count("tool_result"), 3)
        self.assertEqual(kinds.count("human"), 2)
        self.assertEqual(set(L.langfuse_sessions(obs)), {"5e55a0f1-0000-4000-8000-00000000c0de"})

    def test_transcript_marks_failed_calls_and_splits_subagents(self):
        obs = L.dedupe(projected_obs("observations_bridge.json"))
        lines = L.transcript_lines(obs)
        self.assertEqual(set(lines), {"", "call_task1"})
        results = [b for l in lines[""] for b in (l["message"]["content"] if isinstance(l["message"]["content"], list) else [])
                   if b["type"] == "tool_result"]
        self.assertEqual([(b["tool_use_id"], b["is_error"]) for b in results], [("call_c1", True), ("call_c2", False)])
        # The bridge drops a message's source: rrsi-mine's plain-prompt rule decides, as for Copilot
        # messages without one.
        users = [l for l in lines[""] if l["type"] == "user" and isinstance(l["message"]["content"], str)]
        self.assertEqual([l.get("origin") for l in users], [None, None])
        self.assertTrue(all(l["isSidechain"] is True for l in lines["call_task1"]))
        inputs = [b["input"] for l in lines[""] for b in (l["message"]["content"] if isinstance(l["message"]["content"], list) else [])
                  if b["type"] == "tool_use"]
        self.assertEqual(len({json.dumps(i) for i in inputs}), len(inputs))

    def test_exported_message_text_is_redacted(self):
        o = L.project_langfuse({"id": "o1", "traceId": "t", "type": "EVENT", "name": "copilot.message:user",
                                "startTime": "2026-09-05T10:00:00Z", "metadata": {},
                                "input": "LANGFUSE_PUBLIC_KEY=pk-lf-1234abcd-0000-4000 LANGFUSE_SECRET_KEY="
                                         "sk-lf-9999ffff-1111-4000 see /home/someone/notes and gw.example.corp"})
        for leak in ("pk-lf-1234abcd", "sk-lf-9999ffff", "someone", "gw.example.corp"):
            self.assertNotIn(leak, o["input"])

    def test_a_failed_call_is_read_from_the_success_flag_or_the_level(self):
        def complete(level, md):
            return L.project_langfuse({"id": "o1", "traceId": "t", "type": "EVENT", "name": "copilot.tool.complete",
                                       "level": level, "startTime": "2026-09-05T10:00:00Z",
                                       "metadata": {"toolCallId": "c1", **md}})
        cases = [("success false, default level", complete("DEFAULT", {"success": False}), True),
                 ("error level, no flag", complete("ERROR", {}), True),
                 ("success true", complete("DEFAULT", {"success": True}), False)]
        for name, o, want in cases:
            with self.subTest(name):
                (line,) = L.transcript_lines([o])[""]
                self.assertEqual(line["message"]["content"][0]["is_error"], want)
                self.assertEqual(L.langfuse_event("tool_result", o).is_error, want)

    def test_generation_and_turn_classes_in_both_dialects(self):
        bridge = L.langfuse_episodes(L.dedupe(projected_obs("observations_bridge.json")), "copilot")
        otel = L.langfuse_episodes(L.dedupe(projected_obs("observations_otel.json")), "csf")
        self.assertEqual([e["signals"] for e in bridge], [{"llm_truncated": 1, "turn_failed": 1}])
        self.assertEqual([e["signals"] for e in otel], [{"llm_error": 1, "llm_truncated": 1}])
        self.assertEqual(bridge[0]["user_turn"], "no, that's wrong - use the bazel target")
        self.assertEqual(bridge[0]["refs"], ["1f00c0de1f00c0de1f00c0de1f00c0d1"])


EXE = M.CRATE / "target" / "release" / "rrsi-mine"


@unittest.skipUnless(EXE.exists(), "rrsi-mine not built (cargo build --release in tools/rrsi-mine)")
class EndToEnd(unittest.TestCase):
    def test_rrsi_mine_traces_reads_the_projection(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw"
            raw.mkdir()
            rows = projected_rows()
            (raw / "litellm-self.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["key"] == "self"))
            (raw / "litellm-failures.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows if r["status"] == L.FAILURE))
            (raw / "litellm-counts.json").write_text(json.dumps({"total": len(rows)}))
            (raw / "litellm-models.json").write_text(json.dumps(LIMITS))
            for name, label in (("observations_bridge.json", "copilot"), ("observations_otel.json", "csf")):
                (raw / f"langfuse-{label}.jsonl").write_text(
                    "".join(json.dumps(o) + "\n" for o in projected_obs(name)))
            out = Path(d) / "out" / "episodes.jsonl"
            summary = L.mine_raw(raw, out, Path(d) / "work", EXE, "b300")
            eps = [json.loads(l) for l in out.read_text().splitlines()]
        traced = {(e["project"], e["subagent"]): e["signals"] for e in eps if not e["file"].endswith(".generations")
                  and e["project"].startswith("langfuse-")}
        self.assertEqual(traced, {("langfuse-copilot", False): {"tool_error": 1, "user_correction": 1},
                                  ("langfuse-copilot", True): {"tool_error": 1},
                                  ("langfuse-csf", False): {"tool_error": 1}})
        rust = next(e for e in eps if e["project"] == "langfuse-copilot" and not e["file"].endswith(".generations"))
        ours = next(e for e in eps if e["project"] == "litellm-b300-self")
        self.assertEqual(set(rust), set(ours))
        self.assertEqual(set(rust["counts"]), set(ours["counts"]))
        src = summary["sources"]
        self.assertEqual(src["langfuse-copilot"]["observations"], 13)
        self.assertEqual((src["litellm-b300-self"]["observations"], src["litellm-b300-others"]["observations"]), (7, 3))
        self.assertEqual(src["litellm-b300-self"]["classes"]["llm_retry"],
                         {"episodes": 1, "hits": 1, "examples": ["chatcmpl-a3"]})
        self.assertEqual(src["litellm-b300-others"]["classes"],
                         {"model_access_denied": {"episodes": 1, "hits": 2, "examples": ["b1-selftest"]},
                          "auth_error": {"episodes": 1, "hits": 1, "examples": ["b3-authfail"]}})

    def test_injected_text_is_not_an_operator_turn(self):
        obs = [{"id": f"o{i}", "traceId": "t1", "sessionId": "s1", "type": typ, "name": name, "level": "DEFAULT",
                "statusMessage": sm, "startTime": f"2026-09-05T10:00:0{i}Z", "parent": "",
                "metadata": md, **io} for i, (typ, name, sm, md, io) in enumerate([
                    ("TOOL", "copilot.tool:bash", "", {"toolCallId": "c1", "toolName": "bash"}, {}),
                    ("EVENT", "copilot.tool.complete", "success", {"toolCallId": "c1", "success": True}, {}),
                    ("EVENT", "copilot.message:user", "", {}, {"input": "[Scheduled check] no output yet"}),
                    ("EVENT", "copilot.message:assistant", "", {}, {"output": "Checking the build again."}),
                    ("EVENT", "copilot.message:user", "", {}, {"input": "no, wrong file"})])]
        with tempfile.TemporaryDirectory() as d:
            eps, _ = L.traces_episodes(obs, "x", Path(d), EXE)
        self.assertEqual([e["signals"] for e in eps], [{"user_correction": 1}])
        self.assertEqual([c["text"] for c in eps[0]["context"] if "signals" in c], ["no, wrong file"])


class Ranking(unittest.TestCase):
    def test_rows_ranked_by_episodes_with_their_source_denominator(self):
        eps = [{"project": "a", "signals": {"tool_error": 2}, "refs": ["t1"]},
               {"project": "a", "signals": {"tool_error": 1, "retry": 1}, "refs": ["t2"]},
               {"project": "a", "signals": {"tool_error": 1}, "refs": ["t3"]},
               {"project": "b", "signals": {"llm_retry": 3}, "refs": ["r1"]},
               {"project": "b", "signals": {"llm_retry": 1}, "refs": ["r1"]}]
        rows = L.ranked(L.summarize(eps, {"a": 1000, "b": 500}))
        self.assertEqual([(r["class"], r["source"], r["episodes"], r["hits"], r["rate_per_1k"]) for r in rows],
                         [("tool_error", "a", 3, 4, 3.0), ("llm_retry", "b", 2, 4, 4.0), ("retry", "a", 1, 1, 1.0)])
        self.assertEqual(rows[0]["examples"], ["t1", "t2"])
        self.assertEqual(rows[1]["examples"], ["r1"])
        self.assertNotEqual(rows[0]["fix"], "")
        self.assertEqual(L.table(rows).count("\n"), len(rows) + 1)

    def test_every_class_has_a_candidate_fix(self):
        transcript = ["tool_error", "retry", "test_failure", "permission_denial", "hook_timeout",
                      "user_interrupt", "user_correction", "reask", "silence"]
        self.assertEqual(sorted(L.FIXES), sorted(transcript + [s.value for s in S]))


class Fetch(unittest.TestCase):
    ROWS = [{"request_id": f"q{i}", "startTime": t, "metadata": {"user_api_key_alias": "a"}, "api_key": "hash-self"}
            for i, t in enumerate(["2026-09-28T10:01:00+00:00", "2026-09-28T10:05:00+00:00",
                                   "2026-09-28T10:07:00+00:00", "2026-09-28T10:40:00+00:00",
                                   "2026-09-28T11:30:00+00:00"])]
    CAP = 2

    def fake(self, calls):
        def get(url, headers):
            calls.append((url, headers))
            u = urllib.parse.urlparse(url)
            if u.path == "/key/info":
                return {"key": "hash-self"}
            if u.path == "/model/info":
                return {"data": [{"model_name": "fast", "model_info": {"max_input_tokens": 262144}},
                                 {"model_name": "fast", "model_info": {"max_input_tokens": 131072}},
                                 {"model_name": "deep", "model_info": {"max_input_tokens": None}}]}
            q = dict(urllib.parse.parse_qsl(u.query))
            a = datetime.strptime(q["start_date"], L.LITELLM_DATE).replace(tzinfo=timezone.utc)
            b = datetime.strptime(q["end_date"], L.LITELLM_DATE).replace(tzinfo=timezone.utc)
            rows = [r for r in self.ROWS if a <= datetime.fromisoformat(r["startTime"]) <= b
                    and q.get("api_key", r["api_key"]) == r["api_key"]
                    and q.get("status_filter", r.get("status", "success")) == r.get("status", "success")]
            size, page = int(q["page_size"]), int(q["page"])
            capped = len(rows) > self.CAP
            pages = -(-len(rows) // size)
            return {"data": [] if capped else rows[(page - 1) * size:page * size], "total": len(rows),
                    "total_pages": pages, "total_is_capped": capped}
        return get

    def test_capped_windows_split_and_every_row_lands_once(self):
        calls, logs = [], []
        with tempfile.TemporaryDirectory() as d:
            old = L.LITELLM_PAGE
            L.LITELLM_PAGE = 1
            try:
                r = L.fetch_litellm("http://gw", "sk-secret-key", datetime(2026, 9, 28, 10, tzinfo=timezone.utc),
                                    datetime(2026, 9, 28, 12, tzinfo=timezone.utc), Path(d) / "raw", L.Scope.SELF,
                                    jobs=2, get=self.fake(calls), log=logs.append)
            finally:
                L.LITELLM_PAGE = old
            got = [json.loads(l)["request_id"] for l in
                   (Path(d) / "raw" / "litellm-self.jsonl").read_text().splitlines()]
            limits = json.loads((Path(d) / "raw" / "litellm-models.json").read_text())
        self.assertEqual(got, ["q0", "q1", "q2", "q3", "q4"])
        self.assertEqual(r["requests"], 5)
        self.assertEqual(limits, {"fast": 131072})
        self.assertTrue(any("capped" in m for m in logs))
        self.assertTrue(all("sk-secret-key" not in u for u, _ in calls))
        self.assertTrue(all("sk-secret-key" not in m for m in logs))
        logs_queries = [dict(urllib.parse.parse_qsl(urllib.parse.urlparse(u).query)) for u, _ in calls
                        if urllib.parse.urlparse(u).path == L.LITELLM_LOGS]
        self.assertTrue(all(q["api_key"] == "hash-self" and "status_filter" not in q for q in logs_queries))

    def test_failure_scope_and_counts(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw"
            L.fetch_litellm("http://gw", "k", datetime(2026, 9, 28, 10, tzinfo=timezone.utc),
                            datetime(2026, 9, 28, 12, tzinfo=timezone.utc), raw, L.Scope.FAILURES,
                            get=self.fake(calls), log=lambda m: None)
            counts = L.count_litellm("http://gw", "k", datetime(2026, 9, 28, 10, tzinfo=timezone.utc),
                                     datetime(2026, 9, 28, 12, tzinfo=timezone.utc), raw,
                                     get=self.fake(calls), log=lambda m: None)
            written = json.loads((raw / "litellm-counts.json").read_text())
            self.assertTrue((raw / "litellm-failures.jsonl").exists())
        queries = [dict(urllib.parse.parse_qsl(urllib.parse.urlparse(u).query)) for u, _ in calls
                   if urllib.parse.urlparse(u).path == L.LITELLM_LOGS]
        self.assertTrue(all(q.get("status_filter") == "failure" for q in queries if q["page_size"] != "1"))
        self.assertEqual((counts["total"], written["total"]), (5, 5))

    def test_langfuse_pages_follow_the_cursor(self):
        pages = {"": {"data": [{"id": "o1"}], "meta": {"cursor": "c2"}},
                 "c2": {"data": [{"id": "o2"}], "meta": {"cursor": None}}}
        seen = []

        def get(url, headers):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            seen.append(q.get("cursor", ""))
            self.assertTrue(headers["Authorization"].startswith("Basic "))
            return pages[q.get("cursor", "")]
        with tempfile.TemporaryDirectory() as d:
            r = L.fetch_langfuse("http://lf", "pk", "sk", "x", Path(d), get=get, log=lambda m: None)
            ids = [json.loads(l)["id"] for l in (Path(d) / "langfuse-x.jsonl").read_text().splitlines()]
        self.assertEqual((seen, ids, r["observations"]), (["", "c2"], ["o1", "o2"], 2))


if __name__ == "__main__":
    unittest.main()
