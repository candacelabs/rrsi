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
"""Struggle episodes from LLM telemetry, in the `rrsi-mine traces` episode schema.

Two read-only sources:

- **Langfuse** (public API v2 observations, so v4 events-only deployments
  work): GitHub Copilot CLI telemetry in either dialect it reaches Langfuse
  in, the local session bridge (`copilot.message:*`, `copilot.tool:*`,
  `copilot.tool.complete`, `copilot.model:*`, `copilot.agent.turn`) or the
  CLI's own OpenTelemetry GenAI spans (`invoke_agent`, `chat <model>`, TOOL
  spans). Each session is projected onto Claude Code's transcript layout, as
  `rrsi-mine copilot-transcripts` does for `events.jsonl`, so the nine
  detectors of `rrsi-mine traces` run unchanged on it. The classes a
  transcript cannot carry (`llm_truncated`, `llm_error`, `turn_failed`) are
  detected here.
- **LiteLLM gateway request logs** (`/spend/logs/v2`, the list view: request
  metadata, never prompt text): one event per model request. Requests are
  grouped into streams (key, client, model group), split into sessions at
  idle gaps and UTC days, and the gateway classes (failures by kind, a client
  re-sending a failed request, router retries, context pressure, empty
  completions, slow first tokens) are detected here. A busy gateway logs more
  than a crawl can page through, so the crawl is scoped: every request of our
  own key (complete streams, every class), the failed requests of every key
  (failure kinds only), and per-window request totals (the denominator).

    python -m rrsi.harness.langfuse_miner fetch-litellm --scope self|failures --base URL --since D --until D --raw DIR
    python -m rrsi.harness.langfuse_miner count-litellm --base URL --since 2026-09-15 --until 2026-10-05 --raw DIR
    python -m rrsi.harness.langfuse_miner fetch-langfuse --host URL --label NAME --raw DIR
    python -m rrsi.harness.langfuse_miner episodes --raw DIR --out EPISODES.jsonl
    python -m rrsi.harness.langfuse_miner table --summary EPISODES.summary.json

Credentials come only from the environment (`LITELLM_API_KEY`;
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`) and are never written or
logged; every request is a GET. The LiteLLM projection drops caller identity
(addresses, user ids, key hashes; a key alias becomes `self` or a short hash)
before anything is written. DIR, the work directory and the episodes quote
private telemetry and must be outside every git work tree (enforced).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Callable, Iterator

from rrsi.harness import mine as M

# --------------------------------------------------------------- schema

#: Episode assembly constants of `rrsi-mine traces` (miners/traces.rs).
EPISODE_GAP = 6
CONTEXT_BEFORE = 3
CONTEXT_AFTER = 2
CONTEXT_MAX = 24
TEXT_MAX = 400
USER_TURN_MAX = 800
#: Example ids (request ids, trace ids) kept per episode.
REFS_MAX = 5

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64 = (1 << 64) - 1


class Signal(str, Enum):
    """The classes detected here. The nine transcript classes (`tool_error`,
    `retry`, `test_failure`, ...) come from `rrsi-mine traces`."""

    # A gateway request failed, by kind.
    MODEL_ACCESS_DENIED = "model_access_denied"
    AUTH_ERROR = "auth_error"
    CONTEXT_OVERFLOW = "context_overflow"
    RATE_LIMITED = "rate_limited"
    LLM_TIMEOUT = "llm_timeout"
    BAD_REQUEST = "bad_request"
    UPSTREAM_ERROR = "upstream_error"
    LLM_ERROR = "llm_error"
    # Gateway request patterns.
    LLM_RETRY = "llm_retry"
    GATEWAY_RETRY = "gateway_retry"
    CONTEXT_PRESSURE = "context_pressure"
    EMPTY_COMPLETION = "empty_completion"
    SLOW_FIRST_TOKEN = "slow_first_token"
    # Langfuse generations and agent turns.
    LLM_TRUNCATED = "llm_truncated"
    TURN_FAILED = "turn_failed"


def fnv64(data: bytes) -> int:
    """FNV-1a, 64 bit: the episode-id hash of `rrsi-mine` (transcript.rs)."""
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & U64
    return h


def truncate(t: str, n: int) -> str:
    """`transcript::truncate`: trimmed; longer text keeps its head and its length."""
    t = t.strip()
    if len(t) <= n:
        return t
    return f"{t[:n]}… [{len(t)} chars]"


@dataclass(frozen=True)
class Ev:
    """One normalized telemetry event (a model request, a generation, a turn)."""

    ts: str
    kind: str
    text: str
    ref: str
    tool: str | None = None
    is_error: bool | None = None
    #: Counts as a call in `counts.tool_calls` (a model request, a tool call).
    call: bool = False


def assemble(evs: list[Ev], hits: dict[int, list[Signal]], *, project: str, session_id: str,
             file: str, agent_id: str | None = None) -> list[dict]:
    """Hits at most EPISODE_GAP events apart become one episode, with the
    bounded context window and counts of `rrsi-mine traces`."""
    spans: list[list[int]] = []
    for i in sorted(hits):
        if len(spans) > 0 and i - spans[-1][1] <= EPISODE_GAP:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    out = []
    for a, b in spans:
        signals = Counter(s.value for i in range(a, b + 1) for s in hits.get(i, []))
        lo, hi = max(0, a - CONTEXT_BEFORE), min(b + CONTEXT_AFTER, len(evs) - 1)
        if hi - lo < CONTEXT_MAX:
            idxs = list(range(lo, hi + 1))
        else:
            idxs = [*range(lo, lo + CONTEXT_MAX // 2), *range(hi + 1 - CONTEXT_MAX // 2, hi + 1)]
        human = [e.text for e in evs[:a + 1] if e.kind == "human"]
        span = evs[a:b + 1]
        hit_idx = [i for i in range(a, b + 1) if i in hits]
        refs = list(dict.fromkeys(evs[i].ref for i in hit_idx))[:REFS_MAX]
        examples: dict[str, str] = {}
        for i in hit_idx:
            for s in hits[i]:
                examples.setdefault(s.value, evs[i].ref)
        ep = {
            "id": f"{fnv64(f'{file}#{a}'.encode()):016x}",
            "project": project, "session_id": session_id,
            "subagent": "/subagents/" in file, "file": file,
            "start": evs[a].ts, "end": evs[b].ts, "start_event": a, "end_event": b,
            "signals": dict(sorted(signals.items())),
            "user_turn": truncate(human[-1], USER_TURN_MAX) if len(human) > 0 else "",
            "context": [context(evs, i, hits) for i in idxs],
            "counts": {"span_events": len(span), "tool_calls": sum(1 for e in span if e.call),
                       "tool_errors": sum(1 for e in span if e.is_error is True),
                       "human_turns": sum(1 for e in span if e.kind == "human"),
                       "session_events": len(evs)},
            "refs": refs, "examples": dict(sorted(examples.items())),
        }
        # Absent rather than null, as rrsi-mine serializes it.
        if agent_id is not None:
            ep["agent_id"] = agent_id
        out.append(ep)
    return out


def context(evs: list[Ev], i: int, hits: dict[int, list[Signal]]) -> dict:
    e = evs[i]
    c: dict = {"i": i, "ts": e.ts, "kind": e.kind}
    if e.tool is not None:
        c["tool"] = e.tool
    if e.is_error is not None:
        c["is_error"] = e.is_error
    sigs = [s.value for s in hits.get(i, [])]
    if len(sigs) > 0:
        c["signals"] = sigs
    c["text"] = truncate(e.text, TEXT_MAX)
    return c


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _s(d: dict, k: str) -> str:
    v = d.get(k)
    return "" if v is None else str(v)


def _n(d: dict, k: str) -> int:
    v = d.get(k)
    return 0 if v is None else int(v)


def _d(d: dict, k: str) -> dict:
    v = d.get(k)
    return v if isinstance(v, dict) else {}


# ----------------------------------------------------- LiteLLM: projection

UA_PREFIX = "User-Agent: "
ERROR_MAX = 300
SELF_KEY = "self"
KEY_HASH_LEN = 8
USER_ID = re.compile(r"(?i)\b(user_id|user|key|team_id)=\S+")


def client_of(tags: list[str]) -> str:
    """The shortest User-Agent tag: the product family (`OpenAI`, `curl`)."""
    uas = [t[len(UA_PREFIX):] for t in tags if t.startswith(UA_PREFIX)]
    return "" if len(uas) == 0 else min(uas, key=len)


def key_label(api_key: str, alias: str, self_key: str) -> str:
    if api_key == self_key:
        return SELF_KEY
    name = api_key if alias == "" else alias
    return "k-" + hashlib.sha256(name.encode()).hexdigest()[:KEY_HASH_LEN]


def scrub(message: str) -> str:
    """An error message, single-line, without addresses, secrets, ids or hosts."""
    return M._trunc(USER_ID.sub(r"\1=<id>", M.redact(message)), ERROR_MAX)


def project_litellm(rec: dict, self_key: str) -> dict:
    """One `/spend/logs/v2` row without caller identity (no address, user,
    team, key hash or alias; no prompt text: the list view has none)."""
    md = _d(rec, "metadata")
    err = _d(md, "error_information")
    tags = rec.get("request_tags")
    return {
        "request_id": _s(rec, "request_id"), "start": _s(rec, "startTime"), "end": _s(rec, "endTime"),
        "first_token": _s(rec, "completionStartTime"), "duration_ms": _n(rec, "request_duration_ms"),
        "call_type": _s(rec, "call_type"), "model_group": _s(rec, "model_group"), "model": _s(rec, "model"),
        "status": _s(rec, "status"),
        "prompt_tokens": _n(rec, "prompt_tokens"), "completion_tokens": _n(rec, "completion_tokens"),
        "client": client_of(tags if isinstance(tags, list) else []),
        "key": key_label(_s(rec, "api_key"), _s(md, "user_api_key_alias"), self_key),
        "error_code": _s(err, "error_code"), "error_class": _s(err, "error_class"),
        "error": scrub(_s(err, "error_message")),
        "attempted_retries": _n(md, "attempted_retries"),
    }


# ------------------------------------------------------ LiteLLM: detectors

FAILURE = "failure"
#: A session of one stream ends after this much silence (and at UTC midnight).
SESSION_IDLE = timedelta(minutes=30)
#: A request this soon after a failed one, of about its size, re-sends it.
RETRY_WINDOW = timedelta(minutes=2)
RETRY_TOKEN_TOLERANCE = 0.01
#: A prompt at or above this share of the model's input limit is under pressure.
CONTEXT_PRESSURE = 0.9
SLOW_FIRST_TOKEN = timedelta(minutes=1)
COMPLETION_CALLS = ("completion", "acompletion", "text_completion", "atext_completion",
                    "responses", "aresponses", "anthropic_messages")
STATUS_CLASS_WIDTH = 100
ACCESS_MARKERS = ("not allowed to access model", "invalid model name", "model not found",
                  "team not allowed to access model")
CONTEXT_MARKERS = ("context length", "context window", "maximum context", "too many tokens",
                   "prompt is too long", "input is too long", "max_model_len", "contextwindowexceeded")


def _status(code: str) -> int | None:
    return int(code) if code.isdigit() else None


def failure_kind(code: str, error_class: str, message: str) -> Signal:
    """The kind of a failed request; the first matching rule wins."""
    status, cls, msg = _status(code), error_class.lower(), message.lower()
    if any(m in msg for m in ACCESS_MARKERS):
        return Signal.MODEL_ACCESS_DENIED
    if "contextwindowexceeded" in cls or any(m in msg for m in CONTEXT_MARKERS):
        return Signal.CONTEXT_OVERFLOW
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or "authentication" in cls \
            or "permissiondenied" in cls:
        return Signal.AUTH_ERROR
    if status == HTTPStatus.TOO_MANY_REQUESTS or "ratelimit" in cls:
        return Signal.RATE_LIMITED
    if status in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.GATEWAY_TIMEOUT) or "timeout" in cls \
            or "timed out" in msg:
        return Signal.LLM_TIMEOUT
    if status in (HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND, HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                  HTTPStatus.UNPROCESSABLE_ENTITY) or "badrequest" in cls or "unprocessable" in cls:
        return Signal.BAD_REQUEST
    if (status is not None and HTTPStatus.INTERNAL_SERVER_ERROR <= status
            < HTTPStatus.INTERNAL_SERVER_ERROR + STATUS_CLASS_WIDTH) \
            or any(m in cls for m in ("apiconnection", "internalserver", "serviceunavailable")):
        return Signal.UPSTREAM_ERROR
    return Signal.LLM_ERROR


def stream_key(r: dict) -> tuple[str, str, str]:
    return r["key"], r["client"], r["model_group"]


def gateway_sessions(records: list[dict]) -> Iterator[list[dict]]:
    """Requests per stream in start order, split at idle gaps and UTC midnight."""
    streams: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        streams[stream_key(r)].append(r)
    for k in sorted(streams):
        rs = sorted(streams[k], key=lambda r: (parse_ts(r["start"]), r["request_id"]))
        cur: list[dict] = []
        for r in rs:
            if len(cur) > 0:
                prev, t = parse_ts(cur[-1]["start"]), parse_ts(r["start"])
                if t - prev > SESSION_IDLE or t.date() != prev.date():
                    yield cur
                    cur = []
            cur.append(r)
        if len(cur) > 0:
            yield cur


def same_request(prev: dict, r: dict) -> bool:
    """About the same prompt; a request rejected before counting (0 tokens) matches any."""
    a, b = prev["prompt_tokens"], r["prompt_tokens"]
    if a == 0 or b == 0:
        return True
    return abs(a - b) <= RETRY_TOKEN_TOLERANCE * max(a, b)


def gateway_hits(rs: list[dict], max_input: dict[str, int], complete: bool = True) -> dict[int, list[Signal]]:
    """Hits per request index. Retries are inferred only from a complete
    stream: in a failures-only stream the previous row is not the previous
    request."""
    hits: dict[int, list[Signal]] = defaultdict(list)
    for i, r in enumerate(rs):
        start = parse_ts(r["start"])
        if r["status"] == FAILURE:
            hits[i].append(failure_kind(r["error_code"], r["error_class"], r["error"]))
        else:
            limit = max_input.get(r["model_group"])
            if limit is not None and r["prompt_tokens"] >= CONTEXT_PRESSURE * limit:
                hits[i].append(Signal.CONTEXT_PRESSURE)
            if r["completion_tokens"] == 0 and r["call_type"] in COMPLETION_CALLS:
                hits[i].append(Signal.EMPTY_COMPLETION)
            if r["first_token"] != "" and parse_ts(r["first_token"]) - start >= SLOW_FIRST_TOKEN:
                hits[i].append(Signal.SLOW_FIRST_TOKEN)
        if r["attempted_retries"] > 0:
            hits[i].append(Signal.GATEWAY_RETRY)
        if complete and i > 0:
            prev = rs[i - 1]
            if prev["status"] == FAILURE and start - parse_ts(prev["start"]) <= RETRY_WINDOW \
                    and same_request(prev, r):
                hits[i].append(Signal.LLM_RETRY)
    return dict(hits)


def _dash(s: str) -> str:
    return "-" if s == "" else s


def gateway_event(r: dict) -> Ev:
    text = (f"{r['status']} {r['model_group']} via {_dash(r['client'])}: prompt {r['prompt_tokens']} tok, "
            f"completion {r['completion_tokens']} tok, {r['duration_ms']} ms")
    if r["status"] == FAILURE:
        text += f"; {r['error_code']} {r['error_class']}: {r['error']}"
    return Ev(ts=r["start"], kind="llm_request", text=text, ref=r["request_id"], tool=r["model_group"],
              is_error=r["status"] == FAILURE, call=True)


def litellm_episodes(records: list[dict], max_input: dict[str, int], project: str,
                     complete: bool = True) -> list[dict]:
    out = []
    for rs in gateway_sessions(records):
        key, client, group = stream_key(rs[0])
        stream = f"{key}/{_dash(client)}/{_dash(group)}"
        out += assemble([gateway_event(r) for r in rs], gateway_hits(rs, max_input, complete),
                        project=project, session_id=f"{stream}/{rs[0]['request_id']}",
                        file=f"{project}/{stream}/{rs[0]['start']}")
    return out


# ---------------------------------------------------- Langfuse: projection

#: Metadata kept from an observation (the rest is exporter bookkeeping).
LANGFUSE_METADATA = (
    "sourceEventId", "sourceUsageRowId", "success", "toolCallId", "toolName", "parentToolCallId", "agentId", "finishReason", "turnStatus",
    "sourceEventType", "durationMs", "attributes.langfuse.session.id", "session_id", "attributes.session.id",
    "attributes.gen_ai.conversation.id", "attributes.gen_ai.tool.name", "attributes.gen_ai.tool.call.id",
    "attributes.gen_ai.operation.name", "attributes.gen_ai.response.finish_reasons", "attributes.error.type",
)
MESSAGE_MAX = 4000
LANGFUSE_KEY = re.compile(r"\b[ps]k-lf-[0-9A-Fa-f-]{8,}")


def redact(text: str) -> str:
    """rrsi's redaction plus Langfuse project keys pasted into a conversation."""
    return M.redact(LANGFUSE_KEY.sub("<secret>", text))


def project_langfuse(o: dict) -> dict:
    """One v2 observation: identity, timing, level, kept metadata; input and
    output only when they are message text."""
    md = _d(o, "metadata")
    keep = {k: md[k] for k in LANGFUSE_METADATA if k in md}
    io = {k: redact(o[k])[:MESSAGE_MAX] for k in ("input", "output") if isinstance(o.get(k), str)}
    return {"id": _s(o, "id"), "traceId": _s(o, "traceId"), "sessionId": _s(o, "sessionId"),
            "type": _s(o, "type"), "name": _s(o, "name"), "level": _s(o, "level"),
            "statusMessage": redact(_s(o, "statusMessage")), "startTime": _s(o, "startTime"),
            "parent": _s(o, "parentObservationId"), "metadata": keep, **io}


# --------------------------------------------- Langfuse: Copilot sessions

ERROR_LEVEL = "ERROR"
LENGTH = "length"
FAILED_TURN = "failed"
#: Event order within one timestamp: what a turn does, in the order it does it.
RANK = {"human": 0, "assistant": 1, "tool_use": 2, "tool_result": 3, "generation": 4, "agent_turn": 5}


def session_of(o: dict) -> str:
    md = o["metadata"]
    for v in (o["sessionId"], md.get("attributes.langfuse.session.id"), md.get("session_id"),
              md.get("attributes.session.id"), md.get("attributes.gen_ai.conversation.id")):
        if isinstance(v, str) and v != "":
            return v
    return o["traceId"]


def kind_of(o: dict) -> str | None:
    """What an observation is in a transcript, or None when it is bookkeeping."""
    name, typ, md = o["name"], o["type"], o["metadata"]
    if name == "copilot.message:user":
        return "human"
    if name == "copilot.message:assistant":
        return "assistant"
    if name == "copilot.tool.complete":
        return "tool_result"
    if typ == "TOOL" and (name.startswith("copilot.tool:")
                          or md.get("attributes.gen_ai.operation.name") == "execute_tool"):
        return "tool_use"
    if typ == "GENERATION" and (name.startswith("copilot.model:") or name.startswith("chat ")):
        return "generation"
    if typ == "AGENT":
        return "agent_turn"
    return None


def tool_call_id(o: dict) -> str:
    md = o["metadata"]
    for k in ("toolCallId", "attributes.gen_ai.tool.call.id"):
        v = md.get(k)
        if isinstance(v, str) and v != "":
            return v
    return o["id"]


def tool_name(o: dict) -> str:
    md = o["metadata"]
    for v in (md.get("toolName"), md.get("attributes.gen_ai.tool.name")):
        if isinstance(v, str) and v != "":
            return v
    return o["name"].removeprefix("copilot.tool:")


def finish_reasons(o: dict) -> list[str]:
    md = o["metadata"]
    v = md.get("finishReason")
    if isinstance(v, str):
        return [v]
    v = md.get("attributes.gen_ai.response.finish_reasons")
    return [str(x) for x in v] if isinstance(v, list) else []


def _ordered(obs: list[dict]) -> list[tuple[str, dict]]:
    ks = [(kind_of(o), o) for o in obs]
    return sorted(((k, o) for k, o in ks if k is not None),
                  key=lambda ko: (parse_ts(ko[1]["startTime"]), RANK[ko[0]], ko[1]["id"]))


def ref_of(o: dict) -> str:
    """An observation's example id: `<trace id>/<observation id>`."""
    return f"{o['traceId']}/{o['id']}"


def _agent(o: dict) -> str:
    v = o["metadata"].get("parentToolCallId")
    return v if isinstance(v, str) else ""


def transcript_lines(obs: list[dict]) -> dict[str, list[dict]]:
    """One session's observations as Claude Code transcript lines, keyed by
    "" (the main agent) or the subagent's parent tool call id."""
    out: dict[str, list[dict]] = defaultdict(list)
    for kind, o in _ordered(obs):
        agent, ts, sid = _agent(o), o["startTime"], session_of(o)
        # rrsi-mine ignores the extra field; it maps hits back to observations.
        base = {"timestamp": ts, "sessionId": sid, "langfuseRef": ref_of(o)}
        if agent != "":
            base |= {"isSidechain": True, "agentId": agent}
        if kind == "human":
            # The export drops the message's source (operator, peer agent,
            # schedule): rrsi-mine's plain-prompt rule decides, as it does for
            # Copilot messages without one. A subagent's messages are briefs.
            line = {"type": "user", "message": {"content": o.get("input", "")}}
            if agent != "":
                line["origin"] = {"kind": "system"}
        elif kind == "assistant":
            line = {"type": "assistant", "message": {"content": [{"type": "text", "text": o.get("output", "")}]}}
        elif kind == "tool_use":
            # The input is not exported; a per-call placeholder keeps unrelated
            # calls from reading as retries of one another.
            line = {"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": tool_call_id(o), "name": tool_name(o),
                "input": {"unrecorded": o["id"]}}]}}
            out[agent].append(base | line)
            if o["level"] != ERROR_LEVEL:
                continue
            line = {"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": tool_call_id(o), "is_error": True,
                "content": o["statusMessage"]}]}}
        elif kind == "tool_result":
            line = {"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": tool_call_id(o),
                "is_error": o["metadata"].get("success") is False or o["level"] == ERROR_LEVEL,
                "content": o["statusMessage"]}]}}
        else:
            continue
        out[agent].append(base | line)
    return dict(out)


def langfuse_hits(evs: list[tuple[str, dict]]) -> dict[int, list[Signal]]:
    hits: dict[int, list[Signal]] = defaultdict(list)
    for i, (kind, o) in enumerate(evs):
        if kind == "generation" and o["level"] == ERROR_LEVEL:
            hits[i].append(Signal.LLM_ERROR)
        if kind == "agent_turn" and (o["level"] == ERROR_LEVEL or o["metadata"].get("turnStatus") == FAILED_TURN):
            hits[i].append(Signal.TURN_FAILED)
        # The CLI's own spans report finish reasons on the agent invocation.
        if kind in ("generation", "agent_turn") and LENGTH in finish_reasons(o):
            hits[i].append(Signal.LLM_TRUNCATED)
    return dict(hits)


def langfuse_event(kind: str, o: dict) -> Ev:
    if kind == "human":
        return Ev(ts=o["startTime"], kind="human", text=o.get("input", ""), ref=ref_of(o))
    if kind == "assistant":
        return Ev(ts=o["startTime"], kind="assistant", text=o.get("output", ""), ref=ref_of(o))
    if kind == "tool_use":
        return Ev(ts=o["startTime"], kind="tool_use", text="", ref=ref_of(o), tool=tool_name(o), call=True)
    if kind == "tool_result":
        err = o["metadata"].get("success") is False or o["level"] == ERROR_LEVEL
        return Ev(ts=o["startTime"], kind="tool_result", text=o["statusMessage"], ref=ref_of(o), is_error=err)
    reasons = ",".join(finish_reasons(o))
    text = f"{o['name']} level {o['level']}" + (f", finish {reasons}" if reasons != "" else "") \
        + (f": {o['statusMessage']}" if o["statusMessage"] != "" else "")
    return Ev(ts=o["startTime"], kind=kind, text=text, ref=ref_of(o), tool=o["name"],
              is_error=o["level"] == ERROR_LEVEL)


SOURCE_IDS = ("sourceEventId", "sourceUsageRowId")


def dedupe(obs: list[dict]) -> list[dict]:
    """One observation per exported source record: a session the bridge
    exported twice (two traces) counts once."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for o in sorted(obs, key=lambda o: (o["startTime"], o["id"])):
        src = next((v for v in (o["metadata"].get(k) for k in SOURCE_IDS) if isinstance(v, str) and v != ""), "")
        if src != "":
            ident = (o["type"], o["name"], src)
            if ident in seen:
                continue
            seen.add(ident)
        out.append(o)
    return out


def langfuse_sessions(obs: list[dict]) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    for o in obs:
        sessions[session_of(o)].append(o)
    return dict(sessions)


def transcript_rel(label: str, session: str, agent: str) -> str:
    return f"langfuse-{label}/{session}.jsonl" if agent == "" \
        else f"langfuse-{label}/{session}/subagents/agent-{agent}.jsonl"


def line_kind(line: dict) -> str:
    """The rrsi-mine event kind a transcript line becomes."""
    c = line["message"]["content"]
    if isinstance(c, str):
        return "human"
    return {"text": "assistant", "tool_use": "tool_use", "tool_result": "tool_result"}[c[0]["type"]]


@dataclass(frozen=True)
class Refs:
    """Where a mined transcript came from: its session's main trace and the
    observation behind each (timestamp, event kind)."""

    trace: dict[str, str]
    line: dict[tuple[str, str, str], str]

    def of(self, rel: str, ts: str, kind: str) -> str:
        return self.line.get((rel, ts, kind), self.trace[rel])


def write_transcripts(obs: list[dict], label: str, root: Path) -> Refs:
    """Writes every session's transcripts under root."""
    refs = Refs({}, {})
    for sid, os_ in langfuse_sessions(obs).items():
        trace = Counter(o["traceId"] for o in os_).most_common(1)[0][0]
        for agent, lines in transcript_lines(os_).items():
            rel = transcript_rel(label, sid, agent)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("".join(json.dumps(l) + "\n" for l in lines))
            refs.trace[rel] = trace
            for l in lines:
                refs.line.setdefault((rel, l["timestamp"], line_kind(l)), l["langfuseRef"])
    return refs


def langfuse_episodes(obs: list[dict], label: str) -> list[dict]:
    """The generation and turn classes, per session and agent."""
    out = []
    for sid, os_ in sorted(langfuse_sessions(obs).items()):
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for o in os_:
            by_agent[_agent(o)].append(o)
        for agent, aos in sorted(by_agent.items()):
            evs = _ordered(aos)
            rel = transcript_rel(label, sid, agent)
            out += assemble([langfuse_event(k, o) for k, o in evs], langfuse_hits(evs),
                            project=f"langfuse-{label}", session_id=sid,
                            file=rel.removesuffix(".jsonl") + ".generations",
                            agent_id=agent if agent != "" else None)
    return out


def traces_episodes(obs: list[dict], label: str, work: Path, exe: Path) -> tuple[list[dict], dict]:
    """The nine transcript classes: the sessions as transcripts, mined by
    `rrsi-mine traces` unchanged; each episode's refs are its trace id."""
    root = M.ensure_private(work / "transcripts")
    refs = write_transcripts(obs, label, root)
    out = M.ensure_private(work / "traces")
    r = subprocess.run([str(exe), "traces", "--root", str(root), "--out", str(out), "--jobs", "8"],
                       check=True, capture_output=True, text=True)
    summary = json.loads(r.stdout)
    eps = []
    for e in M.load_episodes(out):
        if e["project"] != f"langfuse-{label}":
            continue
        examples: dict[str, str] = {}
        for c in e["context"]:
            for sig in c.get("signals", []):
                examples.setdefault(sig, refs.of(e["file"], c["ts"], c["kind"]))
        for sig in e["signals"]:
            examples.setdefault(sig, refs.trace[e["file"]])
        eps.append(e | {"refs": list(dict.fromkeys(examples.values()))[:REFS_MAX],
                        "examples": dict(sorted(examples.items()))})
    return eps, summary


# ------------------------------------------------------------ ranking

#: The candidate harness fix per class (the evidence decides which ship).
FIXES = {
    "tool_error": "rank failing tools/commands from these episodes; add the missing tool or a pre-check to the "
                  "environment and an instruction naming the working command",
    "retry": "stop-after-one-identical-failure rule: re-read the error and change the command before re-running",
    "test_failure": "repo test entry point in AGENTS.md (one command per language) + run the narrowest test first",
    "permission_denial": "allow-list the denied read-only tools/paths in the harness config",
    "hook_timeout": "raise or remove the slow hook; hooks must answer within the CLI's budget",
    "user_interrupt": "check-in rule: say the plan before long or destructive actions",
    "user_correction": "promote each correction to a persistent instruction (operator rules file)",
    "reask": "answer-the-question-first rule; summarize before continuing work",
    "silence": "progress-report cadence instruction for long tasks",
    Signal.MODEL_ACCESS_DENIED.value: "client model map: request only gateway-served model names "
                                      "(fast/deep); point health probes at an allowed model",
    Signal.AUTH_ERROR.value: "refresh/validate the gateway key at session start; fail fast with a clear message",
    Signal.CONTEXT_OVERFLOW.value: "compact before the model limit (trigger at ~80% of max_input_tokens) "
                                   "and cap tool-output size",
    Signal.RATE_LIMITED.value: "client backoff + concurrency cap per key; spread subagents over time",
    Signal.LLM_TIMEOUT.value: "client timeout above the observed p99 latency; stream responses",
    Signal.BAD_REQUEST.value: "validate request shape (tool schemas, message roles) for this model's server",
    Signal.UPSTREAM_ERROR.value: "retry 5xx with jittered backoff; route to the healthy deployment",
    Signal.LLM_ERROR.value: "surface the error class to the agent and log it; triage the unknown kind",
    Signal.LLM_RETRY.value: "do not re-send a request that failed for a non-transient reason; "
                            "change the request (model, size) first",
    Signal.GATEWAY_RETRY.value: "upstream flakiness: pin the healthy deployment or raise its capacity",
    Signal.CONTEXT_PRESSURE.value: "compact earlier and trim tool output so prompts stay under ~80% of the limit",
    Signal.EMPTY_COMPLETION.value: "detect empty completions and re-prompt once; check max_tokens/stop settings",
    Signal.SLOW_FIRST_TOKEN.value: "smaller prompts (compaction), prefix caching, or the faster model for "
                                   "routine turns",
    Signal.LLM_TRUNCATED.value: "raise max output tokens or ask for shorter outputs/patches per turn",
    Signal.TURN_FAILED.value: "checkpoint rule: each turn ends with a verified, committed step",
}


EXAMPLES = 2
#: Rates are per this many observations.
PER = 1000


def summarize(eps: list[dict], observations: dict[str, int]) -> dict:
    """Per source (an episode's project): observations and, per class,
    episodes, hits and up to EXAMPLES example ids from distinct episodes."""
    out: dict = {src: {"observations": n, "episodes": 0, "classes": {}} for src, n in observations.items()}
    for e in eps:
        src = out.setdefault(e["project"], {"observations": 0, "episodes": 0, "classes": {}})
        src["episodes"] += 1
        refs = e.get("refs", [])
        for sig, n in e["signals"].items():
            c = src["classes"].setdefault(sig, {"episodes": 0, "hits": 0, "examples": []})
            c["episodes"] += 1
            c["hits"] += n
            ex = e.get("examples", {}).get(sig, refs[0] if len(refs) > 0 else None)
            if ex is not None and ex not in c["examples"] and len(c["examples"]) < EXAMPLES:
                c["examples"].append(ex)
    return out


def ranked(summary: dict) -> list[dict]:
    """One row per (class, source), most episodes first; rate per 1k observations of that source."""
    rows = []
    for src, s in summary.items():
        for sig, c in s["classes"].items():
            n = s["observations"]
            rows.append({"class": sig, "source": src, "episodes": c["episodes"], "hits": c["hits"],
                         "observations": n,
                         "rate_per_1k": round(PER * c["episodes"] / n, 3) if n > 0 else None,
                         "examples": c["examples"], "fix": FIXES.get(sig, "")})
    return sorted(rows, key=lambda r: (-r["episodes"], -r["hits"], r["class"], r["source"]))


def table(rows: list[dict]) -> str:
    lines = ["| class | source | episodes | hits | rate/1k obs | example ids | proposed harness fix |",
             "|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        rate = "-" if r["rate_per_1k"] is None else f"{r['rate_per_1k']:.2f}"
        ex = ", ".join(f"`{x}`" for x in r["examples"])
        lines.append(f"| {r['class']} | {r['source']} | {r['episodes']} | {r['hits']} | {rate} | {ex} | {r['fix']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- fetch

HTTP_TIMEOUT_S = 180
HTTP_ATTEMPTS = 4
BACKOFF_S = 5
LITELLM_LOGS = "/spend/logs/v2"
LITELLM_PAGE = 1000
LITELLM_DATE = "%Y-%m-%d %H:%M:%S"
LITELLM_WINDOW = timedelta(hours=1)
LITELLM_MIN_WINDOW = timedelta(minutes=1)
LITELLM_SPLIT = 4
LANGFUSE_OBSERVATIONS = "/api/public/v2/observations"
LANGFUSE_PAGE = 1000
LANGFUSE_FIELDS = "core,basic,time,io,metadata,model,usage"

Get = Callable[[str, dict[str, str]], dict]


class Scope(str, Enum):
    """What a crawl of the gateway's request logs pages through."""

    #: Every request of the caller's own key: complete streams.
    SELF = "self"
    #: The failed requests of every key.
    FAILURES = "failures"


def http_get(url: str, headers: dict[str, str]) -> dict:
    """GET JSON; transient failures (timeouts, 429, 5xx) are retried with backoff."""
    last: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != HTTPStatus.TOO_MANY_REQUESTS and e.code < HTTPStatus.INTERNAL_SERVER_ERROR:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(BACKOFF_S * 2 ** attempt)
    assert last is not None
    raise last


def _window_url(base: str, a: datetime, b: datetime, page: int, extra: dict[str, str],
                size: int = 0) -> str:
    q = urllib.parse.urlencode({"start_date": a.strftime(LITELLM_DATE), "end_date": b.strftime(LITELLM_DATE),
                                "page": page, "page_size": LITELLM_PAGE if size == 0 else size,
                                "exclude_internal_health_checks": "true", **extra})
    return f"{base}{LITELLM_LOGS}?{q}"


def _split(a: datetime, b: datetime) -> list[tuple[datetime, datetime]]:
    step = (b - a) / LITELLM_SPLIT
    return [(a + k * step, a + (k + 1) * step) for k in range(LITELLM_SPLIT)]


def fetch_window(base: str, headers: dict[str, str], a: datetime, b: datetime, get: Get,
                 log: Callable[[str], None], extra: dict[str, str]) -> list[dict]:
    """Every row of [a, b]; a window the server caps is split, never truncated."""
    first = get(_window_url(base, a, b, 1, extra), headers)
    if first.get("total_is_capped") is True:
        if b - a > LITELLM_MIN_WINDOW:
            log(f"[litellm] {a:%m-%d %H:%M:%S}..{b:%H:%M:%S} capped; split in {LITELLM_SPLIT}")
            return [r for x, y in _split(a, b) for r in fetch_window(base, headers, x, y, get, log, extra)]
        log(f"[litellm] WARNING {a:%m-%d %H:%M:%S}..{b:%H:%M:%S} capped at the minimum window: rows are missing")
    rows = list(first.get("data", []))
    for page in range(2, _n(first, "total_pages") + 1):
        rows += get(_window_url(base, a, b, page, extra), headers).get("data", [])
    return rows


def count_window(base: str, headers: dict[str, str], a: datetime, b: datetime, get: Get,
                 log: Callable[[str], None]) -> int:
    """Requests of every key in [a, b]; a capped count is split."""
    first = get(_window_url(base, a, b, 1, {}, size=1), headers)
    if first.get("total_is_capped") is True:
        if b - a > LITELLM_MIN_WINDOW:
            return sum(count_window(base, headers, x, y, get, log) for x, y in _split(a, b))
        log(f"[litellm] WARNING {a:%m-%d %H:%M:%S}..{b:%H:%M:%S} count capped at the minimum window")
    return _n(first, "total")


def windows_of(since: datetime, until: datetime, step: timedelta) -> list[tuple[datetime, datetime]]:
    """[since, until) in `step` windows. Adjacent windows share their
    boundary second (the server's end bound is inclusive): rows are
    deduplicated by request id; a count may include a row stamped exactly on
    a boundary twice."""
    out, t = [], since
    while t < until:
        out.append((t, min(t + step, until)))
        t += step
    return out


def count_litellm(base: str, key: str, since: datetime, until: datetime, raw: Path, jobs: int = 3,
                  get: Get = http_get, log: Callable[[str], None] = print,
                  window: timedelta = LITELLM_WINDOW) -> dict:
    """Requests of every key per window of [since, until) into raw/litellm-counts.json."""
    M.ensure_private(raw)
    headers = {"Authorization": f"Bearer {key}"}
    windows = windows_of(since, until, window)
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        totals = list(ex.map(lambda w: count_window(base, headers, w[0], w[1], get, log), windows))
    out = {"since": since.isoformat(), "until": until.isoformat(), "total": sum(totals),
           "windows": {a.isoformat(): n for (a, _), n in zip(windows, totals)}}
    (raw / "litellm-counts.json").write_text(json.dumps(out, indent=1))
    return {k: out[k] for k in ("since", "until", "total")}


def fetch_litellm(base: str, key: str, since: datetime, until: datetime, raw: Path, scope: Scope,
                  jobs: int = 3, get: Get = http_get, log: Callable[[str], None] = print,
                  window: timedelta = LITELLM_WINDOW) -> dict:
    """The scope's projected request logs of [since, until) into
    raw/litellm-<scope>.jsonl and the model groups' input limits into
    raw/litellm-models.json."""
    M.ensure_private(raw)
    headers = {"Authorization": f"Bearer {key}"}
    self_key = _s(get(f"{base}/key/info", headers), "key")
    extra = {"api_key": self_key} if scope == Scope.SELF else {"status_filter": FAILURE}
    limits: dict[str, int] = {}
    for m in get(f"{base}/model/info", headers).get("data", []):
        n = _d(m, "model_info").get("max_input_tokens")
        if isinstance(n, int):
            limits[m["model_name"]] = min(n, limits.get(m["model_name"], n))
    (raw / "litellm-models.json").write_text(json.dumps(limits, indent=1, sort_keys=True))
    windows = windows_of(since, until, window)
    rows: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        for got in ex.map(lambda w: fetch_window(base, headers, w[0], w[1], get, log, extra), windows):
            for rec in got:
                p = project_litellm(rec, self_key)
                rows[p["request_id"]] = p
            done += 1
            log(f"[litellm {scope.value}] {done}/{len(windows)} windows, {len(rows)} rows")
    ordered = sorted(rows.values(), key=lambda r: (r["start"], r["request_id"]))
    (raw / f"litellm-{scope.value}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ordered))
    return {"scope": scope.value, "requests": len(ordered), "windows": len(windows), "model_groups": limits}


def fetch_langfuse(host: str, public: str, secret: str, label: str, raw: Path, get: Get = http_get,
                   log: Callable[[str], None] = print) -> dict:
    """Every observation of the key's project into raw/langfuse-<label>.jsonl."""
    M.ensure_private(raw)
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    out, cursor = [], ""
    while True:
        q = {"limit": LANGFUSE_PAGE, "fields": LANGFUSE_FIELDS} | ({"cursor": cursor} if cursor != "" else {})
        page = get(f"{host}{LANGFUSE_OBSERVATIONS}?{urllib.parse.urlencode(q)}", headers)
        data = page.get("data", [])
        out += [project_langfuse(o) for o in data]
        cursor = _s(_d(page, "meta"), "cursor")
        if cursor == "" or len(data) == 0:
            break
        log(f"[langfuse {label}] {len(out)} observations")
    (raw / f"langfuse-{label}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    return {"observations": len(out)}


# --------------------------------------------------------------- driver

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip() != ""]


def mine_raw(raw: Path, out: Path, work: Path, exe: Path, gateway_label: str,
             exclude: list[str] | tuple[str, ...] = ()) -> dict:
    """Every source in raw/ -> episodes in `out` and their summary next to it.
    Langfuse sessions whose id contains an `exclude` substring are dropped
    (synthetic fixtures, the experiment's own sessions)."""
    M.ensure_private(out.parent)
    eps: list[dict] = []
    observations: dict[str, int] = {}
    miner: dict[str, dict] = {}
    models, own, failed, counts = (raw / f for f in ("litellm-models.json", "litellm-self.jsonl",
                                                       "litellm-failures.jsonl", "litellm-counts.json"))
    limits = json.loads(models.read_text()) if models.exists() else {}
    own_rows = read_jsonl(own) if own.exists() else []
    if own.exists():
        project = f"litellm-{gateway_label}-self"
        observations[project] = len(own_rows)
        eps += litellm_episodes(own_rows, limits, project, complete=True)
    if failed.exists():
        project = f"litellm-{gateway_label}-others"
        total = json.loads(counts.read_text())["total"] if counts.exists() else len(own_rows)
        observations[project] = total - len(own_rows)
        others = [r for r in read_jsonl(failed) if r["key"] != SELF_KEY]
        eps += litellm_episodes(others, limits, project, complete=False)
    for p in sorted(raw.glob("langfuse-*.jsonl")):
        label = p.stem.removeprefix("langfuse-")
        obs = [o for o in dedupe(read_jsonl(p)) if not any(x != "" and x in session_of(o) for x in exclude)]
        observations[f"langfuse-{label}"] = len(obs)
        eps += langfuse_episodes(obs, label)
        t_eps, miner[label] = traces_episodes(obs, label, work / label, exe)
        eps += t_eps
    out.write_text("".join(json.dumps(e) + "\n" for e in eps))
    summary = {"sources": summarize(eps, observations), "rrsi_mine_traces": miner}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        sys.exit(f"{name} is not set")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rrsi.harness.langfuse_miner", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-litellm", help="gateway request logs (key: LITELLM_API_KEY)")
    f.add_argument("--scope", type=Scope, choices=list(Scope), required=True)
    f.add_argument("--base", required=True)
    f.add_argument("--since", required=True, help="UTC date or datetime")
    f.add_argument("--until", required=True, help="UTC date or datetime (exclusive)")
    f.add_argument("--raw", type=Path, required=True)
    f.add_argument("--jobs", type=int, default=3)
    f.add_argument("--window-hours", type=float, default=1.0, help="query window; sparse scopes go faster with days")
    c = sub.add_parser("count-litellm", help="gateway request totals per window (key: LITELLM_API_KEY)")
    c.add_argument("--base", required=True)
    c.add_argument("--since", required=True)
    c.add_argument("--until", required=True)
    c.add_argument("--raw", type=Path, required=True)
    c.add_argument("--jobs", type=int, default=3)
    c.add_argument("--window-hours", type=float, default=1.0)
    g = sub.add_parser("fetch-langfuse", help="observations (keys: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)")
    g.add_argument("--host", required=True)
    g.add_argument("--label", required=True)
    g.add_argument("--raw", type=Path, required=True)
    e = sub.add_parser("episodes", help="raw/ -> episodes JSONL + summary")
    e.add_argument("--raw", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--work", type=Path, help="private work dir (default: <out>.work)")
    e.add_argument("--rrsi-mine", type=Path, help="rrsi-mine binary (default: build this checkout's)")
    e.add_argument("--gateway-label", default="gateway")
    e.add_argument("--exclude", action="append", default=[], help="drop Langfuse sessions whose id contains this")
    t = sub.add_parser("table", help="ranked markdown table from a summary")
    t.add_argument("--summary", type=Path, required=True)
    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    if a.cmd == "fetch-litellm":
        r = fetch_litellm(a.base.rstrip("/"), _env("LITELLM_API_KEY"), _date(a.since), _date(a.until),
                          a.raw, a.scope, a.jobs, log=log, window=timedelta(hours=a.window_hours))
    elif a.cmd == "count-litellm":
        r = count_litellm(a.base.rstrip("/"), _env("LITELLM_API_KEY"), _date(a.since), _date(a.until),
                          a.raw, a.jobs, log=log, window=timedelta(hours=a.window_hours))
    elif a.cmd == "fetch-langfuse":
        r = fetch_langfuse(a.host.rstrip("/"), _env("LANGFUSE_PUBLIC_KEY"), _env("LANGFUSE_SECRET_KEY"),
                           a.label, a.raw, log=log)
    elif a.cmd == "episodes":
        exe = a.rrsi_mine if a.rrsi_mine is not None else M.rust_binary()
        work = a.work if a.work is not None else a.out.with_suffix(".work")
        r = mine_raw(a.raw, a.out, work, exe, a.gateway_label, a.exclude)
    else:
        print(table(ranked(json.loads(a.summary.read_text())["sources"])))
        return 0
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
