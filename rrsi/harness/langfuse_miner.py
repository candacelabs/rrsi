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
# Stub: the tests define the contract (RED).
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
  completions, slow first tokens) are detected here.

    python -m rrsi.harness.langfuse_miner fetch-litellm --base URL --since 2026-09-15 --until 2026-10-05 --raw DIR
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
EPISODE_GAP = 6
CONTEXT_BEFORE = 3
CONTEXT_AFTER = 2
CONTEXT_MAX = 24
TEXT_MAX = 400
USER_TURN_MAX = 800
REFS_MAX = 5
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64 = (1 << 64) - 1

class Signal(str, Enum):
    """The classes detected here. The nine transcript classes (`tool_error`,
    `retry`, `test_failure`, ...) come from `rrsi-mine traces`."""
    MODEL_ACCESS_DENIED = 'model_access_denied'
    AUTH_ERROR = 'auth_error'
    CONTEXT_OVERFLOW = 'context_overflow'
    RATE_LIMITED = 'rate_limited'
    LLM_TIMEOUT = 'llm_timeout'
    BAD_REQUEST = 'bad_request'
    UPSTREAM_ERROR = 'upstream_error'
    LLM_ERROR = 'llm_error'
    LLM_RETRY = 'llm_retry'
    GATEWAY_RETRY = 'gateway_retry'
    CONTEXT_PRESSURE = 'context_pressure'
    EMPTY_COMPLETION = 'empty_completion'
    SLOW_FIRST_TOKEN = 'slow_first_token'
    LLM_TRUNCATED = 'llm_truncated'
    TURN_FAILED = 'turn_failed'

def fnv64(data: bytes) -> int:
    """FNV-1a, 64 bit: the episode-id hash of `rrsi-mine` (transcript.rs)."""
    raise NotImplementedError

def truncate(t: str, n: int) -> str:
    """`transcript::truncate`: trimmed; longer text keeps its head and its length."""
    raise NotImplementedError

@dataclass(frozen=True)
class Ev:
    """One normalized telemetry event (a model request, a generation, a turn)."""
    ts: str
    kind: str
    text: str
    ref: str
    tool: str | None = None
    is_error: bool | None = None
    call: bool = False

def assemble(evs: list[Ev], hits: dict[int, list[Signal]], *, project: str, session_id: str, file: str, agent_id: str | None=None) -> list[dict]:
    """Hits at most EPISODE_GAP events apart become one episode, with the
    bounded context window and counts of `rrsi-mine traces`."""
    raise NotImplementedError

def context(evs: list[Ev], i: int, hits: dict[int, list[Signal]]) -> dict:
    raise NotImplementedError

def parse_ts(ts: str) -> datetime:
    raise NotImplementedError

def _s(d: dict, k: str) -> str:
    raise NotImplementedError

def _n(d: dict, k: str) -> int:
    raise NotImplementedError

def _d(d: dict, k: str) -> dict:
    raise NotImplementedError
UA_PREFIX = 'User-Agent: '
ERROR_MAX = 300
SELF_KEY = 'self'
KEY_HASH_LEN = 8
USER_ID = re.compile('(?i)\\b(user_id|user|key|team_id)=\\S+')

def client_of(tags: list[str]) -> str:
    """The shortest User-Agent tag: the product family (`OpenAI`, `curl`)."""
    raise NotImplementedError

def key_label(api_key: str, alias: str, self_key: str) -> str:
    raise NotImplementedError

def scrub(message: str) -> str:
    """An error message, single-line, without addresses, secrets, ids or hosts."""
    raise NotImplementedError

def project_litellm(rec: dict, self_key: str) -> dict:
    """One `/spend/logs/v2` row without caller identity (no address, user,
    team, key hash or alias; no prompt text: the list view has none)."""
    raise NotImplementedError
FAILURE = 'failure'
SESSION_IDLE = timedelta(minutes=30)
RETRY_WINDOW = timedelta(minutes=2)
RETRY_TOKEN_TOLERANCE = 0.01
CONTEXT_PRESSURE = 0.9
SLOW_FIRST_TOKEN = timedelta(minutes=1)
COMPLETION_CALLS = ('completion', 'acompletion', 'text_completion', 'atext_completion', 'responses', 'aresponses', 'anthropic_messages')
STATUS_CLASS_WIDTH = 100
ACCESS_MARKERS = ('not allowed to access model', 'invalid model name', 'model not found', 'team not allowed to access model')
CONTEXT_MARKERS = ('context length', 'context window', 'maximum context', 'too many tokens', 'prompt is too long', 'input is too long', 'max_model_len', 'contextwindowexceeded')

def _status(code: str) -> int | None:
    raise NotImplementedError

def failure_kind(code: str, error_class: str, message: str) -> Signal:
    """The kind of a failed request; the first matching rule wins."""
    raise NotImplementedError

def stream_key(r: dict) -> tuple[str, str, str]:
    raise NotImplementedError

def gateway_sessions(records: list[dict]) -> Iterator[list[dict]]:
    """Requests per stream in start order, split at idle gaps and UTC midnight."""
    raise NotImplementedError

def same_request(prev: dict, r: dict) -> bool:
    """About the same prompt; a request rejected before counting (0 tokens) matches any."""
    raise NotImplementedError

def gateway_hits(rs: list[dict], max_input: dict[str, int]) -> dict[int, list[Signal]]:
    raise NotImplementedError

def _dash(s: str) -> str:
    raise NotImplementedError

def gateway_event(r: dict) -> Ev:
    raise NotImplementedError

def litellm_episodes(records: list[dict], max_input: dict[str, int], label: str) -> list[dict]:
    raise NotImplementedError
LANGFUSE_METADATA = ('sourceEventId', 'sourceUsageRowId', 'success', 'toolCallId', 'toolName', 'parentToolCallId', 'agentId', 'finishReason', 'turnStatus', 'sourceEventType', 'durationMs', 'attributes.langfuse.session.id', 'session_id', 'attributes.session.id', 'attributes.gen_ai.conversation.id', 'attributes.gen_ai.tool.name', 'attributes.gen_ai.tool.call.id', 'attributes.gen_ai.operation.name', 'attributes.gen_ai.response.finish_reasons', 'attributes.error.type')
MESSAGE_MAX = 4000

def project_langfuse(o: dict) -> dict:
    """One v2 observation: identity, timing, level, kept metadata; input and
    output only when they are message text."""
    raise NotImplementedError
ERROR_LEVEL = 'ERROR'
LENGTH = 'length'
FAILED_TURN = 'failed'
RANK = {'human': 0, 'assistant': 1, 'tool_use': 2, 'tool_result': 3, 'generation': 4, 'agent_turn': 5}

def session_of(o: dict) -> str:
    raise NotImplementedError

def kind_of(o: dict) -> str | None:
    """What an observation is in a transcript, or None when it is bookkeeping."""
    raise NotImplementedError

def tool_call_id(o: dict) -> str:
    raise NotImplementedError

def tool_name(o: dict) -> str:
    raise NotImplementedError

def finish_reasons(o: dict) -> list[str]:
    raise NotImplementedError

def _ordered(obs: list[dict]) -> list[tuple[str, dict]]:
    raise NotImplementedError

def _agent(o: dict) -> str:
    raise NotImplementedError

def transcript_lines(obs: list[dict]) -> dict[str, list[dict]]:
    """One session's observations as Claude Code transcript lines, keyed by
    "" (the main agent) or the subagent's parent tool call id."""
    raise NotImplementedError

def langfuse_hits(evs: list[tuple[str, dict]]) -> dict[int, list[Signal]]:
    raise NotImplementedError

def langfuse_event(kind: str, o: dict) -> Ev:
    raise NotImplementedError
SOURCE_IDS = ('sourceEventId', 'sourceUsageRowId')

def dedupe(obs: list[dict]) -> list[dict]:
    """One observation per exported source record: a session the bridge
    exported twice (two traces) counts once."""
    raise NotImplementedError

def langfuse_sessions(obs: list[dict]) -> dict[str, list[dict]]:
    raise NotImplementedError

def transcript_rel(label: str, session: str, agent: str) -> str:
    raise NotImplementedError

def write_transcripts(obs: list[dict], label: str, root: Path) -> dict[str, str]:
    """Writes every session's transcripts under root; returns transcript -> trace id."""
    raise NotImplementedError

def langfuse_episodes(obs: list[dict], label: str) -> list[dict]:
    """The generation and turn classes, per session and agent."""
    raise NotImplementedError

def traces_episodes(obs: list[dict], label: str, work: Path, exe: Path) -> tuple[list[dict], dict]:
    """The nine transcript classes: the sessions as transcripts, mined by
    `rrsi-mine traces` unchanged; each episode's refs are its trace id."""
    raise NotImplementedError
FIXES: dict[str, str] = {}
EXAMPLES = 2

def summarize(eps: list[dict], observations: dict[str, int]) -> dict:
    """Per source (an episode's project): observations and, per class,
    episodes, hits and up to EXAMPLES example ids from distinct episodes."""
    raise NotImplementedError

def ranked(summary: dict) -> list[dict]:
    """One row per (class, source), most episodes first; rate per 1k observations of that source."""
    raise NotImplementedError

def table(rows: list[dict]) -> str:
    raise NotImplementedError
HTTP_TIMEOUT_S = 180
HTTP_ATTEMPTS = 4
BACKOFF_S = 5
LITELLM_LOGS = '/spend/logs/v2'
LITELLM_PAGE = 1000
LITELLM_DATE = '%Y-%m-%d %H:%M:%S'
LITELLM_WINDOW = timedelta(hours=1)
LITELLM_MIN_WINDOW = timedelta(minutes=1)
LITELLM_SPLIT = 4
LANGFUSE_OBSERVATIONS = '/api/public/v2/observations'
LANGFUSE_PAGE = 1000
LANGFUSE_FIELDS = 'core,basic,time,io,metadata,model,usage'
Get = Callable[[str, dict[str, str]], dict]

def http_get(url: str, headers: dict[str, str]) -> dict:
    """GET JSON; transient failures (timeouts, 429, 5xx) are retried with backoff."""
    raise NotImplementedError

def _window_url(base: str, a: datetime, b: datetime, page: int) -> str:
    raise NotImplementedError

def fetch_window(base: str, headers: dict[str, str], a: datetime, b: datetime, get: Get, log: Callable[[str], None]) -> list[dict]:
    """Every row of [a, b]; a window the server caps is split, never truncated."""
    raise NotImplementedError

def fetch_litellm(base: str, key: str, since: datetime, until: datetime, raw: Path, jobs: int=3, get: Get=http_get, log: Callable[[str], None]=print) -> dict:
    """Projected request logs of [since, until) into raw/litellm.jsonl and the
    model groups' input limits into raw/litellm-models.json."""
    raise NotImplementedError

def fetch_langfuse(host: str, public: str, secret: str, label: str, raw: Path, get: Get=http_get, log: Callable[[str], None]=print) -> dict:
    """Every observation of the key's project into raw/langfuse-<label>.jsonl."""
    raise NotImplementedError

def read_jsonl(p: Path) -> list[dict]:
    raise NotImplementedError

def mine_raw(raw: Path, out: Path, work: Path, exe: Path, gateway_label: str) -> dict:
    """Every source in raw/ -> episodes in `out` and their summary next to it."""
    raise NotImplementedError

def _date(s: str) -> datetime:
    raise NotImplementedError

def _env(name: str) -> str:
    raise NotImplementedError

def main(argv: list[str] | None=None) -> int:
    raise NotImplementedError
if __name__ == '__main__':
    sys.exit(main())
