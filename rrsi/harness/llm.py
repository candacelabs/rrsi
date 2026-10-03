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
"""Structured (JSON) completions for the harness miner.

    complete_json(backend, model, system, prompt, schema, cwd) -> dict

Backends:

  sdk      The Claude Agent SDK (`claude-agent-sdk`) on the logged-in Claude
           Code: no API key, no tools, no settings or CLAUDE.md, no session
           persistence (so the miner's own calls never become transcripts it
           mines next time), structured output against `schema`.
  copilot  `rrsi.cli_llm` (the logged-in Copilot CLI); the schema goes in the
           prompt and the reply is parsed as JSON.
  codex    the same through the Codex CLI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_MODEL = {"sdk": "claude-opus-5-5", "copilot": "claude-opus-5.5", "codex": "gpt-5.6-luna"}
BACKENDS = tuple(DEFAULT_MODEL)


class LLMError(RuntimeError):
    """The backend failed or returned no parseable JSON."""


class LLMAuthError(LLMError):
    """The backend is not logged in; every further call would fail the same way."""


#: Failure text a retry can cure (a flaky network or proxy, a busy backend). The
#: Copilot CLI's generic help mentions "re-authenticate" on such failures too, so
#: these are checked before the auth markers.
TRANSIENT = ("timed out", "timeout", "econnreset", "econnrefused", "socket hang up", "network error",
             "proxy", "rate limit", "429", "502", "503", "504")


def is_transient(e: BaseException | str) -> bool:
    return any(t in str(e).lower() for t in TRANSIENT)


def _raise(msg: str) -> None:
    low = msg.lower()
    if not is_transient(low) and ("authenticat" in low or "not logged in" in low):
        raise LLMAuthError(f"{msg} (log the CLI in, or pick another --backend)")
    raise LLMError(msg)


def parse_json(text: str) -> dict:
    """The first JSON object in `text` (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    out = None
    for cand in ([m.group(1)] if m else []) + [text[text.find("{"):]]:
        try:  # raw_decode ignores trailing prose after the object
            out = json.JSONDecoder(strict=False).raw_decode(cand)[0]
            break
        except json.JSONDecodeError:
            continue
    if out is None:
        raise LLMError(f"reply is not JSON ({len(text)} chars): {text[:200]!r}", text)
    if not isinstance(out, dict):
        raise LLMError("reply is not a JSON object")
    return out


def _sdk(model: str, system: str, prompt: str, schema: dict, cwd: Path, effort: str) -> dict:
    import anyio
    import claude_agent_sdk as sdk

    opts = sdk.ClaudeAgentOptions(
        model=model, system_prompt=system, tools=[], setting_sources=[], cwd=str(cwd),
        output_format={"type": "json_schema", "schema": schema}, effort=effort, max_turns=4,
        extra_args={"no-session-persistence": None})

    async def run() -> dict:
        result = None
        async for m in sdk.query(prompt=prompt, options=opts):
            if isinstance(m, sdk.ResultMessage):
                result = m
        if result is None:
            raise LLMError("no result message")
        if result.is_error:
            _raise(f"sdk error: {result.result}")
        if isinstance(result.structured_output, dict):
            return result.structured_output
        return parse_json(result.result or "")

    try:
        return anyio.run(run)
    except LLMError:
        raise
    except Exception as e:  # the SDK raises its own process/result errors
        _raise(f"sdk: {e}")


def complete_json(backend: str, model: str | None, system: str, prompt: str, schema: dict,
                  cwd: Path, effort: str = "medium") -> dict:
    if backend not in BACKENDS:
        raise LLMError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    model = model or DEFAULT_MODEL[backend]
    if backend == "sdk":
        return _sdk(model, system, prompt, schema, cwd, effort)
    from rrsi import cli_llm
    ask = (f"{prompt}\n\nReply with ONE JSON object only, no prose, matching this JSON Schema:\n"
           f"{json.dumps(schema)}")
    try:
        return parse_json(cli_llm.complete(backend, model, system, ask))
    except cli_llm.CLIError as e:
        _raise(str(e))
