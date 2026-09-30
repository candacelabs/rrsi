# Copyright 2026 The rrsi Authors.
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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified 2026 by Candace Labs: added the anthropic and openai backends.
"""LLM client for the three search roles (proposer, analyst, critic).

`RRSI_LLM_BACKEND` selects the transport:

  vertex     (default) Claude via AnthropicVertex, round-robin over the GCP
             projects in RRSI_VERTEX_PROJECTS with retry-and-rotate on failure.
  anthropic  Claude via the Anthropic API (ANTHROPIC_API_KEY). The endpoint is
             RRSI_ANTHROPIC_BASE_URL, never an inherited ANTHROPIC_BASE_URL.
  openai     Any OpenAI-compatible /chat/completions server (vLLM, SGLang,
             llama.cpp, ...) at RRSI_OPENAI_BASE_URL. Every role uses
             RRSI_OPENAI_MODEL; the Claude model names in rrsi.json are ignored.

`RRSI_SEARCH_MODEL_OVERRIDE`, when set, replaces the model of every role on
the vertex and anthropic backends.

`cache_prefix` sends a large stable leading block (constitution + harness
source) as its own ephemeral-cached content block on the Claude backends so
repeated turns pay to read it once; the openai backend prepends it as text and
relies on the server's automatic prefix caching.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("RRSI_SEARCH_MODEL", "claude-opus-4-8")
BACKEND = os.environ.get("RRSI_LLM_BACKEND", "vertex").strip().lower()
MODEL_OVERRIDE = os.environ.get("RRSI_SEARCH_MODEL_OVERRIDE", "").strip()
_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")
    if p.strip()
]
MAX_TOKENS = 20_000
# Small local models serve short contexts. A prompt longer than this many
# characters keeps its head and tail and loses the middle (0 = never trim).
OPENAI_MAX_PROMPT_CHARS = int(os.environ.get("RRSI_OPENAI_MAX_PROMPT_CHARS", "0"))
OPENAI_MAX_TOKENS = int(os.environ.get("RRSI_OPENAI_MAX_TOKENS", "4096"))
OPENAI_TIMEOUT = float(os.environ.get("RRSI_OPENAI_TIMEOUT", "600"))

_clients: dict = {}
_clients_lock = threading.Lock()
_rr = itertools.count()
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _client_for(idx: int):
    if BACKEND == "anthropic":
        return _anthropic_client()
    from anthropic import AnthropicVertex
    if not _PROJECTS:
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                           "projects with Claude on Vertex AI enabled")
    with _clients_lock:
        c = _clients.get(idx)
        if c is None:
            c = AnthropicVertex(**_PROJECTS[idx])
            _clients[idx] = c
        return c


def _anthropic_client():
    from anthropic import Anthropic
    with _clients_lock:
        c = _clients.get("anthropic")
        if c is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("RRSI_LLM_BACKEND=anthropic needs ANTHROPIC_API_KEY")
            c = Anthropic(api_key=key, base_url=os.environ.get(
                "RRSI_ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
            _clients["anthropic"] = c
        return c


def trim_middle(text: str, limit: int) -> str:
    """Keep the head and the tail of `text` within `limit` characters."""
    if limit <= 0 or len(text) <= limit:
        return text
    marker = f"\n...[{len(text) - limit} characters trimmed to fit the context]...\n"
    keep = max(0, limit - len(marker))
    head = keep * 2 // 5
    return text[:head] + marker + text[len(text) - (keep - head):]


def _openai_chat(system: str, user: str, json_only: bool, max_tokens: int) -> str:
    base = os.environ.get("RRSI_OPENAI_BASE_URL", "").rstrip("/")
    mdl = os.environ.get("RRSI_OPENAI_MODEL", "")
    if not base or not mdl:
        raise RuntimeError("RRSI_LLM_BACKEND=openai needs RRSI_OPENAI_BASE_URL and "
                           "RRSI_OPENAI_MODEL")
    body = {"model": mdl, "max_tokens": min(max_tokens, OPENAI_MAX_TOKENS),
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": trim_middle(user, OPENAI_MAX_PROMPT_CHARS)}]}
    if json_only:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("RRSI_OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['RRSI_OPENAI_API_KEY']}"
    req = urllib.request.Request(f"{base}/chat/completions", method="POST",
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}") from e
    return out["choices"][0]["message"].get("content") or ""


def extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{") and not t.startswith("["):
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
        if start != -1:
            t = t[start:]
    if t and t[0] == "{" and not t.endswith("}"):
        end = t.rfind("}")
        if end != -1:
            t = t[:end + 1]
    if t and t[0] == "[" and not t.endswith("]"):
        end = t.rfind("]")
        if end != -1:
            t = t[:end + 1]
    return t


def generate(prompt: str, system: str | None = None, max_retries: int = 6,
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None) -> str:
    mdl = MODEL_OVERRIDE or model or MODEL
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")
    content = ([{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt}] if cache_prefix else prompt)
    n = max(1, len(_PROJECTS))
    start = next(_rr)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        idx = (start + attempt) % n
        try:
            if BACKEND == "openai":
                user = (cache_prefix + "\n\n" + prompt) if cache_prefix else prompt
                text = _openai_chat(sys_prompt, user, json_only, max_tokens)
            else:
                client = _client_for(idx)
                kwargs = {"model": mdl, "max_tokens": max_tokens,
                          "messages": [{"role": "user", "content": content}]}
                if sys_prompt:
                    kwargs["system"] = sys_prompt
                resp = client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content
                               if getattr(b, "type", "") == "text")
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - rotate to the other project
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(generate("Reply with exactly: OK", max_tokens=16))
