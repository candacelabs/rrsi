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
"""The frozen policy LLM behind the toy harness, injected by the runner.

RRSI_POLICY_BACKEND selects the transport:

  openai     an OpenAI-compatible /chat/completions server at
             RRSI_POLICY_BASE_URL (default RRSI_OPENAI_BASE_URL)
  anthropic  the Anthropic Messages API with ANTHROPIC_API_KEY

RRSI_POLICY_MODEL names the model. The harness receives `chat` as a plain
callable; it cannot change the model, the backend or the sampling settings.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BACKEND = os.environ.get("RRSI_POLICY_BACKEND", "openai").strip().lower()
MODEL = os.environ.get("RRSI_POLICY_MODEL", "")
MAX_TOKENS = int(os.environ.get("RRSI_POLICY_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("RRSI_POLICY_TEMPERATURE", "0.7"))


class PolicyError(RuntimeError):
    """The policy endpoint failed after retries (infrastructure, not the harness)."""


def _post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code == 400:
                break
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    raise PolicyError(last)


def chat(messages: list[dict], json_mode: bool = False) -> tuple[str, int]:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    -> (reply text, total tokens of this call)."""
    if not MODEL:
        raise PolicyError("set RRSI_POLICY_MODEL")
    if BACKEND == "anthropic":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        out = _post(os.environ.get("RRSI_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
                    + "/v1/messages", body,
                    {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                     "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in out.get("content", []))
        u = out.get("usage") or {}
        return text, int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
    base = (os.environ.get("RRSI_POLICY_BASE_URL")
            or os.environ.get("RRSI_OPENAI_BASE_URL", "")).rstrip("/")
    body = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = ({"Authorization": f"Bearer {os.environ['RRSI_OPENAI_API_KEY']}"}
               if os.environ.get("RRSI_OPENAI_API_KEY") else {})
    out = _post(f"{base}/chat/completions", body, headers)
    text = out["choices"][0]["message"].get("content") or ""
    return text, int((out.get("usage") or {}).get("total_tokens", 0))
