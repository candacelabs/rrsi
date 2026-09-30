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
"""The toy coding agent: a JSON-action loop over one `run_python` tool.

Interface contract (the runner depends on it; do not change the signature):

    run_agent(prompt, entry, chat, run_python, max_steps) -> dict

  chat(messages, json_mode=False) -> (text, tokens)   the frozen policy
  run_python(code, timeout=10) -> {"ok", "exit", "stdout", "stderr", "timeout"}
  max_steps                                           injected by the runner

The returned dict must hold "code" (the submitted module source, or None),
"messages" (the full conversation) and "tokens" (total policy tokens).
"""

from __future__ import annotations

import json

from .prompts import SYSTEM_PROMPT, TASK_TEMPLATE

TOOL_OUTPUT_CHARS = 2000


def _parse(text: str):
    try:
        act = json.loads(text)
    except json.JSONDecodeError:
        return None
    return act if isinstance(act, dict) else None


def run_agent(prompt, entry, chat, run_python, max_steps):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TASK_TEMPLATE.format(prompt=prompt, entry=entry)}]
    tokens = 0
    for _ in range(max_steps):
        text, used = chat(messages)
        tokens += used
        messages.append({"role": "assistant", "content": text})
        act = _parse(text)
        if act is None:
            messages.append({"role": "user", "content": "ERROR: reply with one JSON object."})
            continue
        if act.get("action") == "submit":
            return {"code": act.get("code"), "messages": messages, "tokens": tokens}
        if act.get("action") == "run_python":
            res = run_python(str(act.get("code", "")))
            out = (res["stdout"] + res["stderr"])[:TOOL_OUTPUT_CHARS]
            messages.append({"role": "user", "content": f"exit={res['exit']}\n{out}"})
            continue
        messages.append({"role": "user", "content": "ERROR: unknown action."})
    return {"code": None, "messages": messages, "tokens": tokens}
