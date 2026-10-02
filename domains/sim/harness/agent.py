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
"""The driving-controller agent: a JSON-action loop that submits one controller.

Interface contract (the runner depends on it; do not change the signature):

    run_agent(brief, chat, check, rollout, max_steps) -> dict

  brief                                              the scenario, in words
  chat(messages, json_mode=False) -> (text, tokens)  the frozen policy
  check(controller) -> {"ok": bool, "error": str}    CSF admission of a controller
  rollout(controller) -> dict                        one episode on a cheap
                                                     surrogate plant (never the
                                                     graded simulator)
  max_steps                                          injected by the runner

The returned dict must hold "controller" (the submitted Controller JSON object,
or None), "messages" (the full conversation) and "tokens" (policy tokens).
"""

from __future__ import annotations

import json

from .prompts import SYSTEM_PROMPT, TASK_TEMPLATE


def _parse(text: str):
    try:
        act = json.loads(text)
    except json.JSONDecodeError:
        return None
    return act if isinstance(act, dict) else None


def run_agent(brief, chat, check, rollout, max_steps):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TASK_TEMPLATE.format(brief=brief)}]
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
            return {"controller": act.get("controller"), "messages": messages, "tokens": tokens}
        if act.get("action") == "check":
            res = check(act.get("controller"))
            messages.append({"role": "user", "content": "admitted" if res["ok"] else f"rejected: {res['error']}"})
            continue
        messages.append({"role": "user", "content": "ERROR: unknown action."})
    return {"controller": None, "messages": messages, "tokens": tokens}
