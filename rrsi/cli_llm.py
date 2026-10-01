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
"""One-shot completions through locally logged-in agent CLIs, not an API key.

    complete("copilot", "claude-sonnet-5.5", system, prompt) -> text
    complete("codex", "gpt-5.6-luna", system, prompt) -> text

Each call is a fresh, stateless CLI process run in an empty scratch directory
with the prompt on stdin (a single argv string is capped at 128 KiB). The
system prompt is prepended to the user prompt because neither CLI takes one
separately.

Both CLIs are agents with built-in tools (a shell, file viewers, search), and
a model with a working shell can read anything the operator's user can,
including graded evidence. Every call therefore runs with no tools at all and
checks that from the CLI's own event stream:

  copilot  `copilot -s --available-tools=rrsi-no-tools ... --output-format json`
           An allowlist naming one tool that does not exist leaves the model
           no tool. (`--available-tools ""` is NOT that: Copilot CLI 1.0.90
           reads an empty list as "no restriction" and keeps bash, view, grep
           and the rest.) The call must report its built-in tools disabled.
  codex    `codex exec --ephemeral --sandbox read-only --disable shell_tool
           --disable unified_exec ... --json`

A reply is used only if the event stream shows no tool request and no tool
execution; otherwise the call raises `ToolUseError` and its reply is dropped.
`--no-auto-update` keeps the CLI version fixed for a whole run.

The CLIs do not report token usage in this mode; `estimate_tokens` gives a
characters/4 estimate for the cost rule.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

TIMEOUT = float(os.environ.get("RRSI_CLI_TIMEOUT", "900"))
REASONING = os.environ.get("RRSI_CLI_REASONING", "low")
CLIS = ("copilot", "codex")
# An allowlist holding only a name no tool has: the model gets no tool.
NO_TOOLS = "rrsi-no-tools"
# Built-in Copilot tools that must be reported disabled before a reply counts.
COPILOT_MUST_DISABLE = ("bash", "view", "grep", "glob", "edit", "create")
CODEX_DISABLE = ("shell_tool", "unified_exec", "view_image", "apps", "browser_use",
                 "computer_use", "skill_search", "tool_suggest", "sleep_tool")


class CLIError(RuntimeError):
    """The CLI exited non-zero, timed out or printed nothing."""


class ToolUseError(CLIError):
    """The model requested or ran a tool; the reply is void."""


def estimate_tokens(*texts: str) -> int:
    return sum(len(t or "") for t in texts) // 4


def _join(system: str | None, prompt: str) -> str:
    return (f"{system.strip()}\n\n---\n\n{prompt}" if system else prompt)


def command(cli: str, model: str) -> list[str]:
    """The argv of one tool-less call."""
    if cli == "copilot":
        return ["copilot", "-s", "--no-color", "--no-auto-update", f"--available-tools={NO_TOOLS}",
                "--disable-builtin-mcps", "--disallow-temp-dir", "--no-custom-instructions",
                "--output-format", "json", "--model", model, "--reasoning-effort", REASONING]
    disable = [arg for name in CODEX_DISABLE for arg in ("--disable", name)]
    return ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
            *disable, "--json", "--color", "never", "-m", model,
            "-c", f"model_reasoning_effort={REASONING}", "-"]


def _events(stdout: str) -> list[dict]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def copilot_reply(stdout: str) -> str:
    """The final assistant text of a Copilot JSONL stream, after checking
    that every built-in tool was disabled and none was requested or run."""
    events = _events(stdout)
    disabled: set[str] = set()
    for e in events:
        data = e.get("data") or {}
        message = str(data.get("message") or "")
        if e.get("type") == "session.info" and message.startswith("Disabled tools:"):
            disabled |= {t.strip() for t in message.split(":", 1)[1].split(",")}
        if str(e.get("type", "")).startswith("tool.") or (
                e.get("type") == "assistant.message" and data.get("toolRequests")):
            raise ToolUseError(f"copilot: the model used a tool ({e.get('type')}); reply discarded")
    missing = [t for t in COPILOT_MUST_DISABLE if t not in disabled]
    if missing:
        raise CLIError(f"copilot: could not confirm its built-in tools are disabled (not reported: "
                       f"{', '.join(missing)}); refusing the reply")
    texts = [str((e.get("data") or {}).get("content") or "") for e in events
             if e.get("type") == "assistant.message"]
    return texts[-1].strip() if texts else ""


def codex_reply(stdout: str) -> str:
    """The final agent message of a Codex JSONL stream; any other item kind
    (a command, a file change, a tool or web call) voids the reply."""
    text = ""
    for e in _events(stdout):
        item = e.get("item") or {}
        kind = item.get("type")
        if kind and kind not in ("agent_message", "reasoning"):
            raise ToolUseError(f"codex: the model used a tool ({kind}); reply discarded")
        if kind == "agent_message" and e.get("type") == "item.completed":
            text = str(item.get("text") or "")
    return text.strip()


def complete(cli: str, model: str, system: str | None, prompt: str,
             timeout: float = TIMEOUT) -> str:
    if cli not in CLIS:
        raise CLIError(f"unknown CLI {cli!r}; expected one of {CLIS}")
    if not model:
        raise CLIError(f"{cli} needs a model name")
    text = _join(system, prompt)
    with tempfile.TemporaryDirectory(prefix=f"rrsi-{cli}-") as d:
        try:
            r = subprocess.run(command(cli, model), input=text, cwd=d, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CLIError(f"{cli} timed out after {timeout}s") from e
        if r.returncode != 0:
            raise CLIError(f"{cli} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        reply = copilot_reply(r.stdout) if cli == "copilot" else codex_reply(r.stdout)
        if not reply:
            raise CLIError(f"{cli} printed no reply: {(r.stderr or r.stdout)[-600:]}")
        return reply
