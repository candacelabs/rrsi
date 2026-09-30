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

  copilot  `copilot -s --available-tools "" --disable-builtin-mcps --model M`
           No tools: the reply is text only.
  codex    `codex exec --ephemeral --sandbox read-only -m M -o FILE -`
           Codex keeps its shell tool; the read-only sandbox and the empty
           working directory are what bound it.

The CLIs do not report token usage in this mode; `estimate_tokens` gives a
characters/4 estimate for the cost rule.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

TIMEOUT = float(os.environ.get("RRSI_CLI_TIMEOUT", "900"))
REASONING = os.environ.get("RRSI_CLI_REASONING", "low")
CLIS = ("copilot", "codex")


class CLIError(RuntimeError):
    """The CLI exited non-zero, timed out or printed nothing."""


def estimate_tokens(*texts: str) -> int:
    return sum(len(t or "") for t in texts) // 4


def _join(system: str | None, prompt: str) -> str:
    return (f"{system.strip()}\n\n---\n\n{prompt}" if system else prompt)


def complete(cli: str, model: str, system: str | None, prompt: str,
             timeout: float = TIMEOUT) -> str:
    if cli not in CLIS:
        raise CLIError(f"unknown CLI {cli!r}; expected one of {CLIS}")
    if not model:
        raise CLIError(f"{cli} needs a model name")
    text = _join(system, prompt)
    with tempfile.TemporaryDirectory(prefix=f"rrsi-{cli}-") as d:
        if cli == "copilot":
            cmd = ["copilot", "-s", "--no-color", "--available-tools", "",
                   "--disable-builtin-mcps", "--model", model,
                   "--reasoning-effort", REASONING]
            out_file = None
        else:
            out_file = Path(d) / "last.txt"
            cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
                   "--sandbox", "read-only", "--color", "never", "-m", model,
                   "-c", f"model_reasoning_effort={REASONING}", "-o", str(out_file), "-"]
        try:
            r = subprocess.run(cmd, input=text, cwd=d, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CLIError(f"{cli} timed out after {timeout}s") from e
        reply = (out_file.read_text() if out_file and out_file.exists() else r.stdout).strip()
        if r.returncode != 0 or not reply:
            raise CLIError(f"{cli} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        return reply
