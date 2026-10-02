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
"""Campaign-validity backstop: did any model call use a tool?

CSF is Candace Labs' Go framework for AI-agent systems. The policy and search
models run through logged-in agent CLIs. `rrsi/cli_llm.py` already runs every
call with no tools and voids a reply if the CLI's event stream shows a tool
request or execution, so a tool call becomes infra, never a graded result.

This tool is the independent, after-the-fact check on that: it scans a CLI's
own session-event logs for tool-execution events in a time window and reports
any it finds. A nonzero count voids the evaluations that overlap the window --
a graded verdict must never rest on something a model read with a tool.

Everything is a path argument; no private path, host or salt is baked in.

    python3 validity.py --session-logs DIR --since ISO --until ISO [--model M]
                        [--json OUT]

`--session-logs` is a directory of JSON-lines event files (for example a
Copilot CLI `session-state` tree). An event is a tool use when its `type`
starts with `tool.`, or it is an `assistant.message` carrying a non-empty
`toolRequests`, or (Codex) a completed `item` whose `type` is not
`agent_message`/`reasoning`. `--since`/`--until` are ISO-8601 timestamps.
Exit status is 1 when any tool use is found in the window, else 0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_TOOL_ITEM_KINDS_OK = {"agent_message", "reasoning"}


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp


def _event_time(event: dict) -> dt.datetime | None:
    for key in ("timestamp", "time", "recorded_at", "ts", "createdAt"):
        stamp = _parse_ts(event.get(key))
        if stamp:
            return stamp
    data = event.get("data")
    if isinstance(data, dict):
        return _event_time(data)
    return None


def is_tool_use(event: dict) -> bool:
    kind = str(event.get("type", ""))
    if kind.startswith("tool."):
        return True
    data = event.get("data") or {}
    if kind == "assistant.message" and data.get("toolRequests"):
        return True
    item = event.get("item") or {}
    item_kind = item.get("type")
    if event.get("type") in ("item.completed", "item.started") and item_kind \
            and item_kind not in _TOOL_ITEM_KINDS_OK:
        return True
    return False


def _model_of(event: dict) -> str:
    data = event.get("data") or {}
    return str(data.get("model") or event.get("model") or "")


def scan(paths: list[Path], since: dt.datetime | None, until: dt.datetime | None,
         model: str | None) -> dict:
    hits: list[dict] = []
    files = 0
    for path in paths:
        for log in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if not log.is_file():
                continue
            files += 1
            for line in log.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict) or not is_tool_use(event):
                    continue
                when = _event_time(event)
                if since and when and when < since:
                    continue
                if until and when and when > until:
                    continue
                if model and _model_of(event) and _model_of(event) != model:
                    continue
                hits.append({"file": str(log), "type": event.get("type"),
                             "model": _model_of(event), "when": when.isoformat() if when else None})
    return {"files_scanned": files, "tool_uses": len(hits), "valid": not hits, "hits": hits[:50]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="void a campaign window if any model call used a tool")
    ap.add_argument("--session-logs", action="append", required=True,
                    help="a directory or file of JSON-lines session events (repeatable)")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--model", help="restrict to this model id")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    report = scan([Path(p) for p in a.session_logs], _parse_ts(a.since), _parse_ts(a.until), a.model)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    verdict = "VALID: no tool use in the window" if report["valid"] else \
        f"VOID: {report['tool_uses']} tool use(s) in the window"
    print(f"{verdict} ({report['files_scanned']} log files)")
    for hit in report["hits"][:10]:
        print(f"  {hit['when']} {hit['model']} {hit['type']} {hit['file']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
