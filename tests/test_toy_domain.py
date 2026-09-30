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
"""Toy domain and LLM backends: no API, no container.

    python3 -m pytest -q tests/test_toy_domain.py
"""

import http.server
import importlib
import inspect
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "domains" / "toy"
os.environ["RRSI_TOY_SANDBOX"] = "local"
for p in (str(ROOT), str(TOY), str(TOY / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import reference                          # noqa: E402
import tasks                              # noqa: E402
from harness.agent import run_agent       # noqa: E402
from rrsi.domain import load_domain       # noqa: E402


def test_every_hidden_test_is_passable():
    bad = []
    for t in tasks.TASKS:
        src = inspect.getsource(reference)
        ns = {}
        exec(compile(src, "reference.py", "exec"), ns)
        try:
            exec(t["tests"], {t["entry"]: ns[t["entry"]]})
        except Exception as e:  # noqa: BLE001
            bad.append((t["id"], repr(e)))
    assert not bad, bad


def test_splits_are_disjoint_and_complete():
    assert len(tasks.EVOLVE) == 20 and len(tasks.HELDOUT) == 10
    assert not set(tasks.EVOLVE) & set(tasks.HELDOUT)
    assert set(tasks.EVOLVE) | set(tasks.HELDOUT) == set(tasks.BY_ID)


def test_grader_passes_reference_and_fails_stub():
    import sandbox  # noqa: F401
    sys.path.insert(0, str(TOY / "bench"))
    run_tasks = importlib.import_module("run_tasks")
    task = tasks.BY_ID["rle_encode"]
    good = inspect.getsource(reference.rle_encode)
    assert run_tasks.grade(task, "import re\n" + good)["passed"]
    v = run_tasks.grade(task, "def rle_encode(s):\n    return ''\n")
    assert not v["passed"] and "AssertionError" in v["output"]
    assert run_tasks.grade(task, None)["status"] == "no_submission"


def test_starting_harness_loop_with_scripted_policy():
    replies = iter([
        json.dumps({"action": "run_python", "code": "print(1 + 1)"}),
        "not json",
        json.dumps({"action": "submit", "code": "def f():\n    return 1\n"}),
    ])
    calls = []

    def chat(messages, json_mode=False):
        calls.append(len(messages))
        return next(replies), 10

    def run_python(code, timeout=10):
        return {"ok": True, "exit": 0, "stdout": "2\n", "stderr": "", "timeout": False}

    out = run_agent("spec", "f", chat, run_python, max_steps=5)
    assert out["code"].startswith("def f") and out["tokens"] == 30
    assert any("ERROR" in m["content"] for m in out["messages"] if m["role"] == "user")
    out = run_agent("spec", "f", lambda m, json_mode=False: ("{}", 1), run_python, 2)
    assert out["code"] is None and out["tokens"] == 2


def test_adapter_scores_and_renders_a_job():
    dom = load_domain("toy")
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        for tid, passed in (("rle_encode", True), ("flatten", False)):
            t = runs / "jobs" / "j" / tid / "t0"
            t.mkdir(parents=True)
            (t / "meta.json").write_text(json.dumps({"status": "ok", "tokens": 100, "steps": 2}))
            (t / "traj.json").write_text(json.dumps({"messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": '{"action": "submit", "code": "x"}'}]}))
            (t / "solution.py").write_text("x")
            (t / "verdict.json").write_text(json.dumps(
                {"passed": passed, "status": "ok", "output": "" if passed else "AssertionError"}))
        per, extra = dom.score(runs, "j", ["rle_encode", "flatten", "median"], 1)
        assert per["rle_encode"].rewards == [1.0] and per["median"].missing == 1
        assert extra["total_passes"] == 1
        rec = dom.load_trial(runs, "j", "flatten", 0)
        text = dom.render_trace(rec)
        assert "WRONG-ANSWER" in text and "[step 1]" in text
        assert tasks.BY_ID["flatten"]["tests"].strip() not in text
        assert "flatten" in dom.task_row("flatten", rec, per["flatten"])


def test_critic_denylist_catches_task_leakage():
    from rrsi.critic import precheck
    dom = load_domain("toy")
    assert precheck("+ if entry == 'roman_to_int':", dom.critic_patterns)
    assert precheck("+ open('data/tasks.py')", dom.critic_patterns)
    assert not precheck("+ # run the examples from the specification first",
                        dom.critic_patterns)


def test_trim_middle_keeps_head_and_tail():
    from rrsi.llm import trim_middle
    s = "H" * 500 + "M" * 1000 + "T" * 500
    out = trim_middle(s, 800)
    assert len(out) <= 800 and out.startswith("H") and out.endswith("T")
    assert trim_middle("short", 800) == "short" and trim_middle(s, 0) == s


def test_openai_backend_against_fake_server(monkeypatch):
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"choices": [{"message": {
                "content": "```json\n{\"action\": \"done\"}\n```"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import rrsi.llm as llm
        monkeypatch.setattr(llm, "BACKEND", "openai")
        monkeypatch.setenv("RRSI_OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
        monkeypatch.setenv("RRSI_OPENAI_MODEL", "local-model")
        out = llm.generate("hello", system="sys", json_only=True, cache_prefix="stable")
        assert json.loads(out) == {"action": "done"}
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"]["model"] == "local-model"
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert seen["body"]["messages"][1]["content"].startswith("stable")
    finally:
        srv.shutdown()


def test_cli_backend_uses_stdin_and_reads_reply(monkeypatch, tmp_path):
    import subprocess
    from rrsi import cli_llm
    seen = {}

    def fake_run(cmd, input, cwd, capture_output, text, timeout):
        seen["cmd"], seen["input"] = cmd, input
        if cmd[0] == "codex":
            Path(cmd[cmd.index("-o") + 1]).write_text("codex reply\n")
            return subprocess.CompletedProcess(cmd, 0, "tokens used\n9\n", "")
        return subprocess.CompletedProcess(cmd, 0, "copilot reply\n", "")

    monkeypatch.setattr(cli_llm.subprocess, "run", fake_run)
    big = "x" * 300_000
    assert cli_llm.complete("copilot", "m", "SYS", big) == "copilot reply"
    assert seen["input"].startswith("SYS") and seen["input"].endswith(big)
    assert "--available-tools" in seen["cmd"] and big not in seen["cmd"]
    assert cli_llm.complete("codex", "m", None, "hi") == "codex reply"
    assert seen["cmd"][-1] == "-"
