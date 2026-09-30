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
"""Run untrusted Python: the policy's scratch code and the hidden-test grader.

RRSI_TOY_SANDBOX=docker (default) runs each program in a fresh
`python:3.12-slim` container with no network, a memory, CPU and process cap,
a read-only root and the program mounted read-only. `local` runs a plain
subprocess and exists for unit tests only: policy-written code then runs with
the caller's privileges.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

MODE = os.environ.get("RRSI_TOY_SANDBOX", "docker")
IMAGE = os.environ.get("RRSI_TOY_IMAGE", "python:3.12-slim")
DOCKER = os.environ.get("RRSI_TOY_DOCKER", "docker")


def run_python(code: str, timeout: int = 10) -> dict:
    """-> {"ok": bool, "exit": int|None, "stdout": str, "stderr": str, "timeout": bool}"""
    with tempfile.TemporaryDirectory(prefix="rrsi-toy-") as d:
        Path(d, "main.py").write_text(code)
        os.chmod(d, 0o755)
        os.chmod(Path(d, "main.py"), 0o644)
        if MODE == "local":
            cmd = ["python3", "-I", "main.py"]
            cwd = d
        else:
            cmd = [DOCKER, "run", "--rm", "--network", "none", "--memory", "512m",
                   "--cpus", "1", "--pids-limit", "64", "--read-only",
                   "--tmpfs", "/tmp:size=16m", "--user", "65534:65534",
                   "-v", f"{d}:/work:ro", "-w", "/work", IMAGE,
                   "timeout", "-s", "KILL", str(timeout), "python3", "-I", "main.py"]
            cwd = None
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                               timeout=timeout + 30)
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "exit": None, "stdout": (e.stdout or "")[-4000:]
                    if isinstance(e.stdout, str) else "", "stderr": "timed out",
                    "timeout": True}
        killed = r.returncode in (124, 137) or (MODE != "local" and r.returncode == -9)
        return {"ok": r.returncode == 0, "exit": r.returncode,
                "stdout": r.stdout[-8000:], "stderr": r.stderr[-8000:],
                "timeout": killed}
