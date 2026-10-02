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
"""Optional container isolation for the propose phase (the evolvable harness).

At a single Unix user the salt file, the grader modules and the runner's own
process are all reachable by a hostile harness: forbidden by the critic, but
not prevented. `RRSI_SIM_PROPOSE_SANDBOX=docker` turns the propose phase into
prevention by running it in a container where none of them exist:

  * the grader package (`domains/sim/grader/`) is shadowed by an empty mount,
    so its hidden physics and salt handling cannot be imported;
  * no `RRSI_SIM_PHYS_*` variable and no salt file are mounted or exported;
  * the container has its own PID namespace, so `/proc` holds no host process
    whose environment could carry the salt path;
  * only this job's own runs directory is writable, so a harness cannot read
    another job's evidence.

The policy still runs through the logged-in CLI, which needs the network and
the user's CLI credentials, so those are mounted read-only; this isolates the
grade, not those credentials. The checkout is mounted read-only at its own
path so imports resolve unchanged.

`docker_command` builds the argv; `wrap` applies it only when the mode is set.
This is container-first (see the repository guidance): no host user, unit or
firewall change.
"""

from __future__ import annotations

import os
from pathlib import Path


def enabled() -> str:
    return os.environ.get("RRSI_SIM_PROPOSE_SANDBOX", "").strip().lower()


def docker_command(inner: list[str], checkout: Path, job_dir: Path, workdir: Path,
                   image: str | None = None, docker: str | None = None,
                   home: Path | None = None) -> list[str]:
    """The `docker run` argv that runs `inner` in the isolation described above."""
    checkout = Path(checkout).resolve()
    job_dir = Path(job_dir).resolve()
    workdir = Path(workdir).resolve()
    home = Path(home or os.path.expanduser("~")).resolve()
    image = image or os.environ.get("RRSI_SIM_SANDBOX_IMAGE", "python:3.12-slim")
    docker = docker or os.environ.get("RRSI_SIM_DOCKER", "docker")
    uid, gid = os.getuid(), os.getgid()
    cmd = [docker, "run", "--rm", "-i", "--init",
           "--user", f"{uid}:{gid}",
           "--pids-limit", "512", "--memory", "2g",
           # The checkout, read-only; the grader package shadowed by an empty
           # read-only tmpfs so none of it can be imported.
           "-v", f"{checkout}:{checkout}:ro",
           "--tmpfs", f"{checkout / 'domains' / 'sim' / 'grader'}:ro,size=1k",
           # Only this job's runs directory is writable.
           "-v", f"{job_dir}:{job_dir}:rw",
           "-w", str(workdir),
           "-e", "HOME=" + str(home),
           "-e", "PYTHONDONTWRITEBYTECODE=1"]
    # The logged-in CLI's credentials, read-only, for the policy call.
    for rel in (".copilot", ".config/gh", ".codex"):
        src = home / rel
        if src.exists():
            cmd += ["-v", f"{src}:{src}:ro"]
    # Forward only non-secret configuration; never an RRSI_SIM_PHYS_* variable.
    for name, value in os.environ.items():
        if name.startswith("RRSI_SIM_PHYS_"):
            continue
        if name.startswith(("RRSI_", "COPILOT_", "GH_", "GITHUB_", "CODEX_")) or name in ("PATH",):
            cmd += ["-e", f"{name}={value}"]
    return [*cmd, image, *inner]


def wrap(inner: list[str], checkout: Path, job_dir: Path, workdir: Path) -> list[str]:
    """`inner` unchanged unless RRSI_SIM_PROPOSE_SANDBOX=docker selects the box."""
    if enabled() == "docker":
        return docker_command(inner, checkout, job_dir, workdir)
    return inner
