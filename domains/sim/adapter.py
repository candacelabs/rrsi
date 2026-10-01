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
"""Simulation instance: driving controllers on CSF scenarios, graded in a simulator.

CSF ("Cerebrospinal Fluid") is Candace Labs' Go framework for AI-agent
systems, in developer preview and not yet publicly released; this domain uses
its scenario and controller messages, its bounded controller compiler and
runtime, and its simulator workers (README.md says more).

Tasks are scenarios, not commits: 24 practice (evolve) and 8 held-out
straight-path scenarios (data/scenarios.py). The agent (frozen policy +
evolvable harness) proposes one controller per scenario; CSF checks it with its
bounded controller compiler, executes it on the selected backend (fake, CPU
HighwayEnv or CARLA 0.9.16; bench/backends.py) and RRSI grades the episode from
CSF's events.jsonl by four oracles (bench/oracles.py). Reward = oracle pass
fraction. Cost = policy tokens + `sim_second_tokens` x simulated seconds.

A simulator or runtime failure is Infra: the trial gets no verdict, is retried
once, and otherwise counts as missing; it is never scored as the controller's
failure. Trial i of task X in job J is runs/sim/jobs/J/X/t<i>/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
import briefs                            # noqa: E402
import render                            # noqa: E402
from scenarios import BY_ID, EVOLVE, HELDOUT_IDS, brief  # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text())


class SimDomain(Domain):
    name = "sim"
    harness_path = "harness"
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    critic_patterns = [
        (r"scenarios\.py|oracles\.py|\bBY_ID\b|\bHELDOUT|\bEVOLVE\b|\bPRACTICE\b|verdict\.json|"
         r"events\.jsonl|manifest\.json|run_tasks|backends\.py|data/",
         "reaches for scenario data, the oracles, the grader or evidence"),
        # Seeds are ordinary integers (a gain of -300 is not seed 300), so only
        # the scenario-name form and an explicit seed comparison are flagged;
        # the LLM critic covers numbers copied from a specific scenario.
        (r"straight-\d+|\bseed\b\s*(?:==|=|:|in\b)",
         "scenario id or seed hardcoded in the scaffold"),
        (r"urllib|requests\.|socket\.|http://|https://|pip install|subprocess|os\.system",
         "network, package install or process escape from the harness"),
    ]
    component_signals = [
        ("output_plumbing", [r"_parse\(", r"json\.loads", r"fence", r"```"]),
        ("control_flow",    [r"def run_agent", r"max_steps", r"\"submit\"", r"fallback",
                             r"for _ in range"]),
        ("tool_use",        [r"rollout\(", r"check\("]),
        ("controller_template", [r"OPCODE_", r"def _controller", r"gains?"]),
        ("prompt",          [r"prompts\.py", r"SYSTEM_PROMPT", r"TASK_TEMPLATE"]),
    ]

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(EVOLVE)

    def heldout_ids(self) -> list[str]:
        return list(HELDOUT_IDS)

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        n = CFG.get("smoke_n", 2)
        ids = self.evolve_ids()
        if incumbent_per_task:
            solid = [i for i in ids if i in incumbent_per_task
                     and min(incumbent_per_task[i].rewards or [0]) >= 1.0]
            if len(solid) >= n:
                return solid[:n]
        return ids[:n]

    # ---- Evaluate ----------------------------------------------------------
    def _cmd(self, root: Path, runs_dir: Path, job: str, ids: list[str], k: int,
             backend: str | None = None) -> list[str]:
        cmd = [sys.executable, str(root / "domains" / "sim" / "bench" / "run_tasks.py"),
               "--runs", str(runs_dir), "--job", job, "--ids", ",".join(ids), "--n", str(k)]
        return cmd + (["--backend", backend] if backend else [])

    def run(self, root, runs_dir, job, ids, k, log_prefix="", backend=None):
        root, runs_dir = Path(root), Path(runs_dir)
        log = runs_dir / "logs" / f"{log_prefix or job}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        # A policy-endpoint failure or a simulator crash leaves a trial without
        # a verdict; one retry pass fills them before scoring counts them missing.
        for _ in range(2):
            with open(log, "a") as lf:
                r = subprocess.run(self._cmd(root, runs_dir, job, ids, k, backend),
                                   cwd=str(root / "domains" / "sim"), stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"[sim] WARNING run rc={r.returncode} (see {log})", flush=True)
            if all((runs_dir / "jobs" / job / t / f"t{i}" / "verdict.json").is_file()
                   for t in ids for i in range(k)):
                break

    def _records(self, runs_dir: Path, job: str, tid: str) -> list[dict]:
        out = []
        tdir = runs_dir / "jobs" / job / tid
        for trial in sorted(tdir.glob("t*")) if tdir.is_dir() else []:
            try:
                v = json.loads((trial / "verdict.json").read_text())
                meta = json.loads((trial / "meta.json").read_text())
            except Exception:  # noqa: BLE001 - missing, partial or infra trial
                continue
            out.append({"reward": float(v.get("reward") or 0.0), "status": v.get("status"),
                        "oracles": v.get("oracles") or {}, "crash": meta.get("status") == "crash",
                        "tokens": meta.get("tokens"), "sim_seconds": float(v.get("simulation_seconds") or 0.0)})
        return out

    def _infra(self, runs_dir: Path, job: str, ids: list[str], k: int) -> int:
        return sum(1 for t in ids for i in range(k)
                   if not (runs_dir / "jobs" / job / t / f"t{i}" / "verdict.json").is_file())

    def score(self, runs_dir, job, ids, k):
        runs_dir = Path(runs_dir)
        per_second = float(CFG.get("sim_second_tokens", 20))
        per, full, nosub, rejected, crash, sim_s, toks_only = {}, 0, 0, 0, 0, 0.0, []
        oracle_pass: dict = {}
        for t in ids:
            recs = self._records(runs_dir, job, t)[:k]
            missing = k - len(recs)
            rewards = [r["reward"] for r in recs] + [0.0] * missing
            costs = []
            for r in recs:
                tok = r["tokens"] if isinstance(r["tokens"], int) and r["tokens"] > 0 else None
                costs.append(None if tok is None else tok + per_second * r["sim_seconds"])
                toks_only += [tok] if tok else []
                sim_s += r["sim_seconds"]
                for name, ok in r["oracles"].items():
                    oracle_pass[name] = oracle_pass.get(name, 0) + int(bool(ok))
            full += sum(1 for r in recs if r["reward"] >= 1.0)
            nosub += sum(1 for r in recs if r["status"] == "no_submission")
            rejected += sum(1 for r in recs if r["status"] == "rejected")
            crash += sum(1 for r in recs if r["crash"])
            per[t] = TaskResult(rewards=rewards, tokens=costs + [None] * missing, missing=missing)
        n = max(1, len(ids) * k)
        return per, {"full_passes": full, "n_trials": len(ids) * k,
                     "mean_reward": sum(sum(tr.rewards) for tr in per.values()) / n,
                     "oracle_pass_rate": {o: c / n for o, c in sorted(oracle_pass.items())},
                     "no_submission_rate": nosub / n, "rejected_rate": rejected / n,
                     "crash_rate": crash / n, "infra_rate": self._infra(runs_dir, job, ids, k) / n,
                     "sim_seconds": round(sim_s, 1),
                     "mean_policy_tokens": (sum(toks_only) / len(toks_only)) if toks_only else None,
                     "backend": os.environ.get("RRSI_SIM_BACKEND", "fake")}

    def guards(self, incumbent, candidate) -> list[str]:
        out = []
        lim = CFG.get("max_crash_rate_rise", 0.05)
        d = (candidate.extra.get("crash_rate") or 0.0) - (incumbent.extra.get("crash_rate") or 0.0)
        if d > lim:
            out.append(f"harness crash rate rose {d:+.3f} (limit {lim})")
        inf = candidate.extra.get("infra_rate") or 0.0
        if inf > CFG.get("max_infra_rate", 0.2):
            out.append(f"simulator infra rate {inf:.3f}: re-measure (`reevaluate`), not a harness verdict")
        return out

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        task = BY_ID[task_id]
        return render.load_trial(Path(runs_dir) / "jobs" / job, task, brief(task), trial)

    def render_trace(self, rec, detail=False):
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        meta, v = rec.get("meta") or {}, rec.get("verdict") or {}
        msgs = (rec.get("traj") or {}).get("messages") or []
        checks = sum(1 for m in msgs if m.get("role") == "assistant" and '"check"' in str(m.get("content")))
        return (f"{task_id} | {render.failure_class(rec)} | reward={tr.mean:.2f} | "
                f"steps={meta.get('steps')} checks={checks} tokens={meta.get('tokens')} | "
                f"termination={v.get('termination')} sim_s={v.get('simulation_seconds')}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        """Liveness on the surrogate backend: compile, interface, two real trials."""
        root, runs_dir = Path(root), Path(runs_dir)
        sim = root / "domains" / "sim"
        comp = subprocess.run([sys.executable, "-m", "compileall", "-q",
                               str(self.harness_dir(root))], capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "err": (comp.stdout + comp.stderr)[-1500:]}
        code = ("import inspect\n"
                "from harness.agent import run_agent\n"
                "from harness.prompts import TASK_TEMPLATE\n"
                "assert list(inspect.signature(run_agent).parameters) == "
                "['brief', 'chat', 'check', 'rollout', 'max_steps'], 'signature changed'\n"
                "assert 'SENTINEL-B' in TASK_TEMPLATE.format(brief='SENTINEL-B'), 'template drops the brief'\n"
                "print('CTOR OK')\n")
        ctor = subprocess.run([sys.executable, "-c", code], cwd=str(sim), capture_output=True, text=True)
        if ctor.returncode != 0 or "CTOR OK" not in ctor.stdout:
            return False, {"stage": "ctor", "err": (ctor.stderr or ctor.stdout)[-1500:]}
        shutil.rmtree(runs_dir / "jobs" / job, ignore_errors=True)
        sys.path.insert(0, str(HERE / "bench"))
        import backends
        self.run(root, runs_dir, job, ids, 1, job, backend=backends.surrogate_name())
        per, extra = self.score(runs_dir, job, ids, 1)
        missing = sum(tr.missing for tr in per.values())
        ok = missing == 0 and extra["crash_rate"] == 0.0
        return ok, {"stage": "run", "smoke_ids": ids, "missing": missing, **extra}


DOMAIN = SimDomain()
