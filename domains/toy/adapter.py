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
"""Toy instance: short Python function tasks graded by hidden unit tests.

A cheap, self-contained domain that exercises the whole RRSI loop (baseline,
analyze, propose, critic, smoke, evaluate, select, fast-forward) against any
policy served by an OpenAI-compatible endpoint or the Anthropic API. It is a
prototype for the method, not a benchmark: 20 evolve tasks and 10 held out.

Evaluate = bench/run_tasks.py from the candidate's worktree (k trials per task,
resume-safe), which grades each submission in the sandbox. Trial i of task X in
job J is runs/toy/jobs/J/X/t<i>/{traj.json, meta.json, solution.py, verdict.json}.
"""

from __future__ import annotations

import json
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
from tasks import BY_ID, EVOLVE, HELDOUT  # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text())


class ToyDomain(Domain):
    name = "toy"
    harness_path = "harness"
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    critic_patterns = [
        (r"tasks\.py|reference\.py|\bBY_ID\b|\bHELDOUT\b|\bEVOLVE\b|hidden tests? source|"
         r"data/|verdict\.json|run_tasks", "reaches for task data, references or the grader"),
        # Ids that are ordinary English words (median, rotate, ...) would
        # flag innocent prose; the distinctive ids are enough of a tripwire
        # and the LLM review covers the rest.
        (r"\b(?:" + "|".join(sorted(i for i in BY_ID if "_" in i or i == "levenshtein"))
         + r")\b",
         "task id or function name from the suite hardcoded in the scaffold"),
        (r"urllib|requests\.|socket\.|http://|https://|pip install|subprocess",
         "network, package install or process escape from the harness"),
    ]
    component_signals = [
        ("output_plumbing", [r"TOOL_OUTPUT_CHARS", r"truncat", r"_parse\(", r"json\.loads",
                             r"fence", r"```"]),
        ("control_flow",    [r"def run_agent", r"max_steps", r"\"submit\"", r"fallback",
                             r"for _ in range"]),
        ("context_mgmt",    [r"messages\[", r"summar", r"compact"]),
        ("config",          [r"= *\d+ *#", r"_CHARS\b", r"_LIMIT\b"]),
        ("prompt",          [r"prompts\.py", r"SYSTEM_PROMPT", r"TASK_TEMPLATE"]),
    ]

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(EVOLVE)

    def heldout_ids(self) -> list[str]:
        return list(HELDOUT)

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
    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        root, runs_dir = Path(root), Path(runs_dir)
        toy = root / "domains" / "toy"
        log = runs_dir / "logs" / f"{log_prefix or job}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(toy / "bench" / "run_tasks.py"), "--runs", str(runs_dir),
               "--job", job, "--ids", ",".join(ids), "--n", str(k)]
        # Policy endpoint failures leave a trial without a verdict; one retry
        # pass fills them before scoring counts them as missing.
        for _ in range(2):
            with open(log, "a") as lf:
                r = subprocess.run(cmd, cwd=str(toy), stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"[toy] WARNING run rc={r.returncode} (see {log})", flush=True)
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
            except Exception:  # noqa: BLE001 - missing or partial trial
                continue
            out.append({"passed": bool(v.get("passed")), "status": v.get("status"),
                        "crash": meta.get("status") == "crash",
                        "tokens": meta.get("tokens")})
        return out

    def score(self, runs_dir, job, ids, k):
        runs_dir = Path(runs_dir)
        per, passes, nosub, crash = {}, 0, 0, 0
        for t in ids:
            recs = self._records(runs_dir, job, t)[:k]
            missing = k - len(recs)
            rewards = [1.0 if r["passed"] else 0.0 for r in recs] + [0.0] * missing
            toks = [r["tokens"] if isinstance(r["tokens"], int) and r["tokens"] > 0 else None
                    for r in recs] + [None] * missing
            passes += int(sum(rewards))
            nosub += sum(1 for r in recs if r["status"] == "no_submission")
            crash += sum(1 for r in recs if r["crash"])
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing)
        n = max(1, len(ids) * k)
        return per, {"total_passes": passes, "n_trials": len(ids) * k, "pass_rate": passes / n,
                     "no_submission_rate": nosub / n, "crash_rate": crash / n}

    def guards(self, incumbent, candidate) -> list[str]:
        d = (candidate.extra.get("crash_rate") or 0.0) - (incumbent.extra.get("crash_rate") or 0.0)
        lim = CFG.get("max_crash_rate_rise", 0.05)
        return [f"harness crash rate rose {d:+.3f} (limit {lim})"] if d > lim else []

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        return render.load_trial(Path(runs_dir) / "jobs" / job, BY_ID[task_id], trial)

    def render_trace(self, rec, detail=False):
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        meta, v = rec.get("meta") or {}, rec.get("verdict") or {}
        msgs = (rec.get("traj") or {}).get("messages") or []
        runs = sum(1 for m in msgs if m.get("role") == "assistant"
                   and '"run_python"' in str(m.get("content")))
        return (f"{task_id} | {render.failure_class(rec)} | pass_rate={tr.mean:.2f} | "
                f"steps={meta.get('steps')} run_python={runs} tokens={meta.get('tokens')} | "
                f"status={v.get('status')}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        toy = root / "domains" / "toy"
        comp = subprocess.run([sys.executable, "-m", "compileall", "-q",
                               str(self.harness_dir(root))], capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "err": (comp.stdout + comp.stderr)[-1500:]}
        code = ("import inspect\n"
                "from harness.agent import run_agent\n"
                "from harness.prompts import TASK_TEMPLATE\n"
                "assert list(inspect.signature(run_agent).parameters) == "
                "['prompt', 'entry', 'chat', 'run_python', 'max_steps'], 'signature changed'\n"
                "body = TASK_TEMPLATE.format(prompt='SENTINEL-P', entry='SENTINEL-E')\n"
                "assert 'SENTINEL-P' in body and 'SENTINEL-E' in body, 'template drops fields'\n"
                "print('CTOR OK')\n")
        ctor = subprocess.run([sys.executable, "-c", code], cwd=str(toy),
                              capture_output=True, text=True)
        if ctor.returncode != 0 or "CTOR OK" not in ctor.stdout:
            return False, {"stage": "ctor", "err": (ctor.stderr or ctor.stdout)[-1500:]}
        shutil.rmtree(runs_dir / "jobs" / job, ignore_errors=True)
        self.run(root, runs_dir, job, ids, 1, job)
        per, extra = self.score(runs_dir, job, ids, 1)
        missing = sum(tr.missing for tr in per.values())
        ok = missing == 0 and extra["crash_rate"] == 0.0
        return ok, {"stage": "run", "smoke_ids": ids, "missing": missing, **extra}


DOMAIN = ToyDomain()
