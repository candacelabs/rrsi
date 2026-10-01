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
"""Preview-vs-grade agreement under held-out physics, at zero policy tokens.

    python3 agreement.py --runs RUNS/sim --jobs base,r0A --salt-file F \
        [--arms H0=base+base2,H1=r0A --baseline H0] [--grid] [--scales 0,0.5,1]
        [--before-backend cpu] [--live] [--json OUT] [--workers N]

The agent's rollout preview is HighwayEnv's kinematic bicycle
(bench/fake_backend.py); graded episodes on the `phys` simulator backend run
the grader-side vehicle with per-episode physics (heldout_vehicle.py,
heldout_physics.py). This tool measures how much a preview pass says about
the grade, without asking the policy anything:

  stored    Every stored trial of the named jobs (controller.json, and the
            verdict the job was graded with) is replayed on the preview and
            on the graded vehicle of its own episode id. A harness that never
            sees simulator output submits the same controller whatever the
            grader, so the replay is an exact sample of it under the new
            grader; one that reads the preview is exact too, because the
            preview did not change. Reports n, the stored S, the preview S,
            the graded S, exact reward agreement, the confusion matrix,
            P(grade pass | preview pass), P(grade fail | preview fail),
            Pearson correlation and mean |preview - grade|, the median
            maximum lateral gap between the two trajectories, and how often
            the preview reproduces the stored verdict.
  --arms    Pools jobs into arms and, with --baseline, reports paired
            per-scenario differences of the graded replay against the
            baseline arm (bootstrap over scenarios, 10,000 resamples, seed 0,
            percentile 95% interval).
  --grid    168 linear controllers (KY x KH x KV gains) on the 96 campaign
            episode ids (32 scenarios x t0..t2): agreement, how many ids some
            controller solves, the CSF reference, the prompt template and
            two naive controllers over 32 scenarios x t0..t29, and the dose
            response over --scales (scale 0 = nominal physics: only the
            structural difference).
  --before-backend NAME
            Runs the grid's controllers once per scenario on that simulator
            backend (for example `cpu`, HighwayEnv through CSF's scenario
            worker) and reports the same agreement against the preview: the
            numbers before held-out physics.
  --live    For trials that called the rollout tool, compares the last
            previewed controller with the submitted one, grades the preview
            episode in place and reports in-situ preview-vs-grade agreement;
            splits trials into preview-repaired (the first preview failed)
            and first-try passes.

CSF is Candace Labs' Go framework for AI-agent systems; the controllers are
CSF controllers, evaluated here by the in-process mirror of its runtime.
Grader-side tool: harness code is forbidden to import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics as st
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
for _path in (SIM / "bench", SIM / "data", HERE):
    sys.path.insert(0, str(_path))

import fake_backend                                            # noqa: E402
import oracles                                                 # noqa: E402
from heldout_physics import DISTRIBUTION, draw, load_salt, salt_id  # noqa: E402
from heldout_vehicle import HeldoutVehicle                     # noqa: E402
from scenarios import BY_ID, EVOLVE, HELDOUT_IDS              # noqa: E402

KY = (300, 0, -50, -150, -300, -600, -1200, -2500)
KH = (300, 0, -100, -300, -600, -1200, -2500)
KV = (100, 500, 1500)
CAMPAIGN_TRIALS = 3
FAIRNESS_TRIALS = 30
BATCH_LIMIT = 256   # episodes per batch CSF's scenario worker accepts
_SALT = b""


def linear(ky: int, kh: int, kv: int, name: str = "") -> dict:
    """steering = CLAMP(ky x in0 + kh x in1), acceleration = CLAMP(kv x in2)."""
    def scale(index, gain):
        return {"opcode": "OPCODE_SCALE", "value": gain,
                "arguments": [{"opcode": "OPCODE_INPUT", "input_index": index}]}

    def clamp(expression):
        return {"opcode": "OPCODE_CLAMP", "lower": -1000, "upper": 1000, "arguments": [expression]}

    return {"schema_version": 1, "name": name or f"linear_{ky}_{kh}_{kv}",
            "steering": clamp({"opcode": "OPCODE_ADD", "arguments": [scale(0, ky), scale(1, kh)]}),
            "acceleration": clamp(scale(2, kv))}


REFERENCE = linear(-250, -500, 500, "csf-reference")
TEMPLATE = linear(-300, -600, 800, "prompt-template")
# Fixed controllers reported over 32 scenarios x t0..t29: CSF's training
# baseline, the gain template H_1's prompt hard-codes, and two naive ones.
FIXED = {"reference": REFERENCE, "template": TEMPLATE,
         "aggressive": linear(-1500, -800, 800, "aggressive"),
         "lateral_only": linear(-400, 0, 500, "lateral-only, no heading damping")}


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _init(salt: bytes) -> None:
    global _SALT
    _SALT = salt


def _episode(task: dict, controller, root: Path, name: str, plant_factory=None) -> tuple[float, list]:
    job = {"id": name, "scenario": task["scenario"], "goal_metres": task["goal_metres"], "controller": controller}
    fake_backend.run_episode(job, root / name, "agreement", plant_factory=plant_factory)
    verdict = oracles.grade(root / name, task)
    trace = (root / name / "trace.jsonl").read_text().splitlines()
    return float(verdict.get("reward", 0.0)), [json.loads(row)["state"]["lateral_metres"] for row in trace]


def pair(args) -> tuple[float, float, float | None]:
    """(task id, episode id, controller, scale) -> (preview reward, graded
    reward, max |lateral gap|). A missing controller scores 0 on both."""
    task_id, episode_id, controller, scale = args
    if controller is None:
        return 0.0, 0.0, None
    task = BY_ID[task_id]
    physics, noise_seed = draw(_SALT, episode_id, scale)
    root = Path(tempfile.mkdtemp(prefix="rrsi-agreement-"))
    try:
        preview, lat_p = _episode(task, controller, root, "p")
        graded, lat_g = _episode(task, controller, root, "g", plant_factory=lambda sc: HeldoutVehicle(
            sc, physics, random.Random(noise_seed)))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    gap = max((abs(a - b) for a, b in zip(lat_p, lat_g)), default=None)
    return preview, graded, gap


def preview_only(args) -> float:
    """(task id, controller) -> the kinematic preview's reward."""
    task_id, controller = args
    root = Path(tempfile.mkdtemp(prefix="rrsi-agreement-"))
    try:
        return _episode(BY_ID[task_id], controller, root, "p")[0]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = st.fmean(xs), st.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5


def agreement(preview: list[float], graded: list[float], gaps: list | None = None) -> dict:
    """Preview reward vs graded reward, episode by episode."""
    n = len(preview)
    cm = {"preview_pass_grade_pass": 0, "preview_pass_grade_fail": 0,
          "preview_fail_grade_pass": 0, "preview_fail_grade_fail": 0}
    for p, g in zip(preview, graded):
        cm[f"preview_{'pass' if p == 1.0 else 'fail'}_grade_{'pass' if g == 1.0 else 'fail'}"] += 1
    passed = cm["preview_pass_grade_pass"] + cm["preview_pass_grade_fail"]
    failed = cm["preview_fail_grade_pass"] + cm["preview_fail_grade_fail"]
    real_gaps = [g for g in (gaps or []) if g is not None]
    return {"n": n, "preview_S": st.fmean(preview) if n else None, "graded_S": st.fmean(graded) if n else None,
            "exact_agreement": sum(abs(p - g) < 1e-12 for p, g in zip(preview, graded)) / n if n else None,
            "pearson": _pearson(preview, graded),
            "mean_abs_gap": st.fmean(abs(p - g) for p, g in zip(preview, graded)) if n else None,
            "p_grade_pass_given_preview_pass": cm["preview_pass_grade_pass"] / passed if passed else None,
            "p_grade_fail_given_preview_fail": cm["preview_fail_grade_fail"] / failed if failed else None,
            "confusion": cm,
            "median_max_lateral_gap_metres": st.median(real_gaps) if real_gaps else None}


def stored_trials(runs: Path, job: str) -> list[dict]:
    out = []
    for tdir in sorted((runs / "jobs" / job).glob("straight-*/t*")):
        task_id, trial = tdir.parent.name, tdir.name
        if task_id not in BY_ID or not trial[1:].isdigit():
            continue
        try:
            verdict = json.loads((tdir / "verdict.json").read_text())
        except (OSError, ValueError):
            verdict = None
        try:
            controller = json.loads((tdir / "controller.json").read_text())
        except (OSError, ValueError):
            controller = None
        out.append({"job": job, "task": task_id, "trial": int(trial[1:]), "controller": controller,
                    "stored": float((verdict or {}).get("reward") or 0.0), "dir": tdir})
    return out


def replay_jobs(pool, runs: Path, jobs: list[str]) -> tuple[dict, list[dict]]:
    report, rows = {}, []
    for job in jobs:
        trials = stored_trials(runs, job)
        results = pool.map(pair, [(t["task"], f"{t['task']}--t{t['trial']}", t["controller"], 1.0)
                                  for t in trials])
        for t, (p, g, gap) in zip(trials, results):
            rows.append({"job": job, "task": t["task"], "trial": t["trial"], "stored": t["stored"],
                         "preview": p, "graded": g, "max_lateral_gap_metres": gap,
                         "submitted": t["controller"] is not None})
        job_rows = [r for r in rows if r["job"] == job]
        report[job] = {**agreement([r["preview"] for r in job_rows], [r["graded"] for r in job_rows],
                                   [r["max_lateral_gap_metres"] for r in job_rows]),
                       "stored_S": st.fmean(r["stored"] for r in job_rows) if job_rows else None,
                       "preview_equals_stored": (sum(abs(r["preview"] - r["stored"]) < 1e-12 for r in job_rows)
                                                 / len(job_rows)) if job_rows else None}
    return report, rows


def bootstrap_mean(diffs: list[float], reps: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.fmean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return st.fmean(diffs), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def per_scenario(rows: list[dict], key: str) -> dict:
    by: dict = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r[key])
    return {t: st.fmean(v) for t, v in by.items()}


def arms_report(rows: list[dict], arms: dict, baseline: str | None) -> dict:
    out = {}
    for name, jobs in arms.items():
        arm = [r for r in rows if r["job"] in jobs]
        out[name] = {"jobs": jobs, "n": len(arm), "stored_S": st.fmean(r["stored"] for r in arm),
                     "graded_S": st.fmean(r["graded"] for r in arm)}
    if baseline:
        base_rows = [r for r in rows if r["job"] in arms[baseline]]
        for key in ("graded", "stored"):
            base = per_scenario(base_rows, key)
            for name, jobs in arms.items():
                if name == baseline:
                    continue
                arm = per_scenario([r for r in rows if r["job"] in jobs], key)
                common = sorted(set(arm) & set(base))
                if common:
                    mean, low, high = bootstrap_mean([arm[t] - base[t] for t in common])
                    out[name][f"paired_{key}_vs_{baseline}"] = {"scenarios": len(common), "mean": mean,
                                                                "ci95": [low, high]}
    return out


def grid_report(pool, scales: list[float]) -> dict:
    grid = [(ky, kh, kv) for ky in KY for kh in KH for kv in KV]
    ids = EVOLVE + HELDOUT_IDS
    units = [(g, t, k) for g in grid for t in ids for k in range(CAMPAIGN_TRIALS)]
    results = pool.map(pair, [(t, f"{t}--t{k}", linear(*g), 1.0) for g, t, k in units])
    preview = [r[0] for r in results]
    graded = [r[1] for r in results]
    best: dict = {}
    per_controller: dict = {}
    for (g, t, k), (p, gr, _) in zip(units, results):
        best[(t, k)] = max(best.get((t, k), 0.0), gr)
        per_controller.setdefault(g, []).append((p, gr))
    means = {g: (st.fmean(p for p, _ in v), st.fmean(gr for _, gr in v)) for g, v in per_controller.items()}
    perfect = [m[1] for m in means.values() if m[0] == 1.0]
    report = {"controllers": len(grid), "episode_ids": len(ids) * CAMPAIGN_TRIALS,
              "episodes": agreement(preview, graded),
              "controller_level": agreement([m[0] for m in means.values()], [m[1] for m in means.values()]),
              "ids_solved_by_some_controller": sum(v == 1.0 for v in best.values()),
              "preview_perfect_controllers": len(perfect),
              "preview_perfect_graded_range": [min(perfect), max(perfect)] if perfect else None,
              "per_controller": {f"{g[0]},{g[1]},{g[2]}": {"preview": m[0], "graded": m[1]} for g, m in means.items()}}
    for name, controller in FIXED.items():
        res = pool.map(pair, [(t, f"{t}--t{k}", controller, 1.0) for t in ids for k in range(FAIRNESS_TRIALS)])
        report[name] = {"episodes": len(res), "preview_S": st.fmean(r[0] for r in res),
                        "graded_S": st.fmean(r[1] for r in res),
                        "graded_full_pass": sum(r[1] == 1.0 for r in res) / len(res),
                        "practice_graded_S": st.fmean(r[1] for r, (t, _) in zip(
                            res, [(t, k) for t in ids for k in range(FAIRNESS_TRIALS)]) if t in EVOLVE),
                        "heldout_graded_S": st.fmean(r[1] for r, (t, _) in zip(
                            res, [(t, k) for t in ids for k in range(FAIRNESS_TRIALS)]) if t in HELDOUT_IDS)}
    report["dose_response"] = {}
    for s in scales:
        res = pool.map(pair, [(t, f"{t}--t{k}", linear(*g), s) for g, t, k in units])
        report["dose_response"][str(s)] = agreement([r[0] for r in res], [r[1] for r in res])
    return report


def before_report(pool, name: str, workers: int) -> dict:
    """The grid on another simulator backend (deterministic physics: one
    episode per scenario), against the kinematic preview."""
    import backends
    backend = backends.get(name)
    grid = [(ky, kh, kv) for ky in KY for kh in KH for kv in KV]
    ids = EVOLVE + HELDOUT_IDS
    units = [(g, t) for g in grid for t in ids]
    jobs = [{"id": f"g{n}", "scenario": BY_ID[t]["scenario"],
             "goal_metres": BY_ID[t]["goal_metres"], "controller": linear(*g)} for n, (g, t) in enumerate(units)]
    root = Path(tempfile.mkdtemp(prefix="rrsi-before-"))
    chunks = [jobs[i:i + BATCH_LIMIT] for i in range(0, len(jobs), BATCH_LIMIT)]
    lock, failures = threading.Lock(), []

    def run(index):
        try:
            backend.execute(chunks[index], root / f"chunk{index}", f"before-{index}")
        except Exception as error:  # noqa: BLE001 - reported, never scored
            with lock:
                failures.append(str(error)[-1000:])

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, range(len(chunks))))
        previews = pool.map(preview_only, [(t, linear(*g)) for g, t in units])
        preview, graded = [], []
        for n, (g, t) in enumerate(units):
            verdict = oracles.grade(root / f"chunk{n // BATCH_LIMIT}" / f"g{n}", BY_ID[t])
            if verdict.get("status") == "infra":
                continue
            preview.append(previews[n])
            graded.append(float(verdict.get("reward", 0.0)))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {"backend": name, "controllers": len(grid), "scenarios": len(ids), "infra_failures": failures,
            "episodes": agreement(preview, graded)}


def live_report(runs: Path, jobs: list[str]) -> dict:
    out = {}
    for job in jobs:
        rows = []
        for t in stored_trials(runs, job):
            rollouts = sorted((t["dir"] / "rollouts").glob("rollout-*"), key=lambda p: int(p.name.split("-")[1]))
            graded_rollouts = []
            for r in rollouts:
                manifest_path = r / "rollout" / "manifest.json"
                if not manifest_path.is_file():
                    continue
                verdict = oracles.grade(r / "rollout", BY_ID[t["task"]])
                if verdict.get("status") == "infra":
                    continue
                controller = json.loads(manifest_path.read_text()).get("controller")
                graded_rollouts.append((float(verdict.get("reward", 0.0)), canonical_sha256(controller)))
            if not graded_rollouts:
                continue
            submitted = canonical_sha256(t["controller"]) if t["controller"] is not None else None
            rows.append({"task": t["task"], "trial": t["trial"], "graded": t["stored"],
                         "first_preview": graded_rollouts[0][0], "last_preview": graded_rollouts[-1][0],
                         "last_previewed_is_submitted": graded_rollouts[-1][1] == submitted,
                         "rollouts": len(graded_rollouts)})
        same = [r for r in rows if r["last_previewed_is_submitted"]]
        repaired = [r for r in rows if r["first_preview"] < 1.0]
        first_try = [r for r in rows if r["first_preview"] == 1.0]
        out[job] = {"trials_with_preview": len(rows),
                    "last_previewed_is_submitted": len(same),
                    "in_situ": agreement([r["last_preview"] for r in same], [r["graded"] for r in same]),
                    "first_preview_failed": {"n": len(repaired), "graded_S": st.fmean(r["graded"] for r in repaired)
                                             if repaired else None},
                    "first_preview_passed": {"n": len(first_try), "graded_S": st.fmean(r["graded"] for r in first_try)
                                             if first_try else None}}
    return out


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _line(label: str, a: dict) -> str:
    return (f"{label}: n={a['n']} preview S={_fmt(a['preview_S'])} graded S={_fmt(a['graded_S'])} "
            f"agree={_fmt(a['exact_agreement'])} r={_fmt(a['pearson'])} |gap|={_fmt(a['mean_abs_gap'])} "
            f"P(grade pass|preview pass)={_fmt(a['p_grade_pass_given_preview_pass'])} "
            f"P(grade fail|preview fail)={_fmt(a['p_grade_fail_given_preview_fail'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, help="an RRSI runs directory for this domain (holds jobs/)")
    ap.add_argument("--jobs", default="", help="comma-separated job names whose stored trials to replay")
    ap.add_argument("--salt-file", required=True)
    ap.add_argument("--arms", default="", help="NAME=job+job,NAME=job: pool jobs into arms")
    ap.add_argument("--baseline", default="", help="arm to pair the others against")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--scales", default="0,0.25,0.5,0.75,1,1.25")
    ap.add_argument("--before-backend", default="")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    salt = load_salt(a.salt_file)
    jobs = [j for j in a.jobs.split(",") if j]
    report = {"distribution": DISTRIBUTION, "salt_id": salt_id(salt), "preview": "fake (HighwayEnv kinematic)"}
    with Pool(a.workers, initializer=_init, initargs=(salt,)) as pool:
        if jobs:
            if a.runs is None:
                ap.error("--jobs needs --runs")
            report["stored"], rows = replay_jobs(pool, a.runs, jobs)
            report["rows"] = rows
            for job, r in report["stored"].items():
                print(_line(job, r) + f" stored S={_fmt(r['stored_S'])} preview==stored "
                      f"{_fmt(r['preview_equals_stored'])} median max|dlat|={_fmt(r['median_max_lateral_gap_metres'])}")
            if a.arms:
                arms = {name: jobs_.split("+") for name, jobs_ in (x.split("=") for x in a.arms.split(","))}
                report["arms"] = arms_report(rows, arms, a.baseline or None)
                for name, r in report["arms"].items():
                    extra = " ".join(f"{k}={_fmt(v['mean'])} [{_fmt(v['ci95'][0])}, {_fmt(v['ci95'][1])}]"
                                     for k, v in r.items() if k.startswith("paired_"))
                    print(f"arm {name}: n={r['n']} stored S={_fmt(r['stored_S'])} graded S={_fmt(r['graded_S'])} "
                          f"{extra}")
        if a.grid:
            g = report["grid"] = grid_report(pool, [float(x) for x in a.scales.split(",") if x])
            print(_line(f"grid {g['controllers']} controllers x {g['episode_ids']} ids", g["episodes"]))
            print(_line("grid controller means", g["controller_level"]))
            print(f"ids solved by some grid controller: {g['ids_solved_by_some_controller']}/{g['episode_ids']}; "
                  f"preview-perfect controllers {g['preview_perfect_controllers']}, graded "
                  f"{_fmt(g['preview_perfect_graded_range'] and g['preview_perfect_graded_range'][0])}"
                  f"-{_fmt(g['preview_perfect_graded_range'] and g['preview_perfect_graded_range'][1])}")
            for name in FIXED:
                r = g[name]
                print(f"{name}: preview S={_fmt(r['preview_S'])} graded S={_fmt(r['graded_S'])} "
                      f"(practice {_fmt(r['practice_graded_S'])}, held out {_fmt(r['heldout_graded_S'])}) "
                      f"full pass={_fmt(r['graded_full_pass'])} over {r['episodes']} episodes")
            for s, r in g["dose_response"].items():
                print(_line(f"scale {s}", r))
        if a.before_backend:
            b = report["before"] = before_report(pool, a.before_backend, a.workers)
            print(_line(f"before: preview vs {b['backend']}", b["episodes"]) + f" infra={len(b['infra_failures'])}")
    if a.live:
        if a.runs is None or not jobs:
            ap.error("--live needs --runs and --jobs")
        report["live"] = live_report(a.runs, jobs)
        for job, r in report["live"].items():
            print(_line(f"live {job} (last preview vs grade, same controller)", r["in_situ"]) +
                  f" | first preview failed: n={r['first_preview_failed']['n']} graded "
                  f"S={_fmt(r['first_preview_failed']['graded_S'])}; first preview passed: "
                  f"n={r['first_preview_passed']['n']} graded S={_fmt(r['first_preview_passed']['graded_S'])}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
