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
"""Harness miner: struggle episodes -> recurring struggles -> harness tasks.

    python -m rrsi harness mine [--out ~/rrsi-private/harness] [--since DATE]
                                [--backend sdk|copilot|codex] [--model M]

Stage 1 (deterministic, Rust): `rrsi-mine traces` writes OUT/episodes.jsonl.
Stage 2 (LLM):
  label    batches of redacted episode digests -> one recurring-struggle
           pattern slug per episode (OUT/labels.jsonl; labelled episodes are
           never sent again, so re-runs only pay for new episodes)
  cluster  pattern slugs -> canonical clusters (OUT/clusters.json)
  tasks    one harness task per top cluster (OUT/tasks/<id>.json,
           OUT/tasks/index.json, OUT/exam_candidates.jsonl)
  report   OUT/REPORT.md: the top recurring struggles with counts

Every count (episodes, sessions, projects, signals) is computed here from the
episode records and the labels; the model only names, groups and proposes.
OUT holds quotes of private transcripts and must be outside every git work
tree (enforced).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from rrsi.harness.llm import LLMAuthError, LLMError, complete_json

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tools" / "rrsi-mine"
DEFAULT_OUT = Path.home() / "rrsi-private" / "harness"
FIX_KINDS = ["claude_md_rule", "skill", "house_lint_gate", "memory", "tool_cli_fix", "doc"]
PRIORITIES = ["P0", "P1", "P2", "P3"]
NOISE = "noise"

# (backend-agnostic) system, prompt, schema, cwd -> dict
Complete = Callable[[str, str, dict, Path], dict]


class PrivacyError(RuntimeError):
    pass


def enclosing_work_tree(path: Path) -> Path | None:
    p = path.expanduser().resolve()
    return next((a for a in [p, *p.parents] if (a / ".git").exists()), None)


def ensure_private(out: Path) -> Path:
    tree = enclosing_work_tree(out)
    if tree is not None:
        raise PrivacyError(f"refusing to write harness mining output to {out} inside the git "
                           f"work tree {tree}: it quotes private session transcripts")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------- stage 1

def rust_binary(build: bool = True) -> Path:
    exe = CRATE / "target" / "release" / "rrsi-mine"
    if build:
        subprocess.run(["cargo", "build", "--release", "--locked", "--quiet",
                        "--manifest-path", str(CRATE / "Cargo.toml")], check=True)
    return exe


def run_traces(out: Path, root: Path | None, since: str, jobs: int, exclude: list[str]) -> dict:
    cmd = [str(rust_binary()), "traces", "--out", str(out), "--jobs", str(jobs), "--since", since]
    if root:
        cmd += ["--root", str(root)]
    for x in exclude:
        cmd += ["--exclude", x]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(r.stdout)


def load_episodes(out: Path) -> list[dict]:
    path = out / "episodes.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# ------------------------------------------------------------ redaction

REDACTIONS = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat|sk|sk-ant|xox[abp])[-_][A-Za-z0-9_-]{10,}"), "<secret>"),
    (re.compile(r"(?i)\b(token|password|secret|api[_-]?key)(\s*[=:]\s*)\S+"), r"\1\2<secret>"),
    (re.compile(r"\b[0-9a-f]{32,}\b"), "<hex>"),
    (re.compile(r"/home/[^/\s\"']+"), "~"),
]


def redact(text: str) -> str:
    """Strip addresses, secrets and home paths before any text leaves the machine."""
    for pat, rep in REDACTIONS:
        text = pat.sub(rep, text)
    return text


def _trunc(t: str, n: int) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n] + "…"


def digest(ep: dict, n_ctx: int = 10, width: int = 220) -> str:
    """A compact, redacted text form of one episode for the model."""
    sigs = ",".join(f"{k}x{v}" for k, v in sorted(ep["signals"].items()))
    head = (f"[{ep['id']}] signals={sigs} subagent={str(ep.get('subagent', False)).lower()} "
            f"events={ep['counts']['span_events']}")
    ctx = ep.get("context", [])
    keep = {i for i, c in enumerate(ctx) if c.get("signals")}
    for i in range(len(ctx)):
        if len(keep) >= n_ctx:
            break
        keep.add(i)
    lines = [head, f"  user: {redact(_trunc(ep.get('user_turn', ''), 300))}"]
    for i in sorted(keep)[:max(n_ctx, 1) * 2]:
        c = ctx[i]
        tag = c["kind"] + (f"({c['tool']})" if c.get("tool") else "") + ("!" if c.get("is_error") else "")
        if c.get("signals"):
            tag += " *" + ",".join(c["signals"])
        lines.append(f"  {tag}: {redact(_trunc(c.get('text', ''), width))}")
    return "\n".join(lines)


# --------------------------------------------------------------- label

LABEL_SYSTEM = """You study where an AI coding agent (Claude Code, working in a user's repositories) struggled, so its harness (instructions, skills, memory, lint gates, tools) can be fixed.
Each episode lists the signals a deterministic detector fired (tool_error, retry, hook_timeout, permission_denial, user_interrupt, user_correction, reask, silence, test_failure), the user's request, and the events around it.
For EVERY episode, name the recurring struggle it is an instance of:
- pattern: a short lowercase snake_case slug naming the generalizable CAUSE (not the task), reusable across episodes, e.g. hook_timeout_blocks_writes, wrong_python_env, ignored_operator_style_rule, flaky_ci_wait, edited_wrong_worktree.
- Use "noise" when it is not a real struggle (an expected red test in TDD, a harmless probe such as checking whether a file exists, a question that merely contains "why").
- summary: one sentence on what went wrong, with no names of people, hosts, secrets or private code.
- harness_fixable: true if a change to the agent's instructions/skills/memory/gates/tools would plausibly prevent it."""

LABEL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["labels"],
    "properties": {"labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "pattern", "summary", "harness_fixable"],
        "properties": {"id": {"type": "string"}, "pattern": {"type": "string"},
                       "summary": {"type": "string"}, "harness_fixable": {"type": "boolean"}}}}},
}


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (t or "").lower()).strip("_")[:60] or NOISE


def load_labels(out: Path) -> dict[str, dict]:
    path = out / "labels.jsonl"
    labels = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec["id"]] = rec
    return labels


def label(out: Path, episodes: list[dict], complete: Complete, batch: int, jobs: int,
          log: Callable[[str], None] = print) -> tuple[dict[str, dict], int]:
    """Labels every unlabelled episode; returns (labels, failed batches)."""
    labels = load_labels(out)
    todo = sorted((e for e in episodes if e["id"] not in labels),
                  key=lambda e: (e["project"], e["file"], e["start_event"]))
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    log(f"[label] {len(labels)} cached, {len(todo)} to label in {len(batches)} batches")
    failed = 0

    def one(b: list[dict]) -> list[dict]:
        prompt = "Episodes:\n\n" + "\n\n".join(digest(e) for e in b)
        got = complete(LABEL_SYSTEM, prompt, LABEL_SCHEMA, out)
        want = {e["id"] for e in b}
        recs = [{"id": l["id"], "pattern": slug(l["pattern"]), "summary": _trunc(l["summary"], 300),
                 "harness_fixable": bool(l["harness_fixable"])}
                for l in got.get("labels", []) if l.get("id") in want]
        return recs

    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool, (out / "labels.jsonl").open("a") as f:
        futs = {pool.submit(one, b): k for k, b in enumerate(batches)}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                recs = fut.result()
            except (LLMError, KeyError, TypeError) as e:
                if isinstance(e, LLMAuthError):
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                failed += 1
                log(f"[label] batch {futs[fut]} failed: {str(e)[:200]}")
                continue
            for r in recs:
                if r["id"] not in labels:
                    labels[r["id"]] = r
                    f.write(json.dumps(r) + "\n")
            f.flush()
            log(f"[label] {n}/{len(batches)} batches")
    return labels, failed


# ------------------------------------------------------------- cluster

CLUSTER_SYSTEM = """You merge struggle-pattern slugs that name the same underlying cause of an AI coding agent's trouble into canonical clusters.
Group only true synonyms or the same root cause; keep distinct causes apart. Every input slug must appear in exactly one cluster.
key: lowercase snake_case slug for the cluster; title: a short human title (no names of people, hosts or private code)."""

CLUSTER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clusters"],
    "properties": {"clusters": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["key", "title", "patterns"],
        "properties": {"key": {"type": "string"}, "title": {"type": "string"},
                       "patterns": {"type": "array", "items": {"type": "string"}}}}}},
}


def cluster(out: Path, labels: dict[str, dict], complete: Complete,
            log: Callable[[str], None] = print) -> dict[str, dict]:
    """pattern slug -> {key, title}; cached while the slug set is unchanged."""
    by_pat: dict[str, list[str]] = defaultdict(list)
    for rec in labels.values():
        if rec["pattern"] != NOISE:
            by_pat[rec["pattern"]].append(rec["summary"])
    pats = sorted(by_pat)
    fp = hashlib.sha256(json.dumps(pats).encode()).hexdigest()
    cache = out / "clusters.json"
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get("fingerprint") == fp:
            return old["mapping"]
    lines = [f"{p} ({len(by_pat[p])} episodes): " + " | ".join(_trunc(s, 120) for s in by_pat[p][:2])
             for p in sorted(pats, key=lambda p: -len(by_pat[p]))]
    log(f"[cluster] {len(pats)} pattern slugs")
    mapping: dict[str, dict] = {}
    if pats:
        got = complete(CLUSTER_SYSTEM, "Pattern slugs:\n" + "\n".join(lines), CLUSTER_SCHEMA, out)
        for c in got.get("clusters", []):
            for p in c.get("patterns", []):
                p = slug(p)
                if p in by_pat and p not in mapping:
                    mapping[p] = {"key": slug(c["key"]), "title": _trunc(c["title"], 120)}
    for p in pats:  # anything the model dropped stays its own cluster
        mapping.setdefault(p, {"key": p, "title": p.replace("_", " ")})
    cache.write_text(json.dumps({"fingerprint": fp, "mapping": mapping}, indent=1))
    return mapping


def evidence(episodes: list[dict], labels: dict[str, dict], mapping: dict[str, dict]) -> list[dict]:
    """Deterministic counts per cluster, largest first."""
    groups: dict[str, dict] = {}
    for ep in episodes:
        rec = labels.get(ep["id"])
        if not rec or rec["pattern"] == NOISE or rec["pattern"] not in mapping:
            continue
        m = mapping[rec["pattern"]]
        g = groups.setdefault(m["key"], {"key": m["key"], "title": m["title"], "episodes": [],
                                         "sessions": set(), "projects": set(), "signals": Counter(),
                                         "patterns": Counter(), "fixable": 0, "first": ep["start"],
                                         "last": ep["start"]})
        g["episodes"].append(ep)
        g["sessions"].add(ep["session_id"])
        g["projects"].add(ep["project"])
        g["signals"].update(ep["signals"].keys())
        g["patterns"][rec["pattern"]] += 1
        g["fixable"] += rec["harness_fixable"]
        g["first"], g["last"] = min(g["first"], ep["start"]), max(g["last"], ep["start"])
    return sorted(groups.values(), key=lambda g: (-len(g["episodes"]), -len(g["sessions"]), g["key"]))


def counts(g: dict) -> dict:
    return {"episodes": len(g["episodes"]), "sessions": len(g["sessions"]), "projects": len(g["projects"]),
            "harness_fixable": g["fixable"], "signals": dict(g["signals"].most_common()),
            "patterns": dict(g["patterns"].most_common()), "first": g["first"], "last": g["last"]}


# --------------------------------------------------------------- tasks

TASK_SYSTEM = f"""You turn one recurring struggle of an AI coding agent (Claude Code) into ONE actionable harness task.
The harness is what surrounds the model: CLAUDE.md rules, skills, memory files, house-lint gate rules, tools/CLI verbs, docs.
Propose the smallest change that would stop this struggle from recurring, and a check that shows it worked.
proposed_fix.kind is one of {FIX_KINDS}. priority: P0 (blocks work, frequent) .. P3 (rare/cosmetic).
exam_candidate: checkable=true only when a before/after can be verified mechanically (e.g. a scripted session or a lint fixture that fails before the fix and passes after); describe before, after and the check.
Write generally: no names of people, hosts, accounts, secrets or private code."""

TASK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "struggle_pattern", "root_cause_hypothesis", "proposed_fix", "acceptance_check",
                 "priority", "exam_candidate"],
    "properties": {
        "title": {"type": "string"},
        "struggle_pattern": {"type": "string"},
        "root_cause_hypothesis": {"type": "string"},
        "proposed_fix": {"type": "object", "additionalProperties": False, "required": ["kind", "change"],
                         "properties": {"kind": {"type": "string", "enum": FIX_KINDS},
                                        "change": {"type": "string"}}},
        "acceptance_check": {"type": "string"},
        "priority": {"type": "string", "enum": PRIORITIES},
        "exam_candidate": {"type": "object", "additionalProperties": False,
                           "required": ["checkable", "before", "after", "check"],
                           "properties": {"checkable": {"type": "boolean"}, "before": {"type": "string"},
                                          "after": {"type": "string"}, "check": {"type": "string"}}},
    },
}


def sample(eps: list[dict], k: int) -> list[dict]:
    """Up to k episodes, spread across sessions first."""
    seen, first, rest = set(), [], []
    for e in eps:
        (rest if e["session_id"] in seen else first).append(e)
        seen.add(e["session_id"])
    return (first + rest)[:k]


def make_tasks(out: Path, groups: list[dict], labels: dict[str, dict], complete: Complete, top: int,
               min_episodes: int, jobs: int, log: Callable[[str], None] = print) -> tuple[list[dict], int]:
    tdir = out / "tasks"
    tdir.mkdir(exist_ok=True)
    chosen = [g for g in groups if len(g["episodes"]) >= min_episodes][:top]
    failed = 0

    def one(g: dict) -> dict:
        c = counts(g)
        ids = sorted(e["id"] for e in g["episodes"])
        fp = hashlib.sha256(json.dumps([g["title"], ids]).encode()).hexdigest()
        path = tdir / f"{g['key']}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("fingerprint") == fp:
                return old
        summaries = list(dict.fromkeys(labels[e["id"]]["summary"] for e in g["episodes"]))[:15]
        prompt = (f"Struggle cluster: {g['title']} (key {g['key']})\n"
                  f"Measured: {c['episodes']} episodes in {c['sessions']} sessions across {c['projects']} projects; "
                  f"signals {c['signals']}; member patterns {c['patterns']}.\n\n"
                  "Per-episode summaries:\n- " + "\n- ".join(redact(s) for s in summaries) +
                  "\n\nSample episodes:\n\n" + "\n\n".join(digest(e) for e in sample(g["episodes"], 6)))
        got = complete(TASK_SYSTEM, prompt, TASK_SCHEMA, out)
        task = {"id": g["key"], **{k: got[k] for k in TASK_SCHEMA["required"]},
                "evidence": {"episode_ids": ids, **c}, "fingerprint": fp}
        path.write_text(json.dumps(task, indent=1))
        return task

    tasks = []
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futs = {pool.submit(one, g): g["key"] for g in chosen}
        for fut in as_completed(futs):
            try:
                tasks.append(fut.result())
            except (LLMError, KeyError, TypeError) as e:
                if isinstance(e, LLMAuthError):
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                failed += 1
                log(f"[tasks] {futs[fut]} failed: {str(e)[:200]}")
    keys = {t["id"] for t in tasks}
    for stale in tdir.glob("*.json"):
        if stale.name != "index.json" and stale.stem not in keys:
            stale.unlink()
    tasks.sort(key=lambda t: (-t["evidence"]["episodes"], t["id"]))
    index = [{"id": t["id"], "title": t["title"], "priority": t["priority"], "fix": t["proposed_fix"]["kind"],
              "episodes": t["evidence"]["episodes"], "sessions": t["evidence"]["sessions"],
              "projects": t["evidence"]["projects"]} for t in tasks]
    (tdir / "index.json").write_text(json.dumps(index, indent=1))
    with (out / "exam_candidates.jsonl").open("w") as f:
        for t in tasks:
            if t["exam_candidate"]["checkable"]:
                f.write(json.dumps({"task": t["id"], "title": t["title"], **t["exam_candidate"],
                                    "evidence_episodes": t["evidence"]["episodes"]}) + "\n")
    log(f"[tasks] {len(tasks)} tasks ({failed} failed)")
    return tasks, failed


# -------------------------------------------------------------- report

def _cell(t: str) -> str:
    return " ".join(str(t).split()).replace("|", "\\|")


def report(out: Path, run: dict, tasks: list[dict], labels: dict[str, dict], top: int = 15) -> Path:
    s = run["traces"]
    lab = Counter("noise" if r["pattern"] == NOISE else "struggle" for r in labels.values())
    lines = [
        "# Harness struggles: report", "",
        f"Generated {run['finished']} by `python -m rrsi harness mine` (backend `{run['backend']}`, model "
        f"`{run['model']}`). Private: quotes and derives from session transcripts.", "",
        "## Run", "",
        "| Measure | Value |", "|---|---|",
        f"| Transcripts scanned (main + subagent) | {s['transcripts']} ({s['processed']} parsed, "
        f"{s['skipped_unchanged']} unchanged) |",
        f"| Sessions / projects | {s['sessions']} / {s['projects']} |",
        f"| Events | {s['events']} |",
        f"| Struggle episodes | {s['episodes']} in {s['sessions_with_episodes']} sessions |",
        f"| Episodes labelled | {len(labels)} ({lab['struggle']} struggles, {lab['noise']} noise) |",
        f"| Clusters / tasks emitted | {run['clusters']} / {len(tasks)} |",
        f"| Failed LLM calls | {run['failed_calls']} |",
        f"| Wall time | traces {run['seconds']['traces']:.1f}s, label {run['seconds']['label']:.0f}s, "
        f"cluster {run['seconds']['cluster']:.0f}s, tasks {run['seconds']['tasks']:.0f}s, "
        f"total {run['seconds']['total']:.0f}s |", "",
        "### Episodes per signal", "",
        "An episode counts once per signal it contains (an episode can carry several).", "",
        "| Signal | Episodes | Hits |", "|---|---|---|",
    ]
    for sig, n in sorted(s["episodes_per_signal"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {sig} | {n} | {s['hits_per_signal'].get(sig, 0)} |")
    lines += ["", f"## Top {min(top, len(tasks))} recurring struggles", "",
              "Counts are measured from the episode records: an episode belongs to the cluster of its label.", "",
              "| # | Struggle | Episodes | Sessions | Projects | Main signals | Priority | Fix (kind) | Proposed fix |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, t in enumerate(tasks[:top], 1):
        ev = t["evidence"]
        sigs = ", ".join(f"{a} {b}" for a, b in list(ev["signals"].items())[:3])
        lines.append(f"| {k} | {_cell(t['title'])} | {ev['episodes']} | {ev['sessions']} | {ev['projects']} | "
                     f"{_cell(sigs)} | {t['priority']} | {t['proposed_fix']['kind']} | "
                     f"{_cell(t['proposed_fix']['change'])} |")
    lines += ["", "## Details", ""]
    for k, t in enumerate(tasks[:top], 1):
        ev = t["evidence"]
        lines += [f"### {k}. {t['title']} (`{t['id']}`)", "",
                  f"- **Pattern:** {t['struggle_pattern']}",
                  f"- **Evidence:** {ev['episodes']} episodes, {ev['sessions']} sessions, {ev['projects']} projects, "
                  f"{ev['first'][:10]} to {ev['last'][:10]}; signals {ev['signals']}",
                  f"- **Root cause (hypothesis):** {t['root_cause_hypothesis']}",
                  f"- **Fix ({t['proposed_fix']['kind']}, {t['priority']}):** {t['proposed_fix']['change']}",
                  f"- **Acceptance:** {t['acceptance_check']}",
                  f"- **Exam candidate:** {'yes' if t['exam_candidate']['checkable'] else 'no'}"
                  + (f" — {t['exam_candidate']['check']}" if t['exam_candidate']['checkable'] else ""), ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- main

def mine(out: Path = DEFAULT_OUT, root: Path | None = None, since: str = "", backend: str = "sdk",
         model: str | None = None, jobs: int = 4, batch: int = 25, top: int = 20, min_episodes: int = 2,
         exclude: list[str] | None = None, skip_traces: bool = False, effort: str = "medium",
         complete: Complete | None = None, log: Callable[[str], None] = print) -> dict:
    from rrsi.harness.llm import DEFAULT_MODEL
    out = ensure_private(out.expanduser())
    model = model or DEFAULT_MODEL.get(backend, "")
    if complete is None:
        def complete(system, prompt, schema, o):
            return complete_json(backend, model, system, prompt, schema, ensure_private(o / "llm-cwd"), effort)
    t0 = time.time()
    secs = {}
    if not skip_traces:
        summary = run_traces(out, root, since, 8, list(exclude or []) + ["rrsi-private"])
    else:
        summary = json.loads((out / "traces-summary.json").read_text())
    secs["traces"] = summary.get("seconds", 0.0)
    log(f"[traces] {summary['episodes']} episodes from {summary['transcripts']} transcripts")
    episodes = load_episodes(out)
    t = time.time()
    labels, failed_label = label(out, episodes, complete, batch, jobs, log)
    secs["label"] = time.time() - t
    live = {e["id"] for e in episodes}
    labels = {k: v for k, v in labels.items() if k in live}
    t = time.time()
    mapping = cluster(out, labels, complete, log)
    secs["cluster"] = time.time() - t
    groups = evidence(episodes, labels, mapping)
    t = time.time()
    tasks, failed_tasks = make_tasks(out, groups, labels, complete, top, min_episodes, jobs, log)
    secs["tasks"] = time.time() - t
    secs["total"] = time.time() - t0 + secs["traces"]
    run = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "backend": backend, "model": model,
           "traces": summary, "clusters": len(groups), "tasks": len(tasks),
           "failed_calls": failed_label + failed_tasks, "seconds": secs,
           "top": [{"title": t["title"], **{k: t["evidence"][k] for k in ("episodes", "sessions", "projects")}}
                   for t in tasks[:15]]}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    path = report(out, run, tasks, labels)
    log(f"[report] {path}")
    return run
