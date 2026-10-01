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

Stage 1 (deterministic, Rust): a registered miner (`--miner traces`, the
default, `--miner handoffs` or `--miner pr-gap`) writes OUT/episodes.jsonl;
handoff and pr-gap tasks also carry a trigger rule ("when <condition>, message
<owner> with <payload>"). pr-gap also scores its own rule family on the
measured runs (OUT/pr-gap-summary.json, OUT/pr-gap-task.json).
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
RETRIES = 1

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


def run_miner(miner: str, out: Path, root: Path | None, since: str, jobs: int, exclude: list[str],
              extra: tuple[str, ...] = ()) -> dict:
    """Runs a registered `rrsi-mine` miner; its JSON summary comes back on stdout."""
    cmd = [str(rust_binary()), miner, "--out", str(out), "--jobs", str(jobs), "--since", since, *extra]
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
    extra = ep.get("extra") or {}
    for o in extra.get("outcomes", []):
        lines.append(f"  outcome of message at event {o['event']}: from={'coordinator' if o['coordinator'] else 'peer'} "
                     f"acted={str(o['acted']).lower()} replied={str(o['replied']).lower()} "
                     f"retraction={str(o['retraction']).lower()}")
    if extra.get("outcome"):
        lines.append(f"  run: outcome={extra['outcome']} commits={extra.get('commits')} "
                     f"before_first_push={extra.get('commits_before_first_push')} "
                     f"unpushed_at_end={extra.get('unpushed_at_end')} commit_to_push_s={extra.get('commit_to_push_secs')} "
                     f"push_to_pr_s={extra.get('push_to_pr_secs')} brief={extra.get('brief')}")
    if extra.get("phrases"):
        lines.append(f"  phrases: {redact(', '.join(map(str, extra['phrases'])))[:200]}")
    if extra.get("other_session"):
        lines.append(f"  other session: {str(extra['other_session'])[:8]} (project {redact(str(extra.get('other_project', '')))})")
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
          log: Callable[[str], None] = print, system: str | None = None) -> tuple[dict[str, dict], int]:
    """Labels every unlabelled episode; returns (labels, failed batches)."""
    labels = load_labels(out)
    todo = sorted((e for e in episodes if e["id"] not in labels),
                  key=lambda e: (e["project"], e["file"], e["start_event"]))
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    log(f"[label] {len(labels)} cached, {len(todo)} to label in {len(batches)} batches")
    failed = 0

    def one(b: list[dict]) -> list[dict]:
        prompt = "Episodes:\n\n" + "\n\n".join(digest(e) for e in b)
        got = complete(system or LABEL_SYSTEM, prompt, LABEL_SCHEMA, out)
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


# ------------------------------------------------------------ pr-gap

PR_GAP_LABEL_SYSTEM = """You study why actively working AI coding agents (Claude Code sessions and subagents, working for one operator whose rule is "actively working agents always have a PR") end up without a pushed branch or a pull request.
Each episode is one of: a run (one agent transcript) that committed but never pushed (never_pushed), pushed with no PR (pushed_no_pr), pushed or opened its PR late (slow_push, slow_pr), or whose brief deferred the PR (brief_defers_pr); an orchestrator brief (Agent prompt) that defers or forbids the PR or the push (brief_defers_pr); or an operator turn correcting missing pushes/PRs (operator_pr_correction). The run line gives the measured outcome, commit counts, latencies and the brief class (defers / early / silent / none).
For EVERY episode, name the recurring pattern it is an instance of:
- pattern: a short lowercase snake_case slug for the generalizable cause, e.g. brief_defers_pr_to_end, brief_silent_on_pr, stacked_phase_branch_without_pr, pushes_into_orchestrator_branch, commits_left_unpushed_at_handoff, operator_demands_pr.
- Use "noise" when no PR was expected (a scratch or benchmark repository, a branch merged by another PR on purpose, an instruction not about this agent's branch).
- summary: one sentence on what happened and what would have produced a PR early. No names of people, hosts, secrets or private code.
- harness_fixable: true if a brief rule, an operating rule or a watcher hook would plausibly have produced the PR."""

PR_GAP_TASK_SYSTEM = f"""You turn one recurring cause of AI coding agents working without a pushed branch or a pull request into ONE actionable harness task.
The harness is what surrounds the model: CLAUDE.md agent operating rules, the brief snippet orchestrators paste into subagent prompts, skills, memory files, lint gates, tools/CLI verbs, and watcher hooks.
Above all, propose a TRIGGER RULE of the form "when <condition a hook can detect from git/GitHub state or the transcript>, <action> for <owner, and how it is resolved: the agent whose worktree branch it is, the orchestrator that wrote the brief, ...> with <payload>", e.g. "when an agent's first commit is N minutes old with no push, or a pushed branch has no PR, open a draft PR for it and tell the agent".
proposed_fix.kind is one of {FIX_KINDS + ["ownership_hook"]}. priority: P0 (blocks work, frequent) .. P3 (rare/cosmetic).
exam_candidate: checkable=true only when a before/after can be verified mechanically (e.g. replay a brief and check that a PR exists for the branch within the first push); describe before, after and the check.
Write generally: no names of people, hosts, accounts, secrets or private code."""


def pr_gap_section(s: dict) -> list[str]:
    """The deterministic pr-gap measures: outcomes, latencies, briefs and the scored rules."""
    def row(name: str, p: dict) -> str:
        o = p["outcomes"]
        return (f"| {name} | {p['active_runs']} | {o.get('pr_opened', 0)} | {o.get('pr_existing', 0)} | "
                f"{o.get('pushed_default', 0)} | {o.get('pushed_no_pr', 0)} | {o.get('never_pushed', 0)} | "
                f"{p['gap_runs']} | {p['runs_with_unpushed_commits_at_end']} |")
    lines = ["", "### Runs (transcripts with at least one commit)", "",
             "Gap = pushed_no_pr + never_pushed. pr_existing needs the GitHub join.", "",
             "| Runs | Active | PR opened | PR existing | Pushed to main | Pushed, no PR | Never pushed | Gap | Unpushed at end |",
             "|---|---|---|---|---|---|---|---|---|",
             row("all", s["all"]), row("main sessions", s["main_sessions"]), row("subagents", s["subagents"]), "",
             "| Latency (all active runs) | n | median | p90 | max |", "|---|---|---|---|---|"]
    for k, label in (("commit_to_push_min", "first commit -> first push (min)"),
                     ("push_to_pr_min", "first push -> PR created (min)"),
                     ("commits_before_first_push", "commits before the first push")):
        d = s["all"][k]
        lines.append(f"| {label} | {d['n']} | {d['median']:.1f} | {d['p90']:.1f} | {d['max']:.1f} |")
    lines += ["", "| Subagent brief | Active runs | Gap runs | Gap rate |", "|---|---|---|---|"]
    for k, (n, g) in sorted(s.get("subagent_gap_by_brief", {}).items()):
        lines.append(f"| {k} | {n} | {g} | {100 * g / n:.0f}% |" if n else f"| {k} | 0 | 0 | - |")
    gh = s.get("github", {})
    if gh.get("joined"):
        c = gh.get("pushed_no_pr", {})
        lines += ["", f"GitHub join: {gh['lookups']} lookups ({gh['failed']} failed); pushed_no_pr runs "
                      f"{c.get('confirmed_no_pr', 0)} confirmed with no PR, {c.get('pr_opened_after_run', 0)} "
                      f"got one after the run ended, {c.get('unresolved', 0)} unresolved; {gh['pr_existing']} runs "
                      "pushed into a branch that already had a PR."]
    lines += ["", "### Trigger rules scored on the runs", "",
              "Fires = the rule would have triggered while the run was active; caught = it fired on a gap run; "
              "nags = it fired on a run that got its PR later; score = caught - nags.", "",
              "| N (min) | Fires | Gaps caught | Nags | Gaps missed | Score |", "|---|---|---|---|---|---|"]
    for r in s.get("rules", []):
        lines.append(f"| {r['minutes']} | {r['fires']} | {r['gaps_caught']} | {r['nags']} | {r['gaps_missed']} | {r['score']} |")
    if s.get("top_rule"):
        lines += ["", f"**Top rule:** {s['top_rule']['rule']}."]
    return lines


# ------------------------------------------------------------ handoffs

HANDOFF_FIX_KINDS = FIX_KINDS + ["ownership_hook"]

HANDOFF_LABEL_SYSTEM = """You study how several concurrent AI coding agents (Claude Code sessions and their subagents, working for one operator) coordinate, so their harness can make them talk to each other at the right moments.
Each episode lists the signals a deterministic detector fired: user_relay (the operator carries text between sessions), peer_message / coordinator_message (a message from another session or a coordinator arrived; outcome lines say whether the receiver acted and replied), peer_retraction / peer_churn (a message withdrew, voided or renamed an earlier decision; churn = repeated), message_out (the agent sent a message), vcs_conflict, worktree_collision, file_overlap / branch_overlap (two sessions touched the same file or branch at once), duplicate_work (two sessions opened near-identical issues/PRs), already_done, blocked_on_other, polling (re-running a status check instead of asking), ownership_question, claim.
For EVERY episode, name the recurring coordination pattern it is an instance of:
- pattern: a short lowercase snake_case slug naming the generalizable situation, e.g. operator_relays_between_sessions, concurrent_edit_same_file, decision_rename_storm, poll_instead_of_ask, unclear_owner_of_issue, message_acted_on_and_acknowledged.
- Use "noise" when the signal fired but no coordination was involved (e.g. "another agent" meaning a hypothetical, a conflict the agent itself caused in its own branch).
- summary: one sentence on what happened and whether talking to another agent would have helped (or whether the message that was sent helped or hurt). No names of people, hosts, secrets or private code.
- harness_fixable: true if a harness rule or an ownership-state hook would plausibly have handled it."""

HANDOFF_TASK_SYSTEM = f"""You turn one recurring cross-session coordination pattern of AI coding agents (several concurrent Claude Code sessions for one operator) into ONE actionable harness task.
The harness is what surrounds the model: CLAUDE.md rules, skills, memory files, house-lint gate rules, tools/CLI verbs, docs, and ownership hooks (a planned store of agents, typed addresses and atomic claims on task keys).
Above all, propose a TRIGGER RULE of the form "when <condition an agent or hook can detect>, message <owner, and how the owner is resolved: claim on the issue/branch/file, the session that last edited the file, the coordinator, ...> with <payload>". Prefer conditions that are mechanically detectable. If the pattern shows messages that HURT (churn, voided decisions), the rule may restrict messaging instead (e.g. batch decisions, send only final rulings).
proposed_fix.kind is one of {HANDOFF_FIX_KINDS}. priority: P0 (blocks work, frequent) .. P3 (rare/cosmetic).
exam_candidate: checkable=true only when a before/after can be verified mechanically; describe before, after and the check.
Write generally: no names of people, hosts, accounts, secrets or private code."""

HANDOFF_TASK_SCHEMA = json.loads(json.dumps(TASK_SCHEMA))
HANDOFF_TASK_SCHEMA["properties"]["proposed_fix"]["properties"]["kind"]["enum"] = HANDOFF_FIX_KINDS
HANDOFF_TASK_SCHEMA["required"].append("trigger_rule")
HANDOFF_TASK_SCHEMA["properties"]["trigger_rule"] = {
    "type": "object", "additionalProperties": False, "required": ["when", "owner_resolution", "payload", "rule"],
    "properties": {"when": {"type": "string"}, "owner_resolution": {"type": "string"},
                   "payload": {"type": "string"}, "rule": {"type": "string"}}}


def sample(eps: list[dict], k: int) -> list[dict]:
    """Up to k episodes, spread across sessions first."""
    seen, first, rest = set(), [], []
    for e in eps:
        (rest if e["session_id"] in seen else first).append(e)
        seen.add(e["session_id"])
    return (first + rest)[:k]


def make_tasks(out: Path, groups: list[dict], labels: dict[str, dict], complete: Complete, top: int,
               min_episodes: int, jobs: int, log: Callable[[str], None] = print,
               system: str | None = None, schema: dict | None = None) -> tuple[list[dict], int]:
    system, schema = system or TASK_SYSTEM, schema or TASK_SCHEMA
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
        got = complete(system, prompt, schema, out)
        task = {"id": g["key"], **{k: got[k] for k in schema["required"]},
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


def report(out: Path, run: dict, tasks: list[dict], labels: dict[str, dict], top: int = 15,
           mode: "Mode | None" = None) -> Path:
    mode = mode or MODES["traces"]
    s = run["summary"]
    lab = Counter("noise" if r["pattern"] == NOISE else "struggle" for r in labels.values())
    lines = [
        f"# {mode.title}: report", "",
        f"Generated {run['finished']} by `python -m rrsi harness mine` (backend `{run['backend']}`, model "
        f"`{run['model']}`, miner `{mode.miner}`). Private: quotes and derives from session transcripts.", "",
        "## Run", "",
        "| Measure | Value |", "|---|---|",
        f"| Transcripts scanned (main + subagent) | {s['transcripts']} ({s['processed']} parsed, "
        f"{s['skipped_unchanged']} unchanged) |",
        f"| Sessions / projects | {s['sessions']} / {s['projects']} |",
        f"| Events | {s['events']} |",
        f"| Episodes | {s['episodes']} in {s['sessions_with_episodes']} sessions |",
        f"| Episodes labelled | {len(labels)} ({lab['struggle']} kept, {lab['noise']} noise) |",
        f"| Clusters / tasks emitted | {run['clusters']} / {len(tasks)} |",
        f"| Failed LLM calls | {run['failed_calls']} |",
        f"| Wall time | {mode.miner} {run['seconds']['miner']:.1f}s, label {run['seconds']['label']:.0f}s, "
        f"cluster {run['seconds']['cluster']:.0f}s, tasks {run['seconds']['tasks']:.0f}s, "
        f"total {run['seconds']['total']:.0f}s |", "",
        "### Episodes per signal", "",
        "An episode counts once per signal it contains (an episode can carry several).", "",
        "| Signal | Episodes | Hits |", "|---|---|---|",
    ]
    for sig, n in sorted(s["episodes_per_signal"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {sig} | {n} | {s['hits_per_signal'].get(sig, 0)} |")
    if "peer" in s:
        p = s["peer"]
        pct = (lambda n: f"{n} ({100 * n / p['messages']:.0f}%)") if p["messages"] else str
        lines += ["", "### Cross-session messages that arrived", "",
                  "Acted = a tool call before the next turn; replied = a message sent soon after (see the miner).", "",
                  "| Measure | Value |", "|---|---|",
                  f"| Messages received | {p['messages']} ({p['from_peers']} from peer sessions, "
                  f"{p['from_coordinator']} from coordinators; {p['senders']} distinct senders) |",
                  f"| Receiver acted | {pct(p['acted'])} |", f"| Receiver replied | {pct(p['replied'])} |",
                  f"| Retractions / renames / voids | {pct(p['retractions'])} |",
                  f"| Messages sent (SendMessage / send_message) | {p['message_out_calls']} "
                  f"({p['message_out_failed']} failed) |"]
    if "rules" in s and "all" in s:
        lines += pr_gap_section(s)
    rules = "trigger_rule" in mode.task_schema["properties"]
    lines += ["", f"## Top {min(top, len(tasks))} recurring {mode.unit}s", "",
              "Counts are measured from the episode records: an episode belongs to the cluster of its label.", "",
              f"| # | {mode.unit.capitalize()} | Episodes | Sessions | Projects | Main signals | Priority | Fix (kind) | "
              + ("Trigger rule |" if rules else "Proposed fix |"),
              "|---|---|---|---|---|---|---|---|---|"]
    for k, t in enumerate(tasks[:top], 1):
        ev = t["evidence"]
        sigs = ", ".join(f"{a} {b}" for a, b in list(ev["signals"].items())[:3])
        last = t["trigger_rule"]["rule"] if rules else t["proposed_fix"]["change"]
        lines.append(f"| {k} | {_cell(t['title'])} | {ev['episodes']} | {ev['sessions']} | {ev['projects']} | "
                     f"{_cell(sigs)} | {t['priority']} | {t['proposed_fix']['kind']} | {_cell(last)} |")
    lines += ["", "## Details", ""]
    for k, t in enumerate(tasks[:top], 1):
        ev = t["evidence"]
        lines += [f"### {k}. {t['title']} (`{t['id']}`)", "",
                  f"- **Pattern:** {t['struggle_pattern']}",
                  f"- **Evidence:** {ev['episodes']} episodes, {ev['sessions']} sessions, {ev['projects']} projects, "
                  f"{ev['first'][:10]} to {ev['last'][:10]}; signals {ev['signals']}",
                  f"- **Root cause (hypothesis):** {t['root_cause_hypothesis']}",
                  f"- **Fix ({t['proposed_fix']['kind']}, {t['priority']}):** {t['proposed_fix']['change']}",
                  *([f"- **Trigger rule:** {t['trigger_rule']['rule']} (when: {t['trigger_rule']['when']}; owner: "
                      f"{t['trigger_rule']['owner_resolution']}; payload: {t['trigger_rule']['payload']})"]
                    if rules else []),
                  f"- **Acceptance:** {t['acceptance_check']}",
                  f"- **Exam candidate:** {'yes' if t['exam_candidate']['checkable'] else 'no'}"
                  + (f" — {t['exam_candidate']['check']}" if t['exam_candidate']['checkable'] else ""), ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- modes

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class Mode:
    """One registered Rust miner plus the prompts that synthesize its episodes."""
    miner: str
    summary_file: str
    label_system: str
    task_system: str
    task_schema: dict
    title: str
    unit: str
    out: Path
    miner_args: tuple[str, ...] = ()


MODES = {
    "traces": Mode("traces", "traces-summary.json", LABEL_SYSTEM, TASK_SYSTEM, TASK_SCHEMA,
                   "Harness struggles", "struggle", DEFAULT_OUT),
    "handoffs": Mode("handoffs", "handoffs-summary.json", HANDOFF_LABEL_SYSTEM, HANDOFF_TASK_SYSTEM,
                     HANDOFF_TASK_SCHEMA, "Agent handoffs", "coordination pattern", DEFAULT_OUT / "handoffs"),
    "pr-gap": Mode("pr-gap", "pr-gap-summary.json", PR_GAP_LABEL_SYSTEM, PR_GAP_TASK_SYSTEM, HANDOFF_TASK_SCHEMA,
                   "Agents without a PR", "PR-gap pattern", DEFAULT_OUT / "pr-gap", ("--github",)),
}


# ---------------------------------------------------------------- main

def mine(out: Path | None = None, root: Path | None = None, since: str = "", backend: str = "sdk",
         model: str | None = None, jobs: int = 4, batch: int = 25, top: int = 20, min_episodes: int = 2,
         exclude: list[str] | None = None, skip_traces: bool = False, effort: str = "medium",
         complete: Complete | None = None, log: Callable[[str], None] = print, miner: str = "traces") -> dict:
    from rrsi.harness.llm import DEFAULT_MODEL
    mode = MODES[miner]
    out = ensure_private((out or mode.out).expanduser())
    model = model or DEFAULT_MODEL.get(backend, "")
    if complete is None:
        def complete(system, prompt, schema, o):
            for attempt in range(RETRIES + 1):  # one retry for a malformed reply
                try:
                    return complete_json(backend, model, system, prompt, schema,
                                         ensure_private(o / "llm-cwd"), effort)
                except LLMAuthError:
                    raise
                except LLMError as e:
                    if len(e.args) > 1:  # keep the raw reply, privately, for debugging
                        (o / "llm-failures").mkdir(exist_ok=True)
                        (o / "llm-failures" / f"{time.time_ns()}.txt").write_text(str(e.args[1]))
                    if attempt == RETRIES:
                        raise
    t0 = time.time()
    secs = {}
    if not skip_traces:
        summary = run_miner(mode.miner, out, root, since, 8, list(exclude or []) + ["rrsi-private"], mode.miner_args)
    else:
        summary = json.loads((out / mode.summary_file).read_text())
    secs["miner"] = summary.get("seconds", 0.0)
    log(f"[{mode.miner}] {summary['episodes']} episodes from {summary['transcripts']} transcripts")
    episodes = load_episodes(out)
    t = time.time()
    labels, failed_label = label(out, episodes, complete, batch, jobs, log, mode.label_system)
    secs["label"] = time.time() - t
    live = {e["id"] for e in episodes}
    labels = {k: v for k, v in labels.items() if k in live}
    t = time.time()
    mapping = cluster(out, labels, complete, log)
    secs["cluster"] = time.time() - t
    groups = evidence(episodes, labels, mapping)
    t = time.time()
    tasks, failed_tasks = make_tasks(out, groups, labels, complete, top, min_episodes, jobs, log,
                                     mode.task_system, mode.task_schema)
    secs["tasks"] = time.time() - t
    secs["total"] = time.time() - t0 + secs["miner"]
    run = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "backend": backend, "model": model,
           "miner": mode.miner, "summary": summary, "clusters": len(groups), "tasks": len(tasks),
           "failed_calls": failed_label + failed_tasks, "seconds": secs,
           "top": [{"title": t["title"], **{k: t["evidence"][k] for k in ("episodes", "sessions", "projects")},
                    **({"trigger_rule": t["trigger_rule"]["rule"]} if "trigger_rule" in t else {})}
                   for t in tasks[:15]]}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    path = report(out, run, tasks, labels, mode=mode)
    log(f"[report] {path}")
    return run
