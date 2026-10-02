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
"""The compounding measurement: harness-fixable struggle episodes per 1,000
agent tool calls, per day, by host and by canonical pattern.

    python -m rrsi harness measure [--out ~/rrsi-private/harness] [--root ~/.claude/projects]
                                   [--csf-root ~/.local/state/csf/harness] [--since DATE]
                                   [--backend sdk|copilot|codex] [--skip-miner] [--skip-label]
                                   [--now ISO] [--print]

The procedure (observability ontology: procedure -> run -> series -> panel):

1. `rrsi-mine traces` mines the transcripts incrementally (an unchanged
   transcript is skipped; its facts stay in `traces-state.json`).
2. New episodes are labelled with the canonical vocabulary
   (`patterns.json`); earlier free-slug labels are folded through its
   aliases on load. Labelled episodes are never sent again.
3. Each session gets a denominator and a host behind one interface
   ([`ToolCallSource`]): tool calls per UTC hour from the transcripts, and
   from the CSF harness run directories (`<csf-root>/<assignment>/events.jsonl`)
   for sessions the harness launched. Host: `csf_harness` (a run directory
   names the session), else `desktop_hosted` (Claude Desktop host markers:
   its injected hooks registered on the session, its hook timeouts or its
   worktree-guard rejections), else `cli`.
4. Days close at 23:59:59 America/Los_Angeles. Every day with a tool call or
   an episode gets `OUT/daily/<day>.json` (`daily.schema.json`; `closed`
   says whether the day had ended when it was computed, so a partial day is
   never mistaken for a final one), `OUT/series.jsonl` carries the dashboard
   series, and `OUT/measure-run.json` is the run receipt.
5. `OUT/MEASURE.md` is the report: verdict first, rate by week with n and
   95% CI, top canonical patterns, the host A/B. Counts only, no transcript
   text; it can be posted as is.

Safe to run at any time, as often as wanted (per session end, hourly, at
02:00): every step is incremental or a rebuild from cached inputs.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol
from zoneinfo import ZoneInfo

from rrsi.harness import mine as M
from rrsi.harness.patterns import NOISE, Vocabulary

SCHEMA = "rrsi.harness.daily/1"
TIMEZONE = "America/Los_Angeles"
TZ = ZoneInfo(TIMEZONE)
SCHEMA_PATH = Path(__file__).with_name("daily.schema.json")
SERIES_RATE = "rrsi.harness.struggle_rate_per_1k_tool_calls"
SERIES_EPISODES = "rrsi.harness.struggle_episodes"
DEFAULT_CSF_ROOT = Path.home() / ".local" / "state" / "csf" / "harness"
Host = Literal["csf_harness", "desktop_hosted", "cli"]
HOSTS: tuple[Host, ...] = ("csf_harness", "desktop_hosted", "cli")
Z95 = 1.959964
#: A week (or day) with fewer tool calls than this does not enter the trend fit.
ACTIVE_CALLS = 200
#: The trend needs at least this many weeks with a struggle.
TREND_MIN_WEEKS = 3


# ------------------------------------------------------------------ time

def utc(ts: str) -> datetime:
    """An ISO timestamp (`...Z` or with offset, any fraction) as an aware UTC datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def la_day(ts: str) -> str:
    """The America/Los_Angeles calendar day an ISO timestamp falls in."""
    return utc(ts).astimezone(TZ).date().isoformat()


def hour_day(hour: str) -> str:
    """The LA day of a UTC hour bucket `YYYY-MM-DDTHH` (the transcript facts)."""
    return datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc).astimezone(TZ).date().isoformat()


def day_closed(day: str, now: datetime) -> bool:
    """True once 23:59:59 of `day` in LA has passed at `now`."""
    end = datetime.combine(date.fromisoformat(day) + timedelta(days=1), datetime.min.time(), tzinfo=TZ)
    return now.astimezone(timezone.utc) >= end.astimezone(timezone.utc)


def week_of(day: str) -> str:
    d = date.fromisoformat(day)
    return (d - timedelta(days=d.weekday())).isoformat()


# ------------------------------------------------------------- sessions

@dataclass
class SessionCalls:
    """One session as one source saw it."""
    session_id: str
    calls_by_utc_hour: dict[str, int]
    hook_timeouts: int = 0
    guard_rejections: int = 0
    host_hook_callbacks: int = 0
    assignment: str | None = None

    @property
    def tool_calls(self) -> int:
        return sum(self.calls_by_utc_hour.values())

    @property
    def desktop_markers(self) -> int:
        return self.hook_timeouts + self.guard_rejections + self.host_hook_callbacks


class ToolCallSource(Protocol):
    def sessions(self) -> Iterable[SessionCalls]: ...


class TranscriptCalls:
    """Tool calls and host markers per session from `traces-state.json`
    (the facts `rrsi-mine traces` records per transcript); a subagent
    transcript belongs to its parent session."""

    def __init__(self, state_path: Path):
        self.state_path = state_path

    @staticmethod
    def session_of(rel: str) -> str:
        parts = rel.split("/")
        return parts[1].removesuffix(".jsonl") if len(parts) > 1 else rel.removesuffix(".jsonl")

    def sessions(self) -> Iterable[SessionCalls]:
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        by: dict[str, SessionCalls] = {}
        for rel, st in sorted(state.items()):
            f = st.get("facts") or {}
            s = by.setdefault(self.session_of(rel), SessionCalls(self.session_of(rel), {}))
            for h, n in f.get("tool_calls_by_utc_hour", {}).items():
                s.calls_by_utc_hour[h] = s.calls_by_utc_hour.get(h, 0) + n
            s.hook_timeouts += f.get("hook_timeouts", 0)
            s.guard_rejections += f.get("guard_rejections", 0)
            s.host_hook_callbacks += f.get("host_hook_callbacks", 0)
        return by.values()


class HarnessCalls:
    """Tool calls per session from the CSF harness run directories: each
    `<root>/<assignment>/run.json` names the session, and its `events.jsonl`
    carries every assistant message (subagents included) with its tool calls."""

    def __init__(self, root: Path):
        self.root = root

    def sessions(self) -> Iterable[SessionCalls]:
        if not self.root.is_dir():
            return []
        out: dict[str, SessionCalls] = {}
        for d in sorted(self.root.iterdir()):
            run, events = d / "run.json", d / "events.jsonl"
            if not (run.is_file() and events.is_file()):
                continue
            try:
                sid = json.loads(run.read_text())["session_id"]
            except (ValueError, KeyError):
                continue
            s = out.setdefault(sid, SessionCalls(sid, {}, assignment=d.name))
            with events.open() as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    ev = e.get("event") or {}
                    if ev.get("type") != "assistant":
                        continue
                    hour = str(e.get("time", ""))[:13]
                    for b in (ev.get("message") or {}).get("content", []) or []:
                        if isinstance(b, dict) and b.get("type") == "tool_use" and len(hour) == 13:
                            s.calls_by_utc_hour[hour] = s.calls_by_utc_hour.get(hour, 0) + 1
        return out.values()


@dataclass
class Session:
    session_id: str
    host: Host
    denominator: Literal["csf_harness", "transcript"]
    calls_by_utc_hour: dict[str, int]
    tool_calls_transcript: int
    tool_calls_harness: int | None
    hook_timeouts: int
    guard_rejections: int
    host_hook_callbacks: int
    assignment: str | None

    @property
    def tool_calls(self) -> int:
        return sum(self.calls_by_utc_hour.values())


def attribute(transcripts: Iterable[SessionCalls], harness: Iterable[SessionCalls]) -> dict[str, Session]:
    """Joins both sources on the session id and names each session's host.
    A harness-launched session's denominator is the harness record (typed,
    subagents included); the transcript count is kept next to it."""
    t = {s.session_id: s for s in transcripts}
    h = {s.session_id: s for s in harness}
    out: dict[str, Session] = {}
    for sid in sorted(set(t) | set(h)):
        ts, hs = t.get(sid), h.get(sid)
        if hs is not None:
            host, denom, calls = "csf_harness", "csf_harness", hs.calls_by_utc_hour
        else:
            host, denom, calls = ("desktop_hosted" if ts.desktop_markers else "cli"), "transcript", ts.calls_by_utc_hour
        out[sid] = Session(sid, host, denom, dict(calls), ts.tool_calls if ts else 0, hs.tool_calls if hs else None,
                           ts.hook_timeouts if ts else 0, ts.guard_rejections if ts else 0,
                           ts.host_hook_callbacks if ts else 0, hs.assignment if hs else None)
    return out


# ----------------------------------------------------------------- stats

def chi2_quantile(df: float, upper: bool) -> float:
    """The 0.975 (upper) or 0.025 quantile of chi-square with `df` degrees of
    freedom, by the Wilson-Hilferty approximation (within a few percent of
    exact for df >= 2; the bounds it gives are the Garwood Poisson limits)."""
    z = Z95 if upper else -Z95
    return df * max(0.0, 1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def poisson_ci(k: int, n: int) -> list[float] | None:
    """95% interval of k events per 1,000 units of exposure n."""
    if n <= 0:
        return None
    lo = 0.0 if k == 0 else chi2_quantile(2 * k, False) / 2
    hi = chi2_quantile(2 * k + 2, True) / 2
    return [round(1000 * lo / n, 3), round(1000 * hi / n, 3)]


def rate(k: int, n: int) -> float | None:
    return round(1000 * k / n, 3) if n > 0 else None


@dataclass
class Trend:
    weeks: int
    factor: float | None  # multiplicative change of the rate per week
    ci95: list[float] | None

    def verdict(self) -> str:
        if self.factor is None or self.ci95 is None:
            return (f"Not estimable yet: the trend needs {TREND_MIN_WEEKS} weeks with at least {ACTIVE_CALLS} tool calls "
                    f"and one struggle each ({self.weeks} so far).")
        lo, hi = self.ci95
        shape = f"×{self.factor:.3f} per week (95% CI ×{lo:.3f}..×{hi:.3f}, {self.weeks} weeks)"
        if hi < 1:
            return f"Declining: harness-fixable struggles per 1k tool calls fall {shape}."
        if lo > 1:
            return f"Rising: harness-fixable struggles per 1k tool calls grow {shape}."
        return f"Flat: compounding is not demonstrated; the weekly factor {shape} includes 1."


def trend(points: list[tuple[int, int, int]]) -> Trend:
    """Weighted least squares of ln(k/n) on the week index over (index, k, n)
    points with k >= 1 and n >= ACTIVE_CALLS; weight k (the Poisson variance
    of ln k is about 1/k). Returns the per-week factor with its 95% CI."""
    pts = [(x, k, n) for x, k, n in points if k >= 1 and n >= ACTIVE_CALLS]
    if len(pts) < TREND_MIN_WEEKS:
        return Trend(len(pts), None, None)
    w = [float(k) for _, k, _ in pts]
    xs = [float(x) for x, _, _ in pts]
    ys = [math.log(k / n) for _, k, n in pts]
    sw = sum(w)
    xbar = sum(wi * x for wi, x in zip(w, xs)) / sw
    ybar = sum(wi * y for wi, y in zip(w, ys)) / sw
    sxx = sum(wi * (x - xbar) ** 2 for wi, x in zip(w, xs))
    if sxx <= 0:
        return Trend(len(pts), None, None)
    b = sum(wi * (x - xbar) * (y - ybar) for wi, x, y in zip(w, xs, ys)) / sxx
    se = math.sqrt(1 / sxx)
    return Trend(len(pts), round(math.exp(b), 4), [round(math.exp(b - Z95 * se), 4), round(math.exp(b + Z95 * se), 4)])


# ------------------------------------------------------------------ days

@dataclass
class Slice:
    sessions: int = 0
    tool_calls: int = 0
    struggles: int = 0
    rate_per_1k: float | None = None
    ci95: list[float] | None = None

    def close(self) -> "Slice":
        self.rate_per_1k, self.ci95 = rate(self.struggles, self.tool_calls), poisson_ci(self.struggles, self.tool_calls)
        return self


@dataclass
class DailyRecord:
    day: str
    closed: bool
    computed_at: str
    revision: str
    vocabulary: str
    sessions: int = 0
    tool_calls: int = 0
    episodes: int = 0
    labelled: int = 0
    noise: int = 0
    struggles: int = 0
    rate_per_1k: float | None = None
    ci95: list[float] | None = None
    hosts: dict[str, Slice] = field(default_factory=dict)
    patterns: dict[str, dict] = field(default_factory=dict)
    uncanonical: dict[str, int] = field(default_factory=dict)
    schema: str = SCHEMA
    timezone: str = TIMEZONE

    def to_json(self) -> dict:
        d = asdict(self)
        d["hosts"] = {h: asdict(s) for h, s in self.hosts.items()}
        return d


@dataclass
class Labelled:
    """One episode joined to its label."""
    id: str
    session_id: str
    day: str | None
    pattern: str | None   # None: unlabelled
    canonical: bool
    harness_fixable: bool

    @property
    def struggle(self) -> bool:
        return self.pattern is not None and self.pattern != NOISE and self.harness_fixable


def join(episodes: list[dict], labels: dict[str, dict]) -> list[Labelled]:
    out = []
    for e in episodes:
        rec = labels.get(e["id"])
        day = la_day(e["start"]) if e.get("start") else None
        out.append(Labelled(e["id"], e["session_id"], day, rec["pattern"] if rec else None,
                            bool(rec.get("canonical", True)) if rec else False, bool(rec["harness_fixable"]) if rec else False))
    return out


def days(sessions: dict[str, Session], eps: list[Labelled], now: datetime, revision: str, vocabulary: str) -> list[DailyRecord]:
    computed = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recs: dict[str, DailyRecord] = {}

    def rec(day: str) -> DailyRecord:
        return recs.setdefault(day, DailyRecord(day, day_closed(day, now), computed, revision, vocabulary))

    active: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # day -> host -> sessions
    for s in sessions.values():
        for hour, n in s.calls_by_utc_hour.items():
            r = rec(hour_day(hour))
            r.tool_calls += n
            r.hosts.setdefault(s.host, Slice()).tool_calls += n
            active[r.day][s.host].add(s.session_id)
    for e in eps:
        if e.day is None:
            continue
        r = rec(e.day)
        host = sessions[e.session_id].host if e.session_id in sessions else "cli"
        active[r.day][host].add(e.session_id)
        r.episodes += 1
        if e.pattern is None:
            continue
        r.labelled += 1
        if e.pattern == NOISE:
            r.noise += 1
            continue
        if e.struggle:
            r.struggles += 1
            r.hosts.setdefault(host, Slice()).struggles += 1
        if e.canonical:
            p = r.patterns.setdefault(e.pattern, {"episodes": 0, "harness_fixable": 0, "hosts": {}})
            p["episodes"] += 1
            p["harness_fixable"] += int(e.harness_fixable)
            p["hosts"][host] = p["hosts"].get(host, 0) + 1
        else:
            r.uncanonical[e.pattern] = r.uncanonical.get(e.pattern, 0) + 1
    for r in recs.values():
        for host, sl in r.hosts.items():
            sl.sessions = len(active[r.day][host])
            sl.close()
        r.sessions = len(set().union(*active[r.day].values())) if active[r.day] else 0
        r.rate_per_1k, r.ci95 = rate(r.struggles, r.tool_calls), poisson_ci(r.struggles, r.tool_calls)
    return [recs[d] for d in sorted(recs)]


def series(recs: list[DailyRecord]) -> list[dict]:
    """The dashboard series: the rate per day and host, and episodes per day,
    host and canonical pattern (bounded labels; evidence stays in the records)."""
    out = []
    for r in recs:
        base = {"day": r.day, "closed": r.closed, "computed_at": r.computed_at}
        out.append({"series": SERIES_RATE, "unit": "episodes per 1000 tool calls", **base, "host": "all",
                    "tool_calls": r.tool_calls, "struggles": r.struggles, "rate_per_1k": r.rate_per_1k, "ci95": r.ci95})
        for host in HOSTS:
            if host in r.hosts:
                s = r.hosts[host]
                out.append({"series": SERIES_RATE, "unit": "episodes per 1000 tool calls", **base, "host": host,
                            "tool_calls": s.tool_calls, "struggles": s.struggles, "rate_per_1k": s.rate_per_1k, "ci95": s.ci95})
        for pat, p in sorted(r.patterns.items()):
            for host, n in sorted(p["hosts"].items()):
                out.append({"series": SERIES_EPISODES, "unit": "episodes", **base, "host": host, "pattern": pat, "episodes": n})
    return out


# ---------------------------------------------------------------- report

def _ci(ci: list[float] | None) -> str:
    return f"{ci[0]:.1f}..{ci[1]:.1f}" if ci else "-"


def _rate(r: float | None) -> str:
    return f"{r:.1f}" if r is not None else "-"


def report(recs: list[DailyRecord], sessions: dict[str, Session], eps: list[Labelled], vocab: Vocabulary,
           folding: dict, revision: str, top: int = 12) -> str:
    weeks: dict[str, dict] = defaultdict(lambda: {"sessions": set(), "calls": 0, "struggles": 0, "open": False})
    for r in recs:
        w = weeks[week_of(r.day)]
        w["calls"] += r.tool_calls
        w["struggles"] += r.struggles
        w["open"] |= not r.closed
    for s in sessions.values():
        for hour in s.calls_by_utc_hour:
            weeks[week_of(hour_day(hour))]["sessions"].add(s.session_id)
    order = sorted(weeks)
    tr = trend([(i, weeks[w]["struggles"], weeks[w]["calls"]) for i, w in enumerate(order)])
    struggles = [e for e in eps if e.struggle]
    total_calls = sum(r.tool_calls for r in recs)
    lines = [
        "# Struggle rate per 1k tool calls (harness-fixable episodes)", "",
        f"**Verdict:** {tr.verdict()}", "",
        f"Procedure `python -m rrsi harness measure` at rrsi `{revision}`, vocabulary `{vocab.fingerprint}` "
        f"(version {vocab.version}, {len(vocab.ids)} canonical patterns). Numerator: episodes from the nine `rrsi-mine traces` "
        f"detectors, labelled by a model with a canonical pattern id, non-noise and harness-fixable. Denominator: every agent "
        f"tool call (subagents included), from the CSF harness run record for sessions it launched and from the transcript "
        f"otherwise. Days close at 23:59:59 {TIMEZONE}; weeks start Monday. Intervals: 95% Garwood Poisson bounds "
        f"(Wilson-Hilferty). Trend: weighted log-linear fit over weeks with at least {ACTIVE_CALLS} tool calls and one struggle.", "",
        f"Corpus: {len(sessions)} sessions, {total_calls} tool calls, {len(eps)} episodes "
        f"({sum(1 for e in eps if e.pattern is None)} unlabelled, {sum(1 for e in eps if e.pattern == NOISE)} noise, "
        f"{len(struggles)} harness-fixable struggles) over {len(recs)} days "
        f"({sum(1 for r in recs if not r.closed)} still open).", "",
        "## Rate by week", "",
        "| Week of | Sessions | Tool calls | Struggles | per 1k | 95% CI | |", "|---|---|---|---|---|---|---|",
    ]
    for w in order:
        d = weeks[w]
        lines.append(f"| {w} | {len(d['sessions'])} | {d['calls']} | {d['struggles']} | {_rate(rate(d['struggles'], d['calls']))} | "
                     f"{_ci(poisson_ci(d['struggles'], d['calls']))} | {'open' if d['open'] else ''} |")
    by_pat: Counter = Counter(e.pattern for e in struggles if e.canonical)
    pat_sessions: dict[str, set] = defaultdict(set)
    pat_hosts: dict[str, Counter] = defaultdict(Counter)
    for e in struggles:
        pat_sessions[e.pattern].add(e.session_id)
        pat_hosts[e.pattern][sessions[e.session_id].host if e.session_id in sessions else "cli"] += 1
    lines += ["", f"## Top canonical patterns ({len(struggles)} harness-fixable struggles)", "",
              "| Pattern | Episodes | Sessions | Share | What it names |", "|---|---|---|---|---|"]
    for pat, n in by_pat.most_common(top):
        lines.append(f"| `{pat}` | {n} | {len(pat_sessions[pat])} | {100 * n / max(len(struggles), 1):.0f}% | {vocab.title(pat)} |")
    unc = Counter(e.pattern for e in struggles if not e.canonical)
    if unc:
        lines.append(f"| *uncanonical (new_pattern escapes)* | {sum(unc.values())} | | | {len(unc)} slugs |")
    lines += ["", "## Host split (A/B)", "",
              "| Host | Sessions | Tool calls | Struggles | per 1k | 95% CI |", "|---|---|---|---|---|---|"]
    host_calls: Counter = Counter()
    host_sessions: Counter = Counter()
    host_struggles: Counter = Counter()
    for s in sessions.values():
        host_calls[s.host] += s.tool_calls
        host_sessions[s.host] += 1
    for e in struggles:
        host_struggles[sessions[e.session_id].host if e.session_id in sessions else "cli"] += 1
    for host in HOSTS:
        n, k = host_calls[host], host_struggles[host]
        lines.append(f"| {host} | {host_sessions[host]} | {n} | {k} | {_rate(rate(k, n))} | {_ci(poisson_ci(k, n))} |")
    lines += ["", "Per pattern, episodes per 1k tool calls of that host (episodes in brackets):", "",
              "| Pattern | " + " | ".join(HOSTS) + " |", "|---|" + "---|" * len(HOSTS)]
    for pat, _ in by_pat.most_common(top):
        cells = [f"{_rate(rate(pat_hosts[pat][h], host_calls[h]))} ({pat_hosts[pat][h]})" for h in HOSTS]
        lines.append(f"| `{pat}` | " + " | ".join(cells) + " |")
    csf = [s for s in sessions.values() if s.host == "csf_harness"]
    agree = sum(min(s.tool_calls_transcript, s.tool_calls_harness or 0) for s in csf)
    desk = [s for s in sessions.values() if s.host == "desktop_hosted"]
    lines += ["", "## Vocabulary and coverage", "",
              f"- Hosts: a session is `csf_harness` when a harness run record names it; `desktop_hosted` when the transcript "
              f"shows the Desktop host's injected hooks ({sum(1 for s in desk if s.host_hook_callbacks)} of {len(desk)} "
              f"such sessions register them; {sum(1 for s in desk if s.hook_timeouts)} have hook timeouts, "
              f"{sum(1 for s in desk if s.guard_rejections)} worktree-guard rejections); else `cli`. CSF sessions with any "
              f"Desktop marker: {sum(1 for s in csf if s.hook_timeouts or s.guard_rejections or s.host_hook_callbacks)}.",
              f"- Raw label slugs {folding['raw_slugs']} -> {folding['canonical_ids']} canonical ids in use (+ noise); "
              f"{folding['aliases_folded']} slugs folded through aliases, {folding['uncanonical']} left uncanonical; "
              f"{folding['labels_remapped']} of {folding['labels']} labels changed name.",
              f"- CSF sessions: {len(csf)}; their harness records count {sum(s.tool_calls_harness or 0 for s in csf)} tool calls "
              f"against {sum(s.tool_calls_transcript for s in csf)} in the transcripts ({agree} in common).",
              f"- Days: {len(recs)} with data, {sum(1 for r in recs if r.closed)} closed, "
              f"{sum(1 for r in recs if not r.closed)} open (recomputed by the next run).",
              f"- Unlabelled episodes: {sum(1 for e in eps if e.pattern is None)} (they make the rate a lower bound).", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- run

def revision_of(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def folding_stats(raw_labels: dict[str, dict], labels: dict[str, dict], vocab: Vocabulary) -> dict:
    raw = {r.get("raw", r["pattern"]) for r in raw_labels.values()} - {NOISE}
    return {"labels": len(labels), "raw_slugs": len(raw),
            "canonical_ids": len({r["pattern"] for r in labels.values() if r.get("canonical", True) and r["pattern"] != NOISE}),
            "aliases_folded": sum(1 for s in raw if s in vocab.alias_of),
            "uncanonical": sum(1 for s in raw if s not in vocab.alias_of and s not in vocab.ids),
            "labels_remapped": sum(1 for i, r in raw_labels.items() if labels.get(i, r)["pattern"] != r["pattern"])}


def measure(out: Path | None = None, root: Path | None = None, csf_root: Path | None = None, since: str = "",
            backend: str = "sdk", model: str | None = None, effort: str = "medium", jobs: int = 4, batch: int = 25,
            exclude: list[str] | None = None, skip_miner: bool = False, skip_label: bool = False,
            now: datetime | None = None, complete: M.Complete | None = None,
            log: Callable[[str], None] = print) -> dict:
    from rrsi.harness.llm import DEFAULT_MODEL
    t0 = time.time()
    out = M.ensure_private((out or M.DEFAULT_OUT).expanduser())
    csf_root = (csf_root or DEFAULT_CSF_ROOT).expanduser()
    now = now or datetime.now(timezone.utc)
    vocab = Vocabulary.load()
    revision = revision_of(M.ROOT)
    if skip_miner:
        summary = json.loads((out / "traces-summary.json").read_text()) if (out / "traces-summary.json").exists() else {}
    else:
        summary = M.run_miner("traces", out, root, since, 8, list(exclude or []) + ["rrsi-private"])
    episodes = M.load_episodes(out)
    log(f"[traces] {len(episodes)} episodes; {summary.get('processed', '?')} transcripts parsed, "
        f"{summary.get('skipped_unchanged', '?')} unchanged")
    raw_labels = M.load_labels(out)
    failed = 0
    if skip_label:
        labels = M.load_labels(out, vocab)
    else:
        model = model or DEFAULT_MODEL.get(backend, "")
        complete = complete or M.completer(backend, model, effort)
        labels, failed = M.label(out, episodes, complete, batch, jobs, log, M.LABEL_SYSTEM, vocab)
    live = {e["id"] for e in episodes}
    labels = {k: v for k, v in labels.items() if k in live}
    folding = folding_stats({k: v for k, v in raw_labels.items() if k in live}, labels, vocab)
    sessions = attribute(TranscriptCalls(out / "traces-state.json").sessions(), HarnessCalls(csf_root).sessions())
    eps = join(episodes, labels)
    recs = days(sessions, eps, now, revision, vocab.fingerprint)
    daily = out / "daily"
    daily.mkdir(exist_ok=True)
    for r in recs:
        (daily / f"{r.day}.json").write_text(json.dumps(r.to_json(), indent=1))
    (out / "series.jsonl").write_text("".join(json.dumps(s) + "\n" for s in series(recs)))
    text = report(recs, sessions, eps, vocab, folding, revision)
    (out / "MEASURE.md").write_text(text)
    hosts = Counter(s.host for s in sessions.values())
    receipt = {
        "procedure": "python -m rrsi harness measure", "revision": revision, "vocabulary": vocab.fingerprint,
        "started": datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "now": now.astimezone(timezone.utc).isoformat(),
        "inputs": {"out": str(out), "root": str(root or ""), "csf_root": str(csf_root), "since": since,
                   "backend": None if skip_label else backend, "model": None if skip_label else model},
        "miner": {k: summary.get(k) for k in ("transcripts", "processed", "skipped_unchanged", "episodes")},
        "labels": {"total": len(labels), "unlabelled": sum(1 for e in eps if e.pattern is None), "failed_batches": failed,
                   **folding},
        "sessions": {h: hosts.get(h, 0) for h in HOSTS},
        "days": {"written": len(recs), "closed": sum(1 for r in recs if r.closed), "open": sum(1 for r in recs if not r.closed)},
        "outcome": "ok" if failed == 0 else "partial: some label batches failed; their episodes are unlabelled",
        "evidence": {"daily": str(daily), "series": str(out / "series.jsonl"), "report": str(out / "MEASURE.md")},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "measure-run.json").write_text(json.dumps(receipt, indent=1))
    log(f"[measure] {len(recs)} days ({receipt['days']['open']} open), sessions {dict(hosts)}, report {out / 'MEASURE.md'}")
    return {"receipt": receipt, "report": text, "days": recs}
