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
"""Reference solutions: proof that every hidden test in tasks.py is passable.

tests/test_toy_domain.py runs each task's hidden tests against these. Nothing
at run time reads this file; the critic rejects any harness edit naming it.
"""

import datetime as _dt
import heapq
import re
from decimal import ROUND_HALF_UP, Decimal

_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman_to_int(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and v[c] < v[s[i + 1]]:
            total -= v[c]
        else:
            total += v[c]
    return total


def int_to_roman(n):
    if not 1 <= n <= 3999:
        raise ValueError(n)
    out = []
    for val, sym in _ROMAN:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def rle_encode(s):
    return "".join(f"{m.group(1)}{len(m.group(0))}" for m in re.finditer(r"(.)\1*", s, re.S))


def parse_duration(s):
    m = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s.strip())
    if not s.strip() or not m or not any(m.groups()):
        raise ValueError(s)
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def merge_intervals(xs):
    out = []
    for a, b in sorted(xs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def wildcard_match(p, s):
    i = j = 0
    star, mark = -1, 0
    while j < len(s):
        if i < len(p) and (p[i] == "?" or p[i] == s[j]):
            i += 1
            j += 1
        elif i < len(p) and p[i] == "*":
            star, mark = i, j
            i += 1
        elif star != -1:
            i, mark = star + 1, mark + 1
            j = mark
        else:
            return False
    return all(c == "*" for c in p[i:])


def balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    st, quote = [], None
    for c in s:
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "([{":
            st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]:
                return False
    return not st


def semver_cmp(a, b):
    def key(v):
        core, _, pre = v.partition("-")
        nums = tuple(int(x) for x in core.split("."))
        if not pre:
            return nums, (1,)
        ids = tuple((0, int(x), "") if x.isdigit() else (1, 0, x) for x in pre.split("."))
        return nums, (0, ids)
    ka, kb = key(a), key(b)
    return (ka > kb) - (ka < kb)


def word_wrap(text, width):
    words = []
    for w in text.split():
        words += [w[i:i + width] for i in range(0, len(w), width)]
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def spiral_order(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out


def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in ("+", "-", "*", "/"):
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b if t == "*"
                      else int(a / b))
        else:
            st.append(int(t))
    return st[0]


def flatten(x):
    out, stack = [], [iter(x)]
    while stack:
        for v in stack[-1]:
            if isinstance(v, list):
                stack.append(iter(v))
                break
            out.append(v)
        else:
            stack.pop()
    return out


def snake_case(name):
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def count_islands(grid):
    seen, n = set(), 0
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "#" and (r, c) not in seen:
                n += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < len(grid) and 0 <= nx < len(grid[ny])
                                and grid[ny][nx] == "#" and (ny, nx) not in seen):
                            seen.add((ny, nx))
                            stack.append((ny, nx))
    return n


def next_permutation(xs):
    a = list(xs)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a


def business_days(start, end):
    s, e = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    if e < s:
        return -business_days(end, start)
    return sum(1 for i in range((e - s).days) if (s + _dt.timedelta(i)).weekday() < 5)


def normalize_path(p):
    absolute = p.startswith("/")
    out = []
    for part in p.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if out and out[-1] != "..":
                out.pop()
            elif not absolute:
                out.append("..")
        else:
            out.append(part)
    body = "/".join(out)
    return ("/" + body) if absolute else (body or ".")


def top_k_words(text, k):
    counts = {}
    for w in re.findall(r"[A-Za-z']+", text):
        w = w.lower()
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def csv_line(line):
    out, cur, i, q = [], [], 0, False
    while i < len(line):
        c = line[i]
        if q:
            if c == '"' and i + 1 < len(line) and line[i + 1] == '"':
                cur.append('"')
                i += 1
            elif c == '"':
                q = False
            else:
                cur.append(c)
        elif c == '"':
            q = True
        elif c == ",":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    return out + ["".join(cur)]


def to_base(n, b):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    sign, n, out = ("-" if n < 0 else ""), abs(n), []
    while n:
        n, r = divmod(n, b)
        out.append(digits[r])
    return sign + "".join(reversed(out))


def rotate(xs, k):
    if not xs:
        return []
    k %= len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)


def valid_ipv4(s):
    parts = s.split(".")
    return len(parts) == 4 and all(
        p.isdigit() and p.isascii() and (p == "0" or not p.startswith("0")) and int(p) <= 255
        for p in parts)


def shortest_path(edges, src, dst):
    g = {}
    for a, b, w in edges:
        g.setdefault(a, []).append((b, w))
    dist, pq = {src: 0}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v, w in g.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return -1


def median(xs):
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def dedupe_ci(words):
    seen, out = set(), []
    for w in words:
        k = w.casefold()
        if k not in seen:
            seen.add(k)
            out.append(w)
    return out


def format_money(x):
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    sign = "-" if d < 0 else ""
    return f"{sign}${abs(d):,.2f}"


def run_lengths(xs):
    out = []
    for x in xs:
        if out and out[-1][0] == x:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def title_case(s):
    small = {"a", "an", "the", "of", "and", "in"}
    words = s.split(" ") if s else []
    return " ".join(w.lower() if 0 < i < len(words) - 1 and w.lower() in small
                    else w[:1].upper() + w[1:].lower() for i, w in enumerate(words))


def chunk_sum(xs, size):
    if size < 1:
        raise ValueError(size)
    return [sum(xs[i:i + size]) for i in range(0, len(xs), size)]
