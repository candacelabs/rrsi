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
"""The canonical struggle-pattern vocabulary (`patterns.json`).

A label names a *cause*; a model asked for a free slug names the same cause
many ways (one baseline pattern arrived under four spellings). The
vocabulary fixes the ids: the labeler chooses one, and every slug it or an
earlier run produced is folded to its id through the alias lists.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

PatternId = NewType("PatternId", str)
NOISE = PatternId("noise")
NEW_PATTERN = "new_pattern"
DEFAULT_PATH = Path(__file__).with_name("patterns.json")


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (t or "").lower()).strip("_")[:60] or NOISE


@dataclass(frozen=True)
class Pattern:
    id: PatternId
    title: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Folded:
    """One raw slug's canonical form."""
    pattern: PatternId
    canonical: bool  # False: an escape (`new_pattern`) nothing in the vocabulary names


class VocabularyError(ValueError):
    pass


class Vocabulary:
    def __init__(self, patterns: list[Pattern], version: int, fingerprint: str):
        self.patterns = patterns
        self.version = version
        self.fingerprint = fingerprint
        self.ids: dict[PatternId, Pattern] = {}
        self.alias_of: dict[str, PatternId] = {}
        for p in patterns:
            if p.id in self.ids or p.id == NOISE or p.id == NEW_PATTERN:
                raise VocabularyError(f"duplicate or reserved pattern id {p.id!r}")
            self.ids[p.id] = p
        for p in patterns:
            for a in p.aliases:
                if a in self.ids or a in self.alias_of or a == NOISE:
                    raise VocabularyError(f"alias {a!r} of {p.id!r} is already an id or an alias")
                self.alias_of[a] = p.id

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Vocabulary":
        raw = path.read_bytes()
        doc = json.loads(raw)
        pats = [Pattern(PatternId(p["id"]), p["title"], tuple(p.get("aliases", []))) for p in doc["patterns"]]
        for p in pats:
            for s in (p.id, *p.aliases):
                if slug(s) != s:
                    raise VocabularyError(f"{s!r} in {path.name} is not a lowercase snake_case slug")
        return cls(pats, int(doc["version"]), hashlib.sha256(raw).hexdigest()[:16])

    def fold(self, raw: str) -> Folded:
        s = slug(raw)
        if s == NOISE:
            return Folded(NOISE, True)
        if s in self.ids:
            return Folded(PatternId(s), True)
        if s in self.alias_of:
            return Folded(self.alias_of[s], True)
        return Folded(PatternId(s), False)

    def title(self, pattern: str) -> str:
        p = self.ids.get(PatternId(pattern))
        return p.title if p else pattern.replace("_", " ")

    def choices(self) -> list[str]:
        """What the labeler may answer: every id, `noise`, or the escape."""
        return [p.id for p in self.patterns] + [NOISE, NEW_PATTERN]

    def prompt_lines(self) -> str:
        return "\n".join(f"- {p.id}: {p.title}" for p in self.patterns)
