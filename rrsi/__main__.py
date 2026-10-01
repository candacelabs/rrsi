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
"""`python -m rrsi harness mine ...` (the RRSI loop itself stays in rrsi.py).

    python -m rrsi harness mine [--miner traces|handoffs|pr-gap] [--out DIR] [--root ~/.claude/projects]
        [--since 2026-09-01] [--backend sdk|copilot|codex] [--model M] [--jobs 4]
        [--batch 25] [--top 20] [--skip-traces]
"""

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rrsi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="group", required=True)
    h = sub.add_parser("harness", help="mine agent struggles from session transcripts")
    hs = h.add_subparsers(dest="cmd", required=True)
    m = hs.add_parser("mine", help="traces (Rust) -> label/cluster/tasks (LLM) -> REPORT.md")
    from rrsi.harness.llm import BACKENDS
    from rrsi.harness.mine import MODES
    m.add_argument("--miner", choices=sorted(MODES), default="traces",
                   help="traces: struggles; handoffs: cross-session coordination with trigger rules; "
                        "pr-gap: active agents without a pushed branch or PR (joins GitHub)")
    m.add_argument("--out", type=Path, default=None,
                   help="outside every git work tree; default ~/rrsi-private/harness[/handoffs|/pr-gap]")
    m.add_argument("--root", type=Path, default=None, help="default ~/.claude/projects")
    m.add_argument("--since", default="", help="only episodes on/after this ISO date")
    m.add_argument("--backend", choices=BACKENDS, default="sdk")
    m.add_argument("--model", default=None, help="default claude-opus-5-5 on the SDK")
    m.add_argument("--effort", default="medium", help="SDK effort level")
    m.add_argument("--jobs", type=int, default=4, help="concurrent LLM calls")
    m.add_argument("--batch", type=int, default=25, help="episodes per labelling call")
    m.add_argument("--top", type=int, default=20, help="clusters that become tasks")
    m.add_argument("--min-episodes", type=int, default=2)
    m.add_argument("--exclude", action="append", default=[], help="skip transcripts whose path contains this")
    m.add_argument("--skip-traces", "--skip-miner", dest="skip_traces", action="store_true",
                   help="reuse OUT/episodes.jsonl from the last miner run")
    a = ap.parse_args(argv)
    from rrsi.harness.mine import mine
    mine(out=a.out, root=a.root, since=a.since, backend=a.backend, model=a.model, jobs=a.jobs,
         batch=a.batch, top=a.top, min_episodes=a.min_episodes, exclude=a.exclude,
         skip_traces=a.skip_traces, effort=a.effort, miner=a.miner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
