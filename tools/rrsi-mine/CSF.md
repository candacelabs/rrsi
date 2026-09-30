<!--
Copyright 2026 Candace Labs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# rrsi-mine on CSF-instrumented repositories

CSF (the Go module `github.com/candacelabs/csf`) lets a
Go repository declare its architecture in an `architecture.csf` file: its
components (`service`, `manager`, `library`, `adapter`, `gateway`,
`resource`), their source paths and lifecycles, the source roots to check and
the paths that are generated. CSF's compiler, `csfc`, checks that declaration
against the code. rrsi-mine uses both, so a task mined from such a repository
knows which declared components its fix touches, never counts generated code
as part of a fix, and records whether the fix passes CSF's own gates.

rrsi-mine never parses `.csf` files itself. The compiler owns the language:
rrsi-mine reads the model only as `csfc emit --format json` prints it (a JSON
document with `"format": "csf-architecture"`, `"format_version": 1`) and runs
csfc's `check` and `check-generated` gates. A csfc without `--format json`
predates this interface; detection then says so.

## Commands

```sh
# Is the repository CSF-instrumented, and why? Exits 0 either way.
rrsi-mine csf detect --repo PATH [--rev REV] [--csfc CSFC] [--json]

# CSF's gates on any checkout (a task's commit tree, or an agent's patched tree).
rrsi-mine csf guard --tree DIR [--csfc CSFC] [--csf-grammar GRAMMAR]

# Mine as before; in a CSF repository each task also gets `csf` and `csf_guards`.
rrsi-mine mine --repo PATH --out DIR --csfc CSFC

# Add `csf` (and with --guards, `csf_guards`) to an existing task directory.
rrsi-mine csf annotate --tasks DIR --repo PATH --csfc CSFC [--guards]

# The report shows a "CSF architecture" section whenever tasks carry CSF data.
rrsi-report --tasks DIR --repo PATH --out report.html [--csf-areas]
```

From Python, `rrsi_mine.csf_guard(tree, csfc=None, grammar=None, sources=None)`
returns the same verdicts as `csf guard`, as a list of dicts.

csfc is found by `--csfc`, then `RRSI_CSFC`, then a `csfc` on `PATH`. csfc
reads CSF's grammar at run time: `--csf-grammar`, then `RRSI_CSF_GRAMMAR`,
then `<root>/csf/compiler/architecture/language.ebnf` when the repository is
a CSF checkout itself. To build csfc from a CSF checkout (no host OCaml; the
launcher runs Bazel in a pinned container):

```sh
bash tools/bazel.sh --batch build //csf/compiler/architecture:csfc --lockfile_mode=error
# binary: bazel-bin/csf/compiler/architecture/csfc.exe
```

Without csfc every CSF feature below still reports what it could not do
(`unchecked`, `skipped`, with the reason) instead of guessing.

## Detection

CSF defines no single marker, so detection reports every signal it finds:

| Signal | Fires when | Read with |
|---|---|---|
| `go_mod_is_csf` | a go.mod declares `module github.com/candacelabs/csf` (CSF itself) | `gomod-parser` |
| `go_mod_requires_csf` | a go.mod requires `github.com/candacelabs/csf` | `gomod-parser` |
| `bazel_module_csf` | a MODULE.bazel call passes `name = "csf"`, e.g. `bazel_dep`, `local_repository`, `module` | `starlark_syntax` |
| `csf_file` | a `*.csf` file is tracked | — |
| `architecture_model` | csfc checked an architecture source and printed its model | csfc |

A repository is instrumented when any signal fires. go.mod and MODULE.bazel
files under `testdata/` or `vendor/` are ignored. Detection reads the chosen
revision straight from git (gitoxide), so any commit can be inspected without
a checkout; csfc needs files, so it runs on a temporary export of that
revision.

**Architecture sources** are `<root>/csf/architecture/architecture.csf`, the
path csfc checks by default, plus any `--csf-source PATH`. The model's root
(csfc's `--root`, which its paths are relative to) is the directory holding
`csf/`. Other `.csf` files, such as CSF's documentation vocabulary, are listed
with status `other` and never sent to csfc.

## The component map (`task.json` → `csf`)

Each changed source file of a task's reference fix (non-test, non-generated
Go) maps to the declared component whose `source` is its longest path prefix;
a component's source may be a directory or a single file. The optional field:

```json
"csf": {"models": ["shop"], "model_rev": "<commit or file:PATH>",
        "components": ["orders"], "kinds": ["service"], "state": "existing",
        "files_mapped": 1, "files_total": 1}
```

`state` is `existing`, `planned`, `mixed` or absent (no component matched).
The model is read once, at `--csf-model-rev` (default `HEAD`), or taken from
`--csf-model FILE --csf-root DIR` (a saved `csfc emit --format json`
document and the directory its paths are relative to). It is the
architecture as declared *then*, applied to every task: a component declared
after a task's commit still claims that task's files. `rrsi-report --csf-areas`
uses the components as each task's area where a fix maps to one.

## Generated code

A file is generated, and so never part of a fix or its churn, when any rule
fires (checked in this order):

| Rule | Fires when |
|---|---|
| `cgen_name` | the file stem ends in `_cgen` (CSF's CandaceCodegen outputs) |
| `heuristic` | the path matches the miner's long-standing list (`/gen/`, `.pb.go`, `_cgen`, `zz_generated`, `_string.go`) |
| `csf_generated_path` | the path is, or lies under, a `generated` root the CSF model declares |
| `generated_header` | one of the file's first 5 lines reads `Code generated ... DO NOT EDIT` (Go's convention, which CSF's banner follows); a deleted file is judged on the parent |

A commit whose only source change is generated is not a task: `mine` writes
it to `rejected.jsonl` as `only generated code changed`, next to every other
commit that touched a test but is not a candidate, with its reason.
`rrsi-mine list --rejected` prints the same list.

## Guards (`task.json` → `csf_guards`)

For each architecture source in a tree, two read-only gates run in the
model's root:

| Gate | Command | Skipped when |
|---|---|---|
| `csfc check` | `csfc check --source S --root . --grammar G` | no csfc, or no grammar |
| `csfc check-generated` | the same with `--output <dir of S>/generated` | as above, or that directory does not exist |

A verdict is `pass` (exit 0), `fail` (exit 1; csfc's `file:line:col: CODE:
message` diagnostics are kept), `error` (any other exit or the 300 s timeout;
says nothing about the tree) or `skipped` (with the reason). `emit` never
runs: guards do not write.

During `mine` the guards run on each task's **commit tree** (the reference
fix). A gate the fix itself fails cannot fairly be required of an agent, so
it is recorded with `required_of_agent: false` and the task is **kept**: its
FAIL_TO_PASS tests are still a fair question, and excluding it would drop
every task of a repository whose gate is red at that time. Only gates with
`required_of_agent: true` should bind an agent's patch; the grader applies
them with `rrsi_mine.csf_guard` on the patched tree. These guards are
non-compensatory: a passing test score does not make up for a failed
required gate.

**Use the csfc of the CSF version the commit pins.** `check-generated`
compares the checked-in projections byte for byte with what *this* csfc
emits, and the projections carry the generator's version banner. A newer or
older csfc therefore reports `CSF_GENERATED_DRIFT` on an unchanged model; the
report calls out drift-only failures for this reason.

## Measured on our Go monorepo

Run on 2026-09-30 against the monorepo the existing 86 candidates were mined
from, with a csfc built from its CSF sources:

| Check | Result |
|---|---|
| `csf detect` | instrumented; 18 signals: `go_mod_is_csf` (1), `go_mod_requires_csf` (5), `bazel_module_csf` (2), `csf_file` (9), `architecture_model` (1: 9 components, 9 generated roots); 8 other `.csf` files listed as `other`; 0.7 s |
| `csf detect` on a new one-commit Go repository | not instrumented; 0 signals |
| Valid tasks (70) mapped to a declared component | **4 of 70**: `composition` (manager) 3, `csf_services` (service) 3 (two tasks touch both). The model declares only CSF's own serve-mode composition, so the rest of the exam exercises code it does not describe. |
| Candidate scan with the new generated-code rules | still 86 candidates: 1 dropped (its only source change was a generated file; it was not a valid task), 1 added (1,041 changed lines, of which 646 templ-generated: 395 without them, under the 400-line limit; it then validated FAIL_TO_PASS), and 3 tasks lose a templ-generated file from their fix; 129 other commits rejected with a reason |
| Guards backfilled on all 86 task commits | 73 skipped (commit predates the architecture model); 13 checked: `csfc check` passes on 13, `csfc check-generated` fails on 13, all with `CSF_GENERATED_DRIFT` only — the csfc used was built later than those commits (see above) |
| `mine --limit 2` end to end | both tasks valid; each records `csf` and `csf_guards`; on the newer commit both gates pass and are required of the agent, on the older one `check-generated` fails with drift and is not |

## RRSI in CSF's improvement vocabulary

CSF names the steps of an improvement loop in its generated ontology
(`csf/docs/generated/ontology_cgen.md` in the CSF module): observe, retrieve, choose,
check, execute, evaluate, save evidence, improve. This is how the RRSI pieces
in this fork line up with them, and what exists today.

| CSF step | RRSI piece | Status |
|---|---|---|
| Observe | `mine` reads git history, runs the tests before and after each fix and keeps the logs; csfc's diagnostics | exists |
| Retrieve | `csf detect` and the component map find the declared architecture and the components a fix touches; the fairness `api` stage lists the names the hidden tests need | exists |
| Choose | RRSI's proposer picks the next harness change | exists upstream; not yet run on the Go exam |
| Check | FAIL_TO_PASS validation, the fairness stages and the CSF guards admit a question before it is used; `csf_guard` checks an agent's patch against the required gates | admission exists; applying guards to agent patches needs the grader (planned) |
| Execute | run the agent harness on a task in a sealed container | planned (harness not wired to the Go exam) |
| Evaluate | grade by the hidden tests plus required guards; split health says whether a held-out score can mean anything | split health exists; grader planned |
| Save evidence | task directories (task.json, patches, logs, fairness verdicts, `csf`, `csf_guards`), `rejected.jsonl`, the report | exists |
| Improve | an RRSI round evolves the harness on the practice set and checks it on the held-out exam | planned: no baseline or round has been run |
