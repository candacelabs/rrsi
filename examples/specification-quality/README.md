# How well is our work specified? A first measurement

*Candace Labs, 30 September 2026. Measured on the Candace Labs Go monorepo with
the `rrsi-mine` fairness stages in this fork. The repository is private, so the
examples below are paraphrased and carry no commit IDs or code; every number is
measured.*

## The short version

**Before:** whether a change was "well specified" was a matter of opinion.
Nobody could say how often the written record of a change (its description and
its tests) was enough for someone else to rebuild it without guessing.

**Now:** a tool rebuilds each past change from its written record and asks an
independent model to attempt it from that record alone. On 70 recent changes,
**only 25 (36%) were fully specified**. The other 45 each came with a concrete
reason, such as "the spec never says how long the shutdown grace period is".

That makes the tool a measure of how well we write down what we want. Most of
these changes were made by AI agents working from our prompts, so it is a
first, indirect measure of how well we prompt.

## What was measured

| Step | What happens | Changes |
|---|---|---|
| 1. Collect | Recent commits (since 2026-06-01) that change Go code and that code's tests, at most 400 lines and 3 packages | 86 |
| 2. Keep only real questions | The commit's tests must **fail** before the change and **pass** after it, run offline in a sealed container | 70 |
| 3. Write the spec | A model writes a problem statement from the commit message and the tests. It never sees the code change itself, and a check rejects any statement that copies the change. | 66 written |
| 4. Try to use the spec | A *different* model reads only that statement and answers: could I build this without guessing any behaviour or name? | 31 passed |
| 5. Check the tests | Do the tests pin details that the statement never mentions (exact messages, call counts)? | see below |
| 6. Verdict | A change is **fully specified** only if it passes steps 2–5 | **25 of 70** |

## Scorecard

Unit: changes (one change = one commit). A change can fail more than one
check, so the failure rows add up to more than 45.

| Verdict | Changes | Share of 70 | What it means |
|---|---|---|---|
| Fully specified | **25** | 36% | The description and tests together pin the behaviour; someone else could rebuild the change from them. |
| A reader must guess behaviour or names | **39** | 56% | The record leaves out something the change actually depends on. |
| Tests check details nobody wrote down | **18**, about 14 after a known false positive (see caveats) | 26% (about 20%) | The tests would fail a correct-looking rebuild over a detail no spec mentions. |
| Cannot be described without copying the change | **4** | 6% | The change is its own specification, e.g. adding small typed helper functions. |

Separately, **16 of the 86 collected commits never became questions**: in 14 the
tests already passed *before* the change, and 2 did not build at their own
commit.

## What "underspecified" looked like

Each example is paraphrased from a real rejected change. The **habit** column is
the one-line prompting fix that would have prevented it.

| What the record said | What a reader had to guess | Habit that fixes it |
|---|---|---|
| "On shutdown, let in-flight requests finish within a bounded grace period." | How long the grace period is. | State every limit as a number: "wait up to 5 s, then cancel." |
| "Reject a vote request with an empty candidate ID." (tests check the exact error text) | The exact error message, for five different rejections. | If a test checks an exact message, put that message in the prompt. |
| "Settings can come from the JSON file or from a code option." | Which one wins when both are set. | For any two sources of the same value, say which takes precedence. |
| "Send the readiness update only to the session that reported it." | How a session is identified as "the one that reported". | Name the identifier: "the session whose ID matches the report's session ID." |
| "Keep existing behaviour for errors that do not say whether to retry." | Whether errors without a retry flag are retried; one generated statement got this backwards. | Spell out the default case, not only the new case. |
| A new test checks that an ordered list of requests equals exactly `[old, abort, new]` | That an ordering guarantee was required at all. | When order matters, say "in this order". |

And what "fully specified" looked like, in a passing example: *"Add
`FormatAgo(elapsed time.Duration) string`. Under a minute renders `just now`;
under an hour `<N>m ago`, truncated not rounded; under a day `<N>h ago`; else
`<N>d ago`. Negative durations count as zero."* A reader needs nothing else.

## What this does and does not show

- **It scores the recorded intent, not the original prompts.** Step 3 rebuilt
  each specification from the commit message and tests. Scoring the actual
  prompts (the pull request text, the issue, or the first message of the agent
  session) is the next step and is not wired up yet.
- **Step 4 is strict.** It fails a change for *any* ambiguity, including
  details the hidden tests never check, because it is not allowed to see the
  tests. The "must guess" count is therefore an upper bound. The planned fix
  sends the reviewer's questions back to the writer, which *can* see the tests,
  and lets it resolve only the points the tests depend on.
- **Step 5 has a known false positive.** It also flags Gomega's optional
  failure-description text (e.g. `Expect(x).To(Equal(y), "the archive must be
  byte deterministic")`) as if it were a checked value. About 4 of the 18 flags
  come only from such descriptions.
- **Step 3's copy check cannot yet show its evidence.** 3 of its 4 rejections
  shared short phrases with the change but no whole line. They may be ordinary
  shared wording.
- **Reviewer verdicts vary between runs.** One unchanged change passed on one
  run and failed on another. Treat any single number as ±a few changes.
- **One repository, about three months, 70 changes.** This is a first
  measurement, not a benchmark.

## Reproduce it

```bash
cargo build --release --manifest-path tools/rrsi-mine/Cargo.toml
B=tools/rrsi-mine/target/release/rrsi-mine
$B mine --repo /path/to/go/repo --out /private/tasks --since 2026-06-01 --jobs 8
$B fairness --tasks /private/tasks --repo /path/to/go/repo --jobs 4
cargo run --manifest-path tools/rrsi-mine/Cargo.toml --bin rrsi-report -- --tasks /private/tasks --repo /path/to/go/repo --out /private/report.html
```

`--out` must be outside every git work tree, which the tool enforces, because
the mined tasks contain the repository's source. The model roles run through a
logged-in Copilot or Codex CLI; no API key is needed.
