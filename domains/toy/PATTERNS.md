# Pattern library (toy instance)

Mechanisms that usually transfer across small coding agents. Each is a
starting point, not a prescription; cite trace evidence before using one.

- **Tolerant action parsing.** Accept a JSON object wrapped in markdown
  fences or surrounded by prose; extract the first balanced object. Tell the
  model exactly what was wrong when parsing fails.
- **Submit fallback.** When the step budget runs out without a submit, submit
  the most recent code the agent wrote or ran instead of nothing.
- **Self-test before submit.** Ask the agent to run its function against the
  examples and every rule stated in the specification before submitting; on
  the first submit, optionally run the code once and return any failure.
- **Specification checklist.** Before coding, have the agent list the edge
  cases the specification states (empty input, invalid input, ties, bounds).
- **Robustness defaults.** Prefer iterative algorithms over deep recursion;
  do not mutate inputs unless asked.
- **Useful tool output.** Show the exit code and the tail of a traceback,
  not only its head.
