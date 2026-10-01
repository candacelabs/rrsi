# Pattern library (simulation instance)

Mechanisms that usually transfer across small controller-design agents. Each is
a starting point, not a prescription; cite trace evidence before using one.

- **Tolerant action parsing.** Accept a JSON object wrapped in markdown fences
  or surrounded by prose; extract the first balanced object.
- **Template instead of trees.** Let the policy choose a few integer gains and
  have the harness build the expression tree (e.g. steering = clamp(a*input0 +
  b*input1), acceleration = clamp(c*input2)); malformed trees disappear.
- **Sign conventions up front.** State that a car left of centre (input 0 > 0)
  or pointing left (input 1 > 0) needs negative steering, and that a car below
  target speed (input 2 > 0) needs positive acceleration.
- **Check, then roll out, then submit.** Run `check` on the draft, then
  `rollout`, and feed the surrogate's oracle verdicts and final state back
  before submitting; keep the best controller seen.
- **Submit fallback.** When the step budget runs out, submit the best admitted
  controller seen instead of nothing.
- **Margins over fit.** The graded simulator's actuators differ from the
  surrogate's; prefer moderate gains with damping (heading term) over gains
  that only just pass on the surrogate.
