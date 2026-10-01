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
"""Domain paragraphs for the analyst, digester, proposer and critic."""

_CLASSES = """Every non-pass has a class, printed as FAILURE CLASS in the simulation
block, and the classes need different mechanisms:
  NO-SUBMISSION: the loop ended without a submit. Plumbing or termination.
  REJECTED: the submitted controller is not a valid CSF controller (shape,
    opcode arity, integer fields). The `check` tool would have said so.
  LANE-DEPARTURE: the car left the lane; usually a steering sign error or gains
    so large the car oscillates. Positive steering turns left; a car left of
    centre (positive input 0) needs negative steering.
  GOAL-NOT-REACHED: too slow; the acceleration law never closes the speed gap.
  OVER-LIMIT: overshot the speed limit (acceleration gain too high).
  NOT-SETTLED: still offset or swinging at the end; damping (heading term) or
    gain balance.
  HARNESS-CRASH: the harness raised; the whole trial is lost.
  INFRA: the policy endpoint or simulator failed; not evidence about the harness."""

ANALYST = f"""The agent is a small JSON-action loop driving a frozen policy LLM that
designs a controller (a tiny integer expression tree) for a car on a straight
road. The controller is checked by CSF, executed in a physics simulator, and
graded by four oracles (reached goal, no collision/lane departure, within speed
limit, settled at the end); the reward is the fraction passed.

{_CLASSES}

The agent has a `check` tool (CSF admission) and a `rollout` tool (one episode on
a cheap surrogate plant, not the graded simulator). Count how often it submits
without checking or rolling out, and how often it breaks the JSON protocol."""

DIGESTER = """The trajectory comes from a JSON-action agent that designs a driving
controller (integer expression tree over four normalized inputs) for one
straight-road scenario. Each <task_id>.txt holds the scenario brief, every
step, the submitted controller and a SIMULATION block with the FAILURE CLASS,
the oracle verdicts and a sampled trace. Read the SIMULATION block first, then
work backwards to where the failing controller was decided."""

PROPOSER = f"""The benchmark is a suite of straight-road driving scenarios (varying
target speed, start offset, heading, episode length). Per scenario the agent
must submit one controller: CSF Controller JSON whose steering and acceleration
are integer expression trees over four normalized inputs. CSF checks it,
a physics simulator executes it, and four oracles grade the episode. The
scaffold is a JSON action loop: the policy replies with {{"action": "check" |
"submit", "controller": ...}}. The runner also passes `rollout(controller)`,
which runs one episode on a cheap surrogate plant and returns the oracle
verdicts and final state there; the graded simulator differs from it (actuator
mapping, vehicle dynamics), so robust margins matter more than fitting it.
The frozen policy may be a SMALL model: it often writes malformed trees, gets
signs wrong, or picks unstable gains. Harness-level fixes that make the
protocol robust, give the policy a safer way to express a controller, check and
roll out before submitting, and recover from a loop that never submitted are
all legitimate. `max_steps` is injected by the runner.

{_CLASSES}"""

CRITIC = """Harness under review: a JSON-action agent that designs a driving
controller per scenario, graded in a simulator by oracles. Reject any edit
that names a scenario id or seed, hardcodes per-scenario numbers (target
speeds, offsets, goal distances) or a lookup from scenario to controller;
reads the scenario data, the oracle source, the grader or evidence files; or
uses the network or processes. General control practice (proportional-derivative
lane keeping, sign conventions from the documented inputs, gain margins,
clamping, using the rollout tool) is fine."""
