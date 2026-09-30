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

_CLASSES = """Every non-pass has a class, printed as FAILURE CLASS in the grading
block, and the classes need different mechanisms:
  NO-SUBMISSION: the loop ended without a submit. Plumbing or termination,
    never a coding problem.
  BROKEN-MODULE: the submitted code does not import, or lacks the function.
  WRONG-ANSWER: a hidden edge case failed. The specification usually states
    the case; ask whether the agent ever ran its code on the stated rules
    before submitting.
  TIMEOUT / RECURSION-LIMIT: the solution is too slow or too deep for a large
    input.
  HARNESS-CRASH: the harness raised; the whole trial is lost.
  INFRA: the policy endpoint failed; not evidence about the harness."""

ANALYST = f"""The agent is a small JSON-action loop with one sandboxed
`run_python` tool, driving a frozen policy LLM on short Python function tasks.
Each submission is graded by hidden unit tests; the reward is pass/fail.

{_CLASSES}

Count how often the agent submits without ever running its code, and how often
it fails to follow the JSON protocol: both are harness-level habits."""

DIGESTER = """The trajectory comes from a JSON-action coding agent solving a short
Python function task with a sandboxed `run_python` tool. Each <task_id>.txt
holds the task, every step, the submitted module, and a GRADING block with the
FAILURE CLASS and the hidden-test output tail. Read the GRADING block first,
then work backwards to where the failing behaviour was decided."""

PROPOSER = f"""The benchmark is a suite of short Python function tasks. Per task
the agent receives a natural-language specification and must submit a module
that defines one function; hidden unit tests grade it. The scaffold is a JSON
action loop: the policy replies with {{"action": "run_python" | "submit",
"code": ...}}; run_python executes a script in a no-network sandbox and returns
its exit code and output. The frozen policy may be a SMALL model: it often
breaks the JSON protocol (markdown fences, prose around the object), forgets to
submit, or submits untested code. Harness-level fixes that make the protocol
robust, force or encourage self-testing against the specification's stated
rules, and recover work from a run that never submitted are all legitimate.
`max_steps` is injected by the runner; editing any step budget has no effect.

{_CLASSES}"""

CRITIC = """Harness under review: a JSON-action coding agent solving short Python
function tasks graded by hidden unit tests. Reject any edit that names a
task id, a function name from the suite, or any value, test case or expected
output that could only come from a specific task or its hidden tests; any
attempt to read the task data, reference solutions or grader; and any
network access. General coding practice (write tests from the specification,
check edge cases the specification lists, avoid recursion on deep input) is
fine."""
