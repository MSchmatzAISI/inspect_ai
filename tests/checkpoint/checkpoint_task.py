"""Standalone eval tasks for end-to-end checkpoint testing.

Run from the CLI:
    inspect eval tests/checkpoint/checkpoint_task.py@checkpoint_test_task \
        --model mockllm/model --checkpoint --checkpoint-interval-seconds 0.1

    inspect eval tests/checkpoint/checkpoint_task.py@checkpoint_crash_task \
        --model mockllm/model --checkpoint --checkpoint-interval-seconds 0.1

See tests/checkpoint/run_e2e.sh for the full test script.
"""

import json
import os
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, TaskState, generate, solver, use_tools
from inspect_ai.tool import tool

COMPOSE_FILE = str(Path(__file__).parent / "compose.yaml")

NUM_TOOL_CALLS = 4
SAMPLES = [
    Sample(id="s1", input="Count to 4 using the write_counter tool", target="Done"),
    Sample(id="s2", input="Count to 4 using the write_counter tool", target="Done"),
    Sample(id="s3", input="Count to 4 using the write_counter tool", target="Done"),
]


@tool
def write_counter():
    """Write an incrementing counter to /tmp/counter.txt in the sandbox.

    Also records the counter value in state.store["counter"] so checkpoint
    serialisation captures it.
    """

    async def execute() -> str:
        """Write the next counter value."""
        from inspect_ai.util import sandbox, store

        sbx = sandbox()
        result = await sbx.exec(["cat", "/tmp/counter.txt"])
        current = int(result.stdout.strip()) if result.success else 0
        new_val = current + 1
        await sbx.exec(["bash", "-c", f"echo {new_val} > /tmp/counter.txt"])

        # Mirror the counter into state.store so checkpoint preserves it
        store().set("counter", new_val)

        return f"Counter is now {new_val}"

    return execute


def _make_outputs(num_tool_calls: int = NUM_TOOL_CALLS) -> list[ModelOutput]:
    """Build a sequence of mock model outputs for one sample."""
    outputs: list[ModelOutput] = []
    for _ in range(num_tool_calls):
        outputs.append(
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="write_counter",
                tool_arguments={},
            )
        )
    outputs.append(ModelOutput.from_content(model="mockllm/model", content="Done"))
    return outputs


def _make_mock_model(num_samples: int = 3) -> object:
    """Create a mock model pre-programmed with tool calls for all samples."""
    all_outputs: list[ModelOutput] = []
    for _ in range(num_samples):
        all_outputs.extend(_make_outputs())
    return get_model("mockllm/model", custom_outputs=all_outputs)


@task
def checkpoint_test_task() -> Task:
    """Happy-path task: runs 3 samples to completion with checkpoints."""
    return Task(
        dataset=list(SAMPLES),
        solver=[use_tools([write_counter()]), generate()],
        scorer=includes(),
        sandbox=("docker", COMPOSE_FILE),
        model=_make_mock_model(),
        message_limit=20,
    )


@solver
def crash_after_tool_calls(
    crash_sample_id: str = "s2",
    crash_after: int = 2,
) -> None:
    """Solver that crashes on a specific sample, with resume verification.

    First run (CHECKPOINT_TEST_PASS unset):
        Runs generate(), then raises ValueError on the crash sample after
        enough tool calls — simulating a mid-eval crash.

    Resume run (CHECKPOINT_TEST_PASS=1):
        Before calling generate(), verifies that checkpoint restore worked by
        checking state.store["counter"] and reading /tmp/counter.txt from the
        sandbox. Writes verification results as JSON to the path given by
        CHECKPOINT_VERIFY_FILE so the test script can inspect them.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        is_resume = os.environ.get("CHECKPOINT_TEST_PASS", "") == "1"

        # On a resumed run for the crash sample, verify restored state
        if is_resume and state.sample_id == crash_sample_id:
            from inspect_ai.util import sandbox

            verify: dict[str, object] = {"sample_id": state.sample_id}

            # Check state.store — should have counter from pre-crash run
            store_counter = state.store.get("counter")
            verify["store_counter"] = store_counter
            verify["store_restored"] = store_counter is not None and store_counter > 0

            # Check messages — should have pre-crash conversation history
            verify["message_count"] = len(state.messages)
            verify["messages_restored"] = len(state.messages) > 1

            # Check sandbox filesystem — CRIU should have restored the
            # container with /tmp/counter.txt containing the pre-crash value.
            # Note: Docker 29.x may not support --checkpoint-dir on restore,
            # so CRIU container restore may fail. The sandbox counter check
            # is reported but not required for the test to pass.
            sbx = sandbox()
            result = await sbx.exec(["cat", "/tmp/counter.txt"])
            sandbox_counter: int | None = None
            if result.success:
                try:
                    sandbox_counter = int(result.stdout.strip())
                except ValueError:
                    pass
            verify["sandbox_counter"] = sandbox_counter
            verify["sandbox_restored"] = (
                sandbox_counter is not None and sandbox_counter > 0
            )

            # Store and sandbox counters should agree if CRIU restore worked
            verify["counters_match"] = store_counter == sandbox_counter

            verify_file = os.environ.get("CHECKPOINT_VERIFY_FILE", "")
            if verify_file:
                with open(verify_file, "w") as f:
                    json.dump(verify, f, indent=2)

        # Run the normal generate loop
        state = await generate(state)

        # On first run, crash the target sample
        if not is_resume and state.sample_id == crash_sample_id:
            tool_call_count = sum(
                1
                for m in state.messages
                if m.role == "assistant" and getattr(m, "tool_calls", None)
            )
            if tool_call_count >= crash_after:
                raise ValueError(
                    f"Simulated crash on sample {crash_sample_id} "
                    f"after {tool_call_count} tool calls"
                )

        return state

    return solve


@task
def checkpoint_crash_task() -> Task:
    """Crash-path task: one sample crashes mid-eval for resume testing."""
    return Task(
        dataset=list(SAMPLES),
        solver=[use_tools([write_counter()]), crash_after_tool_calls()],
        scorer=includes(),
        sandbox=("docker", COMPOSE_FILE),
        model=_make_mock_model(),
        message_limit=20,
        fail_on_error=True,
    )
