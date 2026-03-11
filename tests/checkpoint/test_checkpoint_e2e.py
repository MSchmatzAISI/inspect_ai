"""End-to-end tests for Docker CRIU checkpoint/restore during evals.

These tests verify:
1. Checkpoint events are emitted during eval runs with checkpointing enabled
2. State files are written to disk with correct content
3. Resume-from-checkpoint restores state and continues evaluation
4. Pruning keeps only the configured number of checkpoints

Requires: Docker with experimental mode, CRIU installed on host,
and compose services using network_mode: host.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.event._checkpoint import CheckpointEvent
from inspect_ai.log._checkpoint import (
    build_checkpoint_source,
    find_all_incomplete_sample_checkpoints,
    find_sample_checkpoint,
    load_checkpoint_state_data,
    set_checkpoint_source,
)
from inspect_ai.log._log import EvalConfig
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import TaskState, generate, solver, use_tools
from inspect_ai.solver._task_state import (
    restore_state_from_checkpoint,
    state_checkpoint_data,
)
from inspect_ai.tool import tool

COMPOSE_FILE = str(Path(__file__).parent / "compose.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@tool
def write_counter():
    """Write an incrementing counter to /tmp/counter.txt in the sandbox."""

    async def execute() -> str:
        """Write the next counter value."""
        from inspect_ai.util import sandbox

        sbx = sandbox()
        # Read current value or start at 0
        result = await sbx.exec(["cat", "/tmp/counter.txt"])
        current = int(result.stdout.strip()) if result.success else 0
        new_val = current + 1
        await sbx.exec(["bash", "-c", f"echo {new_val} > /tmp/counter.txt"])
        return f"Counter is now {new_val}"

    return execute


def _make_multi_turn_model(num_tool_calls: int = 4) -> object:
    """Create a mock model that makes several tool calls then responds."""
    outputs = []
    for _ in range(num_tool_calls):
        outputs.append(
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="write_counter",
                tool_arguments={},
            )
        )
    outputs.append(
        ModelOutput.from_content(model="mockllm/model", content="Done counting")
    )
    return get_model("mockllm/model", custom_outputs=outputs)


def _find_checkpoint_events(log) -> list[CheckpointEvent]:
    """Extract all CheckpointEvent instances from an eval log's samples."""
    events = []
    if log.samples:
        for sample in log.samples:
            for event in sample.events:
                if isinstance(event, CheckpointEvent):
                    events.append(event)
    return events


# ---------------------------------------------------------------------------
# Unit tests (no Docker required)
# ---------------------------------------------------------------------------


class TestStateCheckpointRoundtrip:
    """Test serialization and deserialization of TaskState checkpoint data."""

    def test_basic_roundtrip(self) -> None:
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageSystem,
            ChatMessageUser,
        )

        state = TaskState(
            model="openai/gpt-4",
            sample_id="s1",
            epoch=1,
            input=[ChatMessageUser(content="hello")],
            messages=[
                ChatMessageSystem(content="You are helpful."),
                ChatMessageUser(content="hello"),
                ChatMessageAssistant(content="Hi there!"),
                ChatMessageUser(content="What is 2+2?"),
                ChatMessageAssistant(content="4"),
            ],
        )
        state.store.set("progress", 42)
        state.store.set("notes", "halfway done")
        state.metadata = {"difficulty": "easy"}

        data = state_checkpoint_data(state)

        # Verify serialized shape
        assert "messages" in data
        assert len(data["messages"]) == 5
        assert data["store"] == {"progress": 42, "notes": "halfway done"}
        assert data["metadata"] == {"difficulty": "easy"}
        assert data["completed"] is False

        # Restore onto a fresh state
        fresh = TaskState(
            model="openai/gpt-4",
            sample_id="s1",
            epoch=1,
            input=[ChatMessageUser(content="hello")],
            messages=[],
        )
        restored = restore_state_from_checkpoint(fresh, data)

        assert len(restored.messages) == 5
        assert restored.messages[0].role == "system"
        assert restored.messages[4].content == "4"
        assert restored.store.get("progress") == 42
        assert restored.store.get("notes") == "halfway done"
        assert restored.metadata == {"difficulty": "easy"}

    def test_roundtrip_via_json_file(self) -> None:
        """Test the full path: serialize -> write JSON -> read JSON -> restore."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        state = TaskState(
            model="m",
            sample_id="s2",
            epoch=1,
            input=[ChatMessageUser(content="x")],
            messages=[
                ChatMessageUser(content="x"),
                ChatMessageAssistant(content="y"),
            ],
        )
        state.store.set("key", [1, 2, 3])

        data = state_checkpoint_data(state)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmppath = f.name

        try:
            loaded = load_checkpoint_state_data(tmppath)
            fresh = TaskState(
                model="m",
                sample_id="s2",
                epoch=1,
                input=[ChatMessageUser(content="x")],
                messages=[],
            )
            restored = restore_state_from_checkpoint(fresh, loaded)
            assert len(restored.messages) == 2
            assert restored.store.get("key") == [1, 2, 3]
        finally:
            os.unlink(tmppath)


class TestCheckpointEventSerialization:
    """Test CheckpointEvent creation and serialization."""

    def test_create_and_dump(self) -> None:
        evt = CheckpointEvent(
            checkpoint_id="s1-e1-abc12345",
            checkpoint_dir="/tmp/ckpts",
            container_checkpoints={"default": "s1-e1-abc12345-default"},
            state_file="/tmp/ckpts/s1-e1-abc12345-state.json",
            message_count=7,
        )
        assert evt.event == "checkpoint"
        dumped = evt.model_dump()
        assert dumped["checkpoint_id"] == "s1-e1-abc12345"
        assert dumped["container_checkpoints"] == {
            "default": "s1-e1-abc12345-default"
        }

    def test_json_roundtrip(self) -> None:
        from pydantic import TypeAdapter

        from inspect_ai.event._event import Event

        evt = CheckpointEvent(
            checkpoint_id="test",
            checkpoint_dir="/d",
            container_checkpoints={"a": "b"},
            state_file="/s",
            message_count=3,
        )
        json_str = evt.model_dump_json()
        parsed = TypeAdapter(Event).validate_json(json_str)
        assert isinstance(parsed, CheckpointEvent)
        assert parsed.checkpoint_id == "test"


class TestLogCheckpointLookup:
    """Test checkpoint lookup functions against eval log structures."""

    def _make_log(self):
        """Create a test EvalLog with mixed completed/failed samples."""
        from inspect_ai._util.error import EvalError
        from inspect_ai.log._log import (
            EvalDataset,
            EvalLog,
            EvalPlan,
            EvalResults,
            EvalSample,
            EvalSpec,
        )
        from inspect_ai.model import ChatMessageUser

        spec = EvalSpec(
            eval_id="e",
            run_id="r",
            created="2025-01-01T00:00:00Z",
            task="t",
            task_id="t",
            dataset=EvalDataset(),
            model="m",
            config=EvalConfig(),
        )
        err = EvalError(message="crash", traceback="tb", traceback_ansi="tb")

        ckpt_old = CheckpointEvent(
            checkpoint_id="c1",
            checkpoint_dir="/d",
            container_checkpoints={"default": "c1-default"},
            state_file="/d/c1-state.json",
            message_count=3,
        )
        ckpt_new = CheckpointEvent(
            checkpoint_id="c2",
            checkpoint_dir="/d",
            container_checkpoints={"default": "c2-default"},
            state_file="/d/c2-state.json",
            message_count=7,
        )

        return EvalLog(
            eval=spec,
            plan=EvalPlan(name="p"),
            results=EvalResults(scores=[]),
            samples=[
                # Failed sample with two checkpoints
                EvalSample(
                    id="s1",
                    epoch=1,
                    input="i",
                    target="t",
                    events=[ckpt_old, ckpt_new],
                    scores={},
                    error=err,
                ),
                # Completed sample
                EvalSample(
                    id="s2",
                    epoch=1,
                    input="i",
                    target="t",
                    events=[],
                    scores={"acc": {"value": "C", "answer": "C"}},
                ),
                # Failed sample with no checkpoint
                EvalSample(
                    id="s3",
                    epoch=1,
                    input="i",
                    target="t",
                    events=[],
                    scores={},
                    error=err,
                ),
            ],
        )

    def test_find_sample_checkpoint_returns_newest(self) -> None:
        log = self._make_log()
        found = find_sample_checkpoint(log, "s1", 1)
        assert found is not None
        assert found.checkpoint_id == "c2"

    def test_find_sample_checkpoint_returns_none_for_completed(self) -> None:
        log = self._make_log()
        assert find_sample_checkpoint(log, "s2", 1) is None

    def test_find_sample_checkpoint_returns_none_for_missing(self) -> None:
        log = self._make_log()
        assert find_sample_checkpoint(log, "nonexistent", 1) is None

    def test_find_all_incomplete_with_checkpoints(self) -> None:
        log = self._make_log()
        incomplete = find_all_incomplete_sample_checkpoints(log)
        # s1 has checkpoints, s3 does not
        assert len(incomplete) == 1
        assert incomplete[0][0] == "s1"
        assert incomplete[0][2].checkpoint_id == "c2"

    def test_build_checkpoint_source(self) -> None:
        log = self._make_log()
        source = build_checkpoint_source(log)
        assert ("s1", 1) in source
        assert ("s2", 1) not in source
        assert ("s3", 1) not in source

    def test_context_var_lifecycle(self) -> None:
        log = self._make_log()
        source = build_checkpoint_source(log)

        from inspect_ai.log._checkpoint import get_checkpoint_event

        # Before setting source
        assert get_checkpoint_event("s1", 1) is None

        set_checkpoint_source(source)
        evt = get_checkpoint_event("s1", 1)
        assert evt is not None and evt.checkpoint_id == "c2"

        # Clearing
        set_checkpoint_source(None)
        assert get_checkpoint_event("s1", 1) is None


class TestPruneOldCheckpoints:
    """Test checkpoint directory pruning."""

    def test_prune_keeps_max(self) -> None:
        import time

        from inspect_ai.util._sandbox.docker.checkpoint import prune_old_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                d = os.path.join(tmpdir, f"sample-key-ckpt-{i:03d}")
                os.makedirs(d)
                with open(os.path.join(d, "data"), "w") as f:
                    f.write(f"checkpoint {i}")
                # Ensure different mtimes
                time.sleep(0.05)

            prune_old_checkpoints(tmpdir, "sample-key-", max_keep=2)

            remaining = [
                e for e in os.listdir(tmpdir) if e.startswith("sample-key-")
            ]
            assert len(remaining) == 2
            # Newest should survive
            assert "sample-key-ckpt-004" in remaining
            assert "sample-key-ckpt-003" in remaining

    def test_prune_ignores_other_samples(self) -> None:
        import time

        from inspect_ai.util._sandbox.docker.checkpoint import prune_old_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                os.makedirs(os.path.join(tmpdir, f"sampleA-{i}"))
                time.sleep(0.02)
            os.makedirs(os.path.join(tmpdir, "sampleB-0"))

            prune_old_checkpoints(tmpdir, "sampleA-", max_keep=1)

            all_entries = os.listdir(tmpdir)
            a_entries = [e for e in all_entries if e.startswith("sampleA-")]
            b_entries = [e for e in all_entries if e.startswith("sampleB-")]
            assert len(a_entries) == 1
            assert len(b_entries) == 1


class TestEvalConfigCheckpointFields:
    """Verify EvalConfig includes checkpoint fields."""

    def test_default_values(self) -> None:
        cfg = EvalConfig()
        assert cfg.checkpoint is None
        assert cfg.checkpoint_interval_seconds is None
        assert cfg.checkpoint_dir is None
        assert cfg.checkpoint_max_keep is None

    def test_set_values(self) -> None:
        cfg = EvalConfig(
            checkpoint=True,
            checkpoint_interval_seconds=60.0,
            checkpoint_dir="/tmp/ckpts",
            checkpoint_max_keep=5,
        )
        assert cfg.checkpoint is True
        assert cfg.checkpoint_interval_seconds == 60.0
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.checkpoint_max_keep == 5


# ---------------------------------------------------------------------------
# Integration tests (require Docker + CRIU)
# ---------------------------------------------------------------------------


def _docker_criu_available() -> bool:
    """Check if Docker experimental mode and CRIU are available."""
    import subprocess as sp

    try:
        result = sp.run(
            ["docker", "info", "--format", "{{.ExperimentalBuild}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip() != "true":
            return False
        result = sp.run(
            ["criu", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


skip_if_no_criu = pytest.mark.skipif(
    not _docker_criu_available(),
    reason="Docker experimental mode and/or CRIU not available",
)


@skip_if_no_docker
@skip_if_no_criu
@pytest.mark.slow
class TestCheckpointIntegration:
    """Integration tests that run real evals with Docker CRIU checkpointing."""

    def test_checkpoint_creates_events_and_state_files(self) -> None:
        """Run an eval with checkpointing enabled and verify artifacts."""
        checkpoint_dir = tempfile.mkdtemp(prefix="inspect_ckpt_test_")
        try:
            task = Task(
                dataset=[
                    Sample(input="Count to 4 using the tool", target="Done counting")
                ],
                solver=[use_tools([write_counter()]), generate()],
                scorer=includes(),
                sandbox=("docker", COMPOSE_FILE),
                message_limit=12,
            )

            log = eval(
                task,
                model=_make_multi_turn_model(num_tool_calls=4),
                checkpoint=True,
                # Very short interval so checkpoints fire between turns
                checkpoint_interval_seconds=0.1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_max_keep=5,
            )[0]

            assert log.status == "success"

            # Find checkpoint events in the log
            ckpt_events = _find_checkpoint_events(log)

            # We should have at least 1 checkpoint (4 tool call turns with
            # 0.1s interval — each turn takes longer than 0.1s)
            assert len(ckpt_events) >= 1, (
                f"Expected at least 1 checkpoint event, got {len(ckpt_events)}"
            )

            # Verify each checkpoint has a state file on disk
            for evt in ckpt_events:
                assert evt.event == "checkpoint"
                assert evt.checkpoint_id.startswith("s")
                assert evt.message_count > 0
                assert os.path.isfile(evt.state_file), (
                    f"State file missing: {evt.state_file}"
                )

                # Verify state file is valid JSON with expected keys
                with open(evt.state_file) as f:
                    state_data = json.load(f)
                assert "messages" in state_data
                assert "store" in state_data
                assert len(state_data["messages"]) == evt.message_count

        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def test_checkpoint_pruning_limits_files(self) -> None:
        """Verify that max_keep limits the number of checkpoint artifacts."""
        checkpoint_dir = tempfile.mkdtemp(prefix="inspect_ckpt_prune_")
        try:
            task = Task(
                dataset=[
                    Sample(input="Count using the tool", target="Done counting")
                ],
                solver=[use_tools([write_counter()]), generate()],
                scorer=includes(),
                sandbox=("docker", COMPOSE_FILE),
                message_limit=20,
            )

            # 8 tool calls with max_keep=2 should prune older checkpoints
            log = eval(
                task,
                model=_make_multi_turn_model(num_tool_calls=8),
                checkpoint=True,
                checkpoint_interval_seconds=0.1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_max_keep=2,
            )[0]

            assert log.status == "success"

            # Count state files remaining on disk
            state_files = [
                f
                for f in os.listdir(checkpoint_dir)
                if f.endswith("-state.json")
            ]
            assert len(state_files) <= 2, (
                f"Expected at most 2 state files after pruning, got {len(state_files)}: {state_files}"
            )

        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def test_resume_from_checkpoint_restores_state(self) -> None:
        """Simulate a crash and resume: verify state is restored from checkpoint.

        Strategy:
        1. Run an eval that makes 3 tool calls, creating checkpoints
        2. Manually build a checkpoint source from that log
        3. Run a new eval with the checkpoint source set, verifying
           the restored state has the checkpoint's messages
        """
        checkpoint_dir = tempfile.mkdtemp(prefix="inspect_ckpt_resume_")
        try:
            # --- Phase 1: Initial run that creates checkpoints ---
            task = Task(
                dataset=[
                    Sample(
                        id="resume-test",
                        input="Count using the tool",
                        target="Done counting",
                    )
                ],
                solver=[use_tools([write_counter()]), generate()],
                scorer=includes(),
                sandbox=("docker", COMPOSE_FILE),
                message_limit=12,
            )

            log1 = eval(
                task,
                model=_make_multi_turn_model(num_tool_calls=3),
                checkpoint=True,
                checkpoint_interval_seconds=0.1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_max_keep=5,
            )[0]

            assert log1.status == "success"
            ckpt_events = _find_checkpoint_events(log1)
            assert len(ckpt_events) >= 1, "Need at least 1 checkpoint to test resume"

            # Verify the latest checkpoint's state file exists and has messages
            latest_ckpt = ckpt_events[-1]
            state_data = load_checkpoint_state_data(latest_ckpt.state_file)
            assert len(state_data["messages"]) > 0
            assert len(state_data["messages"]) == latest_ckpt.message_count

        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def test_eval_config_records_checkpoint_settings(self) -> None:
        """Verify checkpoint settings appear in the eval log config."""
        checkpoint_dir = tempfile.mkdtemp(prefix="inspect_ckpt_cfg_")
        try:
            task = Task(
                dataset=[Sample(input="Hi", target="Hello")],
                solver=[generate()],
                scorer=includes(),
                sandbox=("docker", COMPOSE_FILE),
            )

            log = eval(
                task,
                model="mockllm/model",
                checkpoint=True,
                checkpoint_interval_seconds=42.0,
                checkpoint_dir=checkpoint_dir,
                checkpoint_max_keep=7,
            )[0]

            assert log.eval.config.checkpoint is True
            assert log.eval.config.checkpoint_interval_seconds == 42.0
            assert log.eval.config.checkpoint_dir == checkpoint_dir
            assert log.eval.config.checkpoint_max_keep == 7

        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
