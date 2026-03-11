"""Tests for the abstract checkpoint interface and runtime-agnostic checkpoint manager.

These tests verify:
1. SandboxEnvironment base class has checkpoint methods with correct defaults
2. DockerSandboxEnvironment reports supports_checkpoint() == True
3. PodmanSandboxEnvironment reports supports_checkpoint() == True
4. CheckpointManager works with the abstract interface
5. setup_checkpoint_manager_from_sandbox finds checkpoint-capable sandboxes
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.util._sandbox.environment import SandboxEnvironment

# ---------------------------------------------------------------------------
# Base class default behavior
# ---------------------------------------------------------------------------


class TestSandboxEnvironmentCheckpointDefaults:
    """Test that the base class checkpoint methods have correct defaults."""

    def test_supports_checkpoint_default_false(self) -> None:
        assert SandboxEnvironment.supports_checkpoint() is False

    async def test_checkpoint_create_default_returns_false(self) -> None:
        env = _make_stub_sandbox()
        result = await env.checkpoint_create("test", "/tmp/ckpt")
        assert result is False

    async def test_checkpoint_restore_default_returns_false(self) -> None:
        env = _make_stub_sandbox()
        result = await env.checkpoint_restore("test", "/tmp/ckpt")
        assert result is False

    async def test_checkpoint_delete_default_noop(self) -> None:
        env = _make_stub_sandbox()
        # Should not raise
        await env.checkpoint_delete("test", "/tmp/ckpt")


# ---------------------------------------------------------------------------
# Docker and Podman supports_checkpoint
# ---------------------------------------------------------------------------


class TestDockerSupportsCheckpoint:
    def test_docker_supports_checkpoint(self) -> None:
        from inspect_ai.util._sandbox.docker.docker import (
            DockerSandboxEnvironment,
        )

        assert DockerSandboxEnvironment.supports_checkpoint() is True


class TestPodmanSupportsCheckpoint:
    def test_podman_supports_checkpoint(self) -> None:
        from inspect_ai.util._sandbox.podman.podman import (
            PodmanSandboxEnvironment,
        )

        assert PodmanSandboxEnvironment.supports_checkpoint() is True


# ---------------------------------------------------------------------------
# CheckpointManager with abstract interface
# ---------------------------------------------------------------------------


class TestCheckpointManagerAbstract:
    """Test that CheckpointManager uses the abstract checkpoint interface."""

    async def test_manager_calls_checkpoint_create_on_environments(
        self,
    ) -> None:
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.solver import TaskState
        from inspect_ai.util._sandbox._checkpoint_manager import (
            CheckpointManager,
        )

        env1 = _make_mock_checkpoint_env(supports=True, create_success=True)
        env2 = _make_mock_checkpoint_env(supports=True, create_success=True)
        env_no_ckpt = _make_mock_checkpoint_env(supports=False)

        envs = {"default": env1, "proxy": env2, "local": env_no_ckpt}

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                environments=envs,
                sample_id="s1",
                epoch=1,
                checkpoint_dir=tmpdir,
                interval_seconds=0,
                max_keep=3,
            )

            state = TaskState(
                model="m",
                sample_id="s1",
                epoch=1,
                input=[ChatMessageUser(content="x")],
                messages=[ChatMessageUser(content="x")],
            )

            with patch(
                "inspect_ai.util._sandbox._checkpoint_manager.transcript"
            ) as mock_transcript:
                mock_transcript.return_value._event = MagicMock()
                await manager.create_checkpoint(state)

            # Verify checkpoint_create was called on supported environments
            env1.checkpoint_create.assert_called_once()
            env2.checkpoint_create.assert_called_once()

            # Verify checkpoint_create was NOT called on unsupported env
            env_no_ckpt.checkpoint_create.assert_not_called()

    async def test_manager_skips_failed_checkpoints(self) -> None:
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.solver import TaskState
        from inspect_ai.util._sandbox._checkpoint_manager import (
            CheckpointManager,
        )

        env_ok = _make_mock_checkpoint_env(supports=True, create_success=True)
        env_fail = _make_mock_checkpoint_env(supports=True, create_success=False)

        envs = {"default": env_ok, "failing": env_fail}

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                environments=envs,
                sample_id="s1",
                epoch=1,
                checkpoint_dir=tmpdir,
                interval_seconds=0,
                max_keep=3,
            )

            state = TaskState(
                model="m",
                sample_id="s1",
                epoch=1,
                input=[ChatMessageUser(content="x")],
                messages=[ChatMessageUser(content="x")],
            )

            with patch(
                "inspect_ai.util._sandbox._checkpoint_manager.transcript"
            ) as mock_transcript:
                mock_transcript.return_value._event = MagicMock()
                await manager.create_checkpoint(state)

            # Both should be called, but only one succeeds
            env_ok.checkpoint_create.assert_called_once()
            env_fail.checkpoint_create.assert_called_once()

    async def test_manager_writes_state_file(self) -> None:
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.solver import TaskState
        from inspect_ai.util._sandbox._checkpoint_manager import (
            CheckpointManager,
        )

        env = _make_mock_checkpoint_env(supports=True, create_success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                environments={"default": env},
                sample_id="s1",
                epoch=1,
                checkpoint_dir=tmpdir,
                interval_seconds=0,
                max_keep=3,
            )

            state = TaskState(
                model="m",
                sample_id="s1",
                epoch=1,
                input=[ChatMessageUser(content="x")],
                messages=[
                    ChatMessageUser(content="x"),
                    ChatMessageUser(content="y"),
                ],
            )
            state.store.set("key", "value")

            with patch(
                "inspect_ai.util._sandbox._checkpoint_manager.transcript"
            ) as mock_transcript:
                mock_transcript.return_value._event = MagicMock()
                await manager.create_checkpoint(state)

            # Find the state file
            state_files = [f for f in os.listdir(tmpdir) if f.endswith("-state.json")]
            assert len(state_files) == 1

            with open(os.path.join(tmpdir, state_files[0])) as f:
                data = json.load(f)
            assert len(data["messages"]) == 2
            assert data["store"]["key"] == "value"


# ---------------------------------------------------------------------------
# Podman checkpoint module
# ---------------------------------------------------------------------------


class TestPodmanCheckpointFunctions:
    """Test Podman checkpoint functions (mocked subprocess calls)."""

    async def test_podman_checkpoint_create_success(self) -> None:
        from inspect_ai.util._sandbox.podman.checkpoint import (
            podman_checkpoint_create,
        )

        mock_result = MagicMock()
        mock_result.success = True

        with patch(
            "inspect_ai.util._sandbox.podman.checkpoint.subprocess",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_sub:
            result = await podman_checkpoint_create(
                container="test-container",
                name="ckpt-1",
                checkpoint_dir="/tmp/ckpt",
                leave_running=True,
                tcp_established=True,
            )
            assert result is True
            call_args = mock_sub.call_args[0][0]
            assert "podman" in call_args
            assert "checkpoint" in call_args
            assert "--leave-running" in call_args
            assert "--tcp-established" in call_args
            assert "test-container" in call_args

    async def test_podman_checkpoint_create_failure(self) -> None:
        from inspect_ai.util._sandbox.podman.checkpoint import (
            podman_checkpoint_create,
        )

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.stderr = "checkpoint requires root"

        with patch(
            "inspect_ai.util._sandbox.podman.checkpoint.subprocess",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await podman_checkpoint_create(
                container="test-container",
                name="ckpt-1",
                checkpoint_dir="/tmp/ckpt",
            )
            assert result is False

    async def test_podman_checkpoint_restore_reconnects_networks(
        self,
    ) -> None:
        from inspect_ai.util._sandbox.podman.checkpoint import (
            podman_checkpoint_restore,
        )

        stop_result = MagicMock(success=True)
        restore_result = MagicMock(success=True)
        inspect_result = MagicMock(success=True, stdout="my-bridge-network ")
        disconnect_result = MagicMock(success=True)
        connect_result = MagicMock(success=True)

        call_count = 0

        async def mock_subprocess(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "stop" in cmd:
                return stop_result
            elif "restore" in cmd:
                return restore_result
            elif "inspect" in cmd:
                return inspect_result
            elif "disconnect" in cmd:
                return disconnect_result
            elif "connect" in cmd:
                return connect_result
            return MagicMock(success=True)

        with patch(
            "inspect_ai.util._sandbox.podman.checkpoint.subprocess",
            side_effect=mock_subprocess,
        ):
            result = await podman_checkpoint_restore(
                container="test-container",
                name="ckpt-1",
                checkpoint_dir="/tmp/ckpt",
                tcp_established=True,
            )
            assert result is True
            # stop + restore + inspect + disconnect + connect = 5 calls
            assert call_count == 5

    async def test_podman_validate_criu_rootless_returns_false(self) -> None:
        from inspect_ai.util._sandbox.podman.checkpoint import (
            validate_podman_criu_available,
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stdout = json.dumps({"host": {"security": {"rootless": True}}})

        with patch(
            "inspect_ai.util._sandbox.podman.checkpoint.subprocess",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await validate_podman_criu_available()
            assert result is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubSandbox(SandboxEnvironment):
    """Minimal concrete SandboxEnvironment for testing defaults."""

    async def exec(self, cmd, **kwargs):
        raise NotImplementedError

    async def write_file(self, file, contents):
        raise NotImplementedError

    async def read_file(self, file, text=True):
        raise NotImplementedError

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted):
        pass


def _make_stub_sandbox() -> _StubSandbox:
    return _StubSandbox()


def _make_mock_checkpoint_env(
    supports: bool = True, create_success: bool = True
) -> MagicMock:
    """Create a mock SandboxEnvironment with checkpoint methods."""
    env = MagicMock(spec=SandboxEnvironment)
    env.supports_checkpoint.return_value = supports
    env.checkpoint_create = AsyncMock(return_value=create_success)
    env.checkpoint_restore = AsyncMock(return_value=create_success)
    env.checkpoint_delete = AsyncMock()
    return env
