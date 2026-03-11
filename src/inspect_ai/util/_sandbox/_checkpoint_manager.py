"""Checkpoint manager for periodic CRIU checkpointing during eval runs.

Coordinates time-based checkpointing of sandbox containers and TaskState
serialization. Designed to be called from the generate loop between
complete turns. Works with any SandboxEnvironment that supports checkpointing.
"""

import json
import os
import time
from contextvars import ContextVar
from logging import getLogger

from shortuuid import uuid

from inspect_ai.event._checkpoint import CheckpointEvent
from inspect_ai.log._transcript import transcript
from inspect_ai.solver._task_state import TaskState, state_checkpoint_data
from inspect_ai.util._sandbox.environment import SandboxEnvironment

from .docker.checkpoint import prune_old_checkpoints

logger = getLogger(__name__)


class CheckpointManager:
    """Manages periodic checkpointing of sandbox containers and task state.

    Created per-sample when checkpoint mode is enabled. Tracks timing
    and coordinates the checkpoint workflow. Uses the abstract
    SandboxEnvironment checkpoint interface to support Docker, Podman,
    and other runtimes.
    """

    def __init__(
        self,
        environments: dict[str, SandboxEnvironment],
        sample_id: int | str,
        epoch: int,
        checkpoint_dir: str,
        interval_seconds: float,
        max_keep: int = 3,
    ) -> None:
        self._environments = environments
        self._sample_id = sample_id
        self._epoch = epoch
        self._checkpoint_dir = checkpoint_dir
        self._interval_seconds = interval_seconds
        self._max_keep = max_keep
        self._last_checkpoint_time = time.monotonic()
        self._count = 0

    @property
    def sample_key(self) -> str:
        """Key prefix for identifying checkpoints belonging to this sample."""
        return f"s{self._sample_id}-e{self._epoch}"

    async def maybe_checkpoint(self, state: TaskState) -> None:
        """Check if enough time has elapsed and create a checkpoint if due.

        Args:
            state: Current TaskState to snapshot.
        """
        now = time.monotonic()
        if now - self._last_checkpoint_time < self._interval_seconds:
            return
        await self.create_checkpoint(state)

    async def create_checkpoint(self, state: TaskState) -> None:
        """Create a full checkpoint: containers + state + event.

        Args:
            state: Current TaskState to snapshot.
        """
        checkpoint_id = f"{self.sample_key}-{uuid()[:8]}"

        # Checkpoint all sandbox environments that support it
        container_checkpoints: dict[str, str] = {}
        for service_name, env in self._environments.items():
            if not env.supports_checkpoint():
                continue
            checkpoint_name = f"{checkpoint_id}-{service_name}"
            success = await env.checkpoint_create(
                name=checkpoint_name,
                checkpoint_dir=self._checkpoint_dir,
                leave_running=True,
            )
            if success:
                container_checkpoints[service_name] = checkpoint_name
            else:
                logger.warning(f"Skipping checkpoint for service '{service_name}'")

        if not container_checkpoints:
            logger.warning(
                f"No containers were checkpointed for sample {self._sample_id} "
                f"epoch {self._epoch}"
            )
            return

        # Serialize TaskState to disk
        state_data = state_checkpoint_data(state)
        state_filename = f"{checkpoint_id}-state.json"
        state_file = os.path.join(self._checkpoint_dir, state_filename)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        # Emit CheckpointEvent into the transcript
        event = CheckpointEvent(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=self._checkpoint_dir,
            container_checkpoints=container_checkpoints,
            state_file=state_file,
            message_count=len(state.messages),
        )
        transcript()._event(event)

        # Update internal state
        self._count += 1
        self._last_checkpoint_time = time.monotonic()

        # Prune old checkpoints to bound disk usage
        prune_old_checkpoints(
            checkpoint_dir=self._checkpoint_dir,
            sample_key=self.sample_key,
            max_keep=self._max_keep,
        )

        logger.info(
            f"Checkpoint #{self._count} created for sample {self._sample_id} "
            f"epoch {self._epoch}: {checkpoint_id} "
            f"({len(container_checkpoints)} containers, "
            f"{len(state.messages)} messages)"
        )


# Context variable for accessing the checkpoint manager from the generate loop
_checkpoint_manager: ContextVar[CheckpointManager | None] = ContextVar(
    "_checkpoint_manager", default=None
)


def set_checkpoint_manager(manager: CheckpointManager | None) -> None:
    """Set the checkpoint manager for the current sample context."""
    _checkpoint_manager.set(manager)


def get_checkpoint_manager() -> CheckpointManager | None:
    """Get the checkpoint manager for the current sample context."""
    return _checkpoint_manager.get(None)


def setup_checkpoint_manager_from_sandbox(
    sample_id: int | str,
    epoch: int,
    checkpoint_dir: str,
    interval_seconds: float,
    max_keep: int = 3,
) -> bool:
    """Set up a CheckpointManager using the active sandbox environments.

    Looks up the current sandbox context and finds environments that
    support checkpointing.

    Args:
        sample_id: Sample ID.
        epoch: Epoch number.
        checkpoint_dir: Directory for checkpoint storage.
        interval_seconds: Interval between checkpoints.
        max_keep: Maximum checkpoints to keep per sample.

    Returns:
        True if manager was set up, False if no checkpoint-capable sandbox found.
    """
    from inspect_ai.util._sandbox.context import sandbox_environments_context_var
    from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy

    environments = sandbox_environments_context_var.get(None)
    if not environments:
        return False

    # Collect environments that support checkpointing (unwrap proxies)
    checkpoint_envs: dict[str, SandboxEnvironment] = {}
    for name, env in environments.items():
        raw_env = env._sandbox if isinstance(env, SandboxEnvironmentProxy) else env
        if raw_env.supports_checkpoint():
            checkpoint_envs[name] = raw_env

    if not checkpoint_envs:
        logger.warning("No checkpoint-capable sandbox environments found")
        return False

    logger.info(
        f"Setting up CheckpointManager with {len(checkpoint_envs)} environments: "
        f"{list(checkpoint_envs.keys())}, interval={interval_seconds}s"
    )
    manager = CheckpointManager(
        environments=checkpoint_envs,
        sample_id=sample_id,
        epoch=epoch,
        checkpoint_dir=checkpoint_dir,
        interval_seconds=interval_seconds,
        max_keep=max_keep,
    )
    set_checkpoint_manager(manager)
    return True


async def maybe_checkpoint(state: TaskState) -> None:
    """Module-level convenience: checkpoint if a manager is active and interval elapsed.

    Called from the generate loop after each complete turn.

    Args:
        state: Current TaskState.
    """
    manager = _checkpoint_manager.get(None)
    if manager is not None:
        await manager.maybe_checkpoint(state)
