"""Docker CRIU checkpoint/restore operations for long-running eval resumption.

This module provides low-level wrappers around Docker's experimental
checkpoint feature (backed by CRIU) and sample-level orchestration
for checkpointing/restoring all containers in a compose project.
"""

import json
import os
import shutil
from logging import getLogger

from inspect_ai.util._subprocess import ExecResult, subprocess

from .compose import compose_ps
from .util import ComposeProject

logger = getLogger(__name__)


async def validate_criu_available() -> bool:
    """Check whether CRIU and Docker experimental mode are available.

    Returns:
        True if docker checkpoint commands are available, False otherwise.
    """
    try:
        result: ExecResult[str] = await subprocess(
            ["docker", "info", "--format", "json"],
            timeout=30,
        )
        if not result.success:
            return False
        info = json.loads(result.stdout)
        server_info = info.get("ServerVersion", "")
        if not server_info:
            return False
        # Try to actually run checkpoint --help to verify CRIU is available
        check_result: ExecResult[str] = await subprocess(
            ["docker", "checkpoint", "ls", "--help"],
            timeout=10,
        )
        return check_result.success
    except Exception:
        return False


async def docker_checkpoint_create(
    container: str,
    name: str,
    checkpoint_dir: str,
    leave_running: bool = True,
) -> bool:
    """Create a CRIU checkpoint of a running Docker container.

    Args:
        container: Container ID or name.
        name: Checkpoint name.
        checkpoint_dir: Directory to store checkpoint data.
        leave_running: If True, container continues running after checkpoint.

    Returns:
        True if checkpoint was created successfully.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    cmd = [
        "docker",
        "checkpoint",
        "create",
        "--checkpoint-dir",
        checkpoint_dir,
    ]
    if leave_running:
        cmd.append("--leave-running")
    cmd.extend([container, name])

    result: ExecResult[str] = await subprocess(cmd, timeout=120)
    if not result.success:
        logger.warning(
            f"Failed to create checkpoint '{name}' for container '{container}': "
            f"{result.stderr}"
        )
    return result.success


async def docker_checkpoint_restore(
    container: str,
    name: str,
    checkpoint_dir: str,
) -> bool:
    """Restore a Docker container from a CRIU checkpoint.

    Stops the container and then starts it with the specified checkpoint.

    Args:
        container: Container ID or name.
        name: Checkpoint name.
        checkpoint_dir: Directory where checkpoint data is stored.

    Returns:
        True if restore was successful.
    """
    # Stop the container first
    stop_result: ExecResult[str] = await subprocess(
        ["docker", "stop", container],
        timeout=60,
    )
    if not stop_result.success:
        logger.warning(
            f"Failed to stop container '{container}' before restore: "
            f"{stop_result.stderr}"
        )
        return False

    # Start with checkpoint
    start_result: ExecResult[str] = await subprocess(
        [
            "docker",
            "start",
            "--checkpoint",
            name,
            "--checkpoint-dir",
            checkpoint_dir,
            container,
        ],
        timeout=120,
    )
    if not start_result.success:
        logger.warning(
            f"Failed to restore checkpoint '{name}' for container '{container}': "
            f"{start_result.stderr}"
        )
    return start_result.success


async def docker_checkpoint_delete(
    container: str,
    name: str,
    checkpoint_dir: str,
) -> None:
    """Delete a CRIU checkpoint.

    Args:
        container: Container ID or name.
        name: Checkpoint name.
        checkpoint_dir: Directory where checkpoint data is stored.
    """
    result: ExecResult[str] = await subprocess(
        [
            "docker",
            "checkpoint",
            "rm",
            "--checkpoint-dir",
            checkpoint_dir,
            container,
            name,
        ],
        timeout=30,
    )
    if not result.success:
        # Checkpoint may already have been cleaned up; just log at debug level
        logger.debug(
            f"Failed to delete checkpoint '{name}' for container '{container}': "
            f"{result.stderr}"
        )


async def checkpoint_sample_containers(
    project: ComposeProject,
    checkpoint_id: str,
    checkpoint_dir: str,
) -> dict[str, str]:
    """Checkpoint all running containers in a compose project.

    Args:
        project: The compose project.
        checkpoint_id: Unique ID for this checkpoint.
        checkpoint_dir: Base directory for checkpoint storage.

    Returns:
        Mapping of service name to checkpoint name.
    """
    containers = await compose_ps(project, status="running")
    container_checkpoints: dict[str, str] = {}

    for container_info in containers:
        service = container_info.get("Service", "")
        container_name = container_info.get("Name", "")
        if not service or not container_name:
            continue

        checkpoint_name = f"{checkpoint_id}-{service}"
        success = await docker_checkpoint_create(
            container=container_name,
            name=checkpoint_name,
            checkpoint_dir=checkpoint_dir,
            leave_running=True,
        )
        if success:
            container_checkpoints[service] = checkpoint_name
        else:
            logger.warning(
                f"Skipping checkpoint for service '{service}' "
                f"(container '{container_name}')"
            )

    return container_checkpoints


async def restore_sample_containers(
    project: ComposeProject,
    container_checkpoints: dict[str, str],
    checkpoint_dir: str,
) -> None:
    """Restore all containers in a compose project from CRIU checkpoints.

    Args:
        project: The compose project.
        container_checkpoints: Mapping of service name to checkpoint name.
        checkpoint_dir: Directory where checkpoint data is stored.
    """
    containers = await compose_ps(project, all=True)

    # Build service -> container name mapping
    service_to_container: dict[str, str] = {}
    for container_info in containers:
        service = container_info.get("Service", "")
        container_name = container_info.get("Name", "")
        if service and container_name:
            service_to_container[service] = container_name

    for service, checkpoint_name in container_checkpoints.items():
        container_name = service_to_container.get(service)
        if not container_name:
            logger.warning(
                f"Cannot restore service '{service}': container not found in project"
            )
            continue

        success = await docker_checkpoint_restore(
            container=container_name,
            name=checkpoint_name,
            checkpoint_dir=checkpoint_dir,
        )
        if not success:
            logger.warning(f"Failed to restore service '{service}' from checkpoint")


def prune_old_checkpoints(
    checkpoint_dir: str,
    sample_key: str,
    max_keep: int,
) -> None:
    """Prune old checkpoint directories, keeping only the most recent ones.

    Checkpoint subdirectories for a sample are identified by having the
    sample_key as a prefix. They are sorted by modification time and the
    oldest are removed.

    Args:
        checkpoint_dir: Base checkpoint directory.
        sample_key: Prefix identifying checkpoints for this sample.
        max_keep: Maximum number of checkpoints to keep.
    """
    if not os.path.isdir(checkpoint_dir):
        return

    # Find checkpoint subdirectories matching the sample key
    matching_dirs: list[tuple[float, str]] = []
    for entry in os.scandir(checkpoint_dir):
        if entry.is_dir() and entry.name.startswith(sample_key):
            matching_dirs.append((entry.stat().st_mtime, entry.path))

    # Sort by modification time (newest first)
    matching_dirs.sort(reverse=True)

    # Remove oldest checkpoints beyond max_keep
    for _, dir_path in matching_dirs[max_keep:]:
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            logger.debug(f"Failed to remove old checkpoint dir '{dir_path}': {e}")

    # Also remove state files beyond max_keep
    matching_files: list[tuple[float, str]] = []
    for entry in os.scandir(checkpoint_dir):
        if entry.is_file() and entry.name.startswith(sample_key):
            matching_files.append((entry.stat().st_mtime, entry.path))

    matching_files.sort(reverse=True)
    for _, file_path in matching_files[max_keep:]:
        try:
            os.remove(file_path)
        except OSError as e:
            logger.debug(f"Failed to remove old state file '{file_path}': {e}")
