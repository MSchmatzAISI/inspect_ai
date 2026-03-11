"""Podman CRIU checkpoint/restore operations.

Podman checkpoint/restore with bridge networking requires:
- Root privileges (CRIU requirement)
- Containers must NOT use --init (catatonit mount breaks CRIU restore)
- After restore, DNS needs network disconnect/reconnect to work again
- In-place restore only (not export/import)
"""

import json
from logging import getLogger

from inspect_ai.util._subprocess import ExecResult, subprocess

logger = getLogger(__name__)


async def validate_podman_criu_available() -> bool:
    """Check whether Podman and CRIU are available for checkpointing.

    Podman checkpoint requires root and CRIU installed.

    Returns:
        True if podman checkpoint commands are available, False otherwise.
    """
    try:
        result: ExecResult[str] = await subprocess(
            ["podman", "info", "--format", "json"],
            timeout=30,
        )
        if not result.success:
            return False
        info = json.loads(result.stdout)
        # Check that we're running as root (required for CRIU)
        host = info.get("host", {})
        security = host.get("security", {})
        if security.get("rootless", True):
            logger.debug("Podman checkpoint requires root; running rootless")
            return False
        # Check CRIU availability
        check_result: ExecResult[str] = await subprocess(
            ["podman", "container", "checkpoint", "--help"],
            timeout=10,
        )
        return check_result.success
    except Exception:
        return False


async def podman_checkpoint_create(
    container: str,
    name: str,
    checkpoint_dir: str,
    leave_running: bool = True,
    tcp_established: bool = False,
) -> bool:
    """Create a CRIU checkpoint of a running Podman container.

    Uses in-place checkpointing (not export). The checkpoint is stored
    internally by Podman under its container storage.

    Args:
        container: Container ID or name.
        name: Checkpoint name (used for tracking, not passed to podman).
        checkpoint_dir: Directory to store checkpoint metadata.
        leave_running: If True, container continues running after checkpoint.
        tcp_established: If True, checkpoint active TCP connections.

    Returns:
        True if checkpoint was created successfully.
    """
    cmd = ["podman", "container", "checkpoint"]
    if leave_running:
        cmd.append("--leave-running")
    if tcp_established:
        cmd.append("--tcp-established")
    cmd.append(container)

    result: ExecResult[str] = await subprocess(cmd, timeout=120)
    if not result.success:
        logger.warning(
            f"Failed to create checkpoint '{name}' for container '{container}': "
            f"{result.stderr}"
        )
    return result.success


async def podman_checkpoint_restore(
    container: str,
    name: str,
    checkpoint_dir: str,
    tcp_established: bool = False,
) -> bool:
    """Restore a Podman container from a CRIU checkpoint.

    Stops the container and then restores it. After restore, reconnects
    the container's networks to fix DNS resolution.

    Args:
        container: Container ID or name.
        name: Checkpoint name (used for tracking).
        checkpoint_dir: Directory where checkpoint metadata is stored.
        tcp_established: If True, restore active TCP connections.

    Returns:
        True if restore was successful.
    """
    # Stop the container first
    stop_result: ExecResult[str] = await subprocess(
        ["podman", "stop", container],
        timeout=60,
    )
    if not stop_result.success:
        logger.warning(
            f"Failed to stop container '{container}' before restore: "
            f"{stop_result.stderr}"
        )
        return False

    # Restore the container
    cmd = ["podman", "container", "restore"]
    if tcp_established:
        cmd.append("--tcp-established")
    cmd.append(container)

    restore_result: ExecResult[str] = await subprocess(cmd, timeout=120)
    if not restore_result.success:
        logger.warning(
            f"Failed to restore checkpoint for container '{container}': "
            f"{restore_result.stderr}"
        )
        return False

    # Fix DNS resolution by reconnecting container to its networks.
    # After CRIU restore, the aardvark-dns plugin loses track of the
    # container's DNS entries. Disconnecting and reconnecting restores them.
    await _reconnect_networks(container)

    return True


async def podman_checkpoint_delete(
    container: str,
    name: str,
    checkpoint_dir: str,
) -> None:
    """Delete a Podman CRIU checkpoint.

    For in-place checkpoints, there's no separate checkpoint artifact
    to delete — Podman manages the checkpoint data internally.

    Args:
        container: Container ID or name.
        name: Checkpoint name (for logging).
        checkpoint_dir: Directory where checkpoint metadata is stored.
    """
    # Podman in-place checkpoints are managed internally;
    # no explicit delete needed for the CRIU data.
    logger.debug(
        f"Podman checkpoint '{name}' for container '{container}' "
        f"does not require explicit deletion"
    )


async def _reconnect_networks(container: str) -> None:
    """Disconnect and reconnect a container to all its networks.

    This fixes DNS resolution after CRIU restore, as the aardvark-dns
    plugin loses track of restored containers.
    """
    # Get the container's network list
    inspect_result: ExecResult[str] = await subprocess(
        [
            "podman",
            "inspect",
            container,
            "--format",
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
        ],
        timeout=30,
    )
    if not inspect_result.success:
        logger.warning(
            f"Failed to inspect container '{container}' networks: "
            f"{inspect_result.stderr}"
        )
        return

    networks = inspect_result.stdout.strip().split()
    for network in networks:
        if not network:
            continue
        # Disconnect
        await subprocess(
            ["podman", "network", "disconnect", network, container],
            timeout=30,
        )
        # Reconnect
        await subprocess(
            ["podman", "network", "connect", network, container],
            timeout=30,
        )
        logger.debug(
            f"Reconnected container '{container}' to network '{network}' "
            f"for DNS resolution"
        )
