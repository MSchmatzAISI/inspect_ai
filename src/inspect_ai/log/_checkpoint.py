"""Utilities for finding checkpoint events in eval logs.

Used during resume-from-checkpoint to identify the most recent
checkpoint for incomplete samples.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from inspect_ai.event._checkpoint import CheckpointEvent
from inspect_ai.log._log import EvalLog, EvalSample

# Context variable for checkpoint source callback.
# Set in _eval_async_inner when --resume-from-checkpoint is used.
_checkpoint_source_var: ContextVar[
    dict[tuple[int | str, int], CheckpointEvent] | None
] = ContextVar("_checkpoint_source", default=None)


def set_checkpoint_source(
    source: dict[tuple[int | str, int], CheckpointEvent] | None,
) -> None:
    """Set the checkpoint source for the current eval context."""
    _checkpoint_source_var.set(source)


def get_checkpoint_event(sample_id: int | str, epoch: int) -> CheckpointEvent | None:
    """Look up a checkpoint event for a given sample from the current source."""
    source = _checkpoint_source_var.get(None)
    if source is None:
        return None
    return source.get((sample_id, epoch))


def build_checkpoint_source(
    eval_log: EvalLog,
) -> dict[tuple[int | str, int], CheckpointEvent]:
    """Build a checkpoint source lookup from a failed eval log.

    Args:
        eval_log: The eval log from a crashed/failed run.

    Returns:
        Dictionary mapping (sample_id, epoch) to the latest CheckpointEvent.
    """
    checkpoints = find_all_incomplete_sample_checkpoints(eval_log)
    return {(sid, ep): ckpt for sid, ep, ckpt in checkpoints}


def find_sample_checkpoint(
    eval_log: EvalLog,
    sample_id: int | str,
    epoch: int,
) -> CheckpointEvent | None:
    """Find the most recent CheckpointEvent for a given sample.

    Scans events from newest to oldest and returns the first
    CheckpointEvent found.

    Args:
        eval_log: The eval log to search.
        sample_id: Sample ID to match.
        epoch: Epoch to match.

    Returns:
        The most recent CheckpointEvent, or None if not found.
    """
    if not eval_log.samples:
        return None

    for sample in eval_log.samples:
        if sample.id == sample_id and sample.epoch == epoch:
            return _find_checkpoint_in_sample(sample)

    return None


def _find_checkpoint_in_sample(sample: EvalSample) -> CheckpointEvent | None:
    """Find the most recent CheckpointEvent in a sample's events."""
    # Scan events from newest to oldest
    for event in reversed(sample.events):
        if isinstance(event, CheckpointEvent):
            return event
    return None


def find_all_incomplete_sample_checkpoints(
    eval_log: EvalLog,
) -> list[tuple[int | str, int, CheckpointEvent]]:
    """Find the latest checkpoint for all incomplete samples in an eval log.

    A sample is considered incomplete if it has an error or no scores.

    Args:
        eval_log: The eval log to search.

    Returns:
        List of (sample_id, epoch, CheckpointEvent) tuples for incomplete
        samples that have checkpoints.
    """
    if not eval_log.samples:
        return []

    results: list[tuple[int | str, int, CheckpointEvent]] = []
    for sample in eval_log.samples:
        # Skip completed samples (those with scores and no error)
        if sample.scores and sample.error is None:
            continue

        checkpoint = _find_checkpoint_in_sample(sample)
        if checkpoint is not None:
            results.append((sample.id, sample.epoch, checkpoint))

    return results


def load_checkpoint_state_data(state_file: str) -> dict[str, Any]:
    """Load serialized TaskState data from a checkpoint state file.

    Args:
        state_file: Path to the JSON state file.

    Returns:
        Dictionary of TaskState checkpoint data.
    """
    with open(state_file) as f:
        data: dict[str, Any] = json.load(f)
        return data
