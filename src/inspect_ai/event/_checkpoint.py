from typing import Literal

from pydantic import Field

from inspect_ai.event._base import BaseEvent


class CheckpointEvent(BaseEvent):
    """Event recording a CRIU checkpoint of Docker containers and task state."""

    event: Literal["checkpoint"] = Field(default="checkpoint")
    """Event type."""

    checkpoint_id: str
    """Unique ID for this checkpoint."""

    checkpoint_dir: str
    """Directory where CRIU checkpoint data is stored."""

    container_checkpoints: dict[str, str]
    """Mapping of service name to checkpoint name."""

    state_file: str
    """Path to serialized TaskState JSON file."""

    message_count: int
    """Number of messages at checkpoint time."""
