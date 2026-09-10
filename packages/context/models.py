from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

class ContextArtifactState(str, Enum):
    """
    Lifecycle state of an externally stored context artifact.

    HIDDEN means the artifact is intentionally excluded from the active LLM context but remains fully recoverable.

    DELETED is terminal and means the original artifact is no longer expected to be recoverable.
    """
    ACTIVE = "active"
    HIDDEN = "hidden"
    DELETED = "deleted"

class ContextArtifact(BaseModel):
    """
    Provider-neutral representation of an externally stored context artifact.

    The content field is the canonical, lossless representation of the artifact.
    Context optimization must never replace it with a summary.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: str
    token_count: int = Field(ge=0)
    state: ContextArtifactState = (ContextArtifactState.ACTIVE)
    created_at: datetime
    source: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)

class ContextArtifactReference(BaseModel):
    """
    Lightweight representation of a hidden context artifact.

    This object is suitable for active-context packaging. It contains enough information for an agent/context manager
    to identify and restore the original artifact without embedding the original content.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    state: ContextArtifactState
    source: str = Field(min_length=1)
    namespace: str = Field(min_length=1)

_ALLOWED_TRANSITIONS: dict[
    ContextArtifactState, set[ContextArtifactState],
] = {
    ContextArtifactState.ACTIVE: {
        ContextArtifactState.HIDDEN, 
        ContextArtifactState.DELETED,
    },
    ContextArtifactState.HIDDEN: {
        ContextArtifactState.ACTIVE,
        ContextArtifactState.DELETED,
    },
    ContextArtifactState.DELETED: set(),
}

def can_transition(
    current: ContextArtifactState,
    target: ContextArtifactState,
) -> bool:
    """
    Return whether a context artifact may transition between states.
    """
    return target in _ALLOWED_TRANSITIONS[current]