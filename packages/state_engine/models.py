from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .types import MemoryImportance, MemoryStatus, MemoryType

class Memory(BaseModel):
    """
    Domain representation of a CAML memory.

    This is a deliberately independent of PostgreSQL, pgvector, Redis, Qdrant, or any other storage implementation.
    """
    id: UUID = Field(default_factory=uuid4)
    namespace: str
    content: str
    memory_type: MemoryType
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: MemoryImportance = MemoryImportance.MEDIUM
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    superseded_by: UUID | None = None