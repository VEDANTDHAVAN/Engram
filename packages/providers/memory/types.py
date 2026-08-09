from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

class MemoryRecord(BaseModel):
    """Provider-neutral representation of a persisted memory."""
    id: UUID
    namespace: str
    content: str
    memory_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime | None = None