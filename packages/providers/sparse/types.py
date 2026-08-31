from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

class SparseRecord(BaseModel):
    """
    Provider-neutral result returned by sparse retrieval.

    PostgreSQL remains the canonical memory store.
    The sparse provider only returns retrieval candidates and associated searchable metadata.
    """
    id: str
    score: float = Field(ge=0.0)
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)