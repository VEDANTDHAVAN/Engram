from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

class RetrievalQuery(BaseModel):
    """Provider-neutral representation of a retrieval request."""
    query: str
    namespace: str
    top_k: int = Field(
        default=10, ge=1, le=100,
    )
    filters: dict[str, Any] = Field(default_factory=dict)

class RetrievalEvidence(BaseModel):
    """A piece of evidence returned by any retrieval strategy."""
    id: str
    content: str
    score: float = Field(ge=0.0)
    source: str
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

class RetrievalResult(BaseModel):
    """Strategy-independent retrieval result."""
    evidence: list[RetrievalEvidence] = Field(default_factory=list)
    strategy: str
    latency_ms: float | None = None