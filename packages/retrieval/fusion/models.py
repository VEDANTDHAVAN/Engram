from __future__ import annotations
from pydantic import BaseModel, Field
from ..models import RetrievalEvidence

class FusionInput(BaseModel):
    """Retrieval result supplied to the fusion layer."""
    result: list[RetrievalEvidence] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)

class FusedCandidate(BaseModel):
    """Provider-independent candidate after fusion."""
    id: str
    content: str
    score: float = Field(ge=0.0)
    sources: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)