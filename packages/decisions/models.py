from __future__ import annotations

from pydantic import BaseModel, Field

from packages.retrieval.policy import RetrievalPolicy

class DecisionRequest(BaseModel):
    """
    Provider-neutral input for a structured Engram decision.

    The initial decision use case is retrieval-strategy selection.
    """
    query: str
    namespace: str
    top_k: int = Field(default=10, ge=1, le=100)

class DecisionResponse(BaseModel):
    """
    Provider-neutral result of a structured decision.
    
    The provider makes the decision; Engram remains responsible for executing it.
    """
    decision: RetrievalPolicy
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provider: str
    model: str | None = None
    latency_ms: float | None = None

    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)