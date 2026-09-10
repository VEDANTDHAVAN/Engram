from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

class RetrievalStrategyName(str, Enum):
    VECTOR = "vector"
    SPARSE = "sparse"

class FusionStrategyName(str, Enum):
    WEIGHTED = "weighted"
    RRF = "rrf"

class RetrievalPolicy(BaseModel):
    """Deterministic configuration describing how retrieval should execute."""
    strategies: list[RetrievalStrategyName] = Field(
        default_factory=lambda: [
            RetrievalStrategyName.VECTOR, 
            RetrievalStrategyName.SPARSE,
        ], min_length=1,
    )
    fusion: FusionStrategyName = FusionStrategyName.RRF
    top_k: int = Field(
        default=10, ge=1, le=100,
    )
    @field_validator("strategies")
    @classmethod
    def deduplicate_strategies(
        cls, strategies: list[RetrievalStrategyName],
    ) -> list[RetrievalStrategyName]:
        return list(dict.fromkeys(strategies))