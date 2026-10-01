from __future__ import annotations
from time import perf_counter

from packages.retrieval.policy import (
    FusionStrategyName, RetrievalPolicy, RetrievalStrategyName,
)
from .base import DecisionProvider
from .models import DecisionRequest, DecisionResponse

class RuleBasedDecisionProvider(DecisionProvider):
    """
    Deterministic baseline for structured retrieval decisions.
    
    This provider exists both as a production fallback and as the baseline against which
    future decision providers such as Jev can be evaluated.
    """
    name = "rules"

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        start = perf_counter()
        query = request.query.lower().strip()

        strategies: list[RetrievalStrategyName] = []
        reasons: list[str] = []

        # Exact identifiers, names, codes, and technical terminology tend to benefit from sparse retrieval.
        sparse_indicators = (
            "exact", "identifier", "id", "code", "error",
            "exception", "filename", "function", "class",
        )
        # Natural-language conceptual queries generally benefit from semantic/vector retrieval.
        semantic_indicators = (
            "why", "explain", "meaning", "similar", 
            "related", "concept", "understand",
        )

        has_sparse_signal = any(
            indicator in query for indicator in sparse_indicators
        )
        has_semantic_signal = any(
            indicator in query for indicator in semantic_indicators
        )

        if has_sparse_signal:
            strategies.append(RetrievalStrategyName.SPARSE)
            reasons.append(
                "Query contains signals that may benefit from lexical retrieval."
            )
        
        if has_semantic_signal:
            strategies.append(RetrievalStrategyName.VECTOR)
            reasons.append(
                "Query contains semantic signals that may benefit from vector retrieval."
            )

        if not strategies:
            strategies = [
                RetrievalStrategyName.VECTOR,
                RetrievalStrategyName.SPARSE,
            ]
            reasons.append(
                "No specialized retrieval signal detected; use hybrid retrieval."
            )

        policy = RetrievalPolicy(
            strategies=strategies, top_k=request.top_k,
            fusion=FusionStrategyName.RRF,
        )

        confidence = (
            0.85 if len(strategies) == 1
            else 0.70
        )
        latency_ms = (perf_counter() - start) * 1000

        return DecisionResponse(
            decision=policy, confidence=confidence,
            provider=self.name, reasons=reasons, 
            latency_ms=latency_ms, metadata={
                "namespace": request.namespace,
            },
        )