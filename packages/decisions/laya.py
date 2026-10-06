from __future__ import annotations

from time import perf_counter
from typing import Any

from packages.retrieval.policy import (
    FusionStrategyName,
    RetrievalPolicy,
    RetrievalStrategyName,
)

from .base import DecisionProvider
from .models import DecisionRequest, DecisionResponse


RETRIEVAL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": [
                "vector",
                "sparse",
                "hybrid",
            ],
            "description": (
                "Select the retrieval strategy that best matches the query. "
                "Use vector for conceptual or semantic queries. "
                "Use sparse for exact identifiers, filenames, symbols, "
                "error messages, or lexical matches. "
                "Use hybrid when both semantic and lexical evidence are useful."
            ),
        },
    },
}


class LayaDecisionProvider(DecisionProvider):
    """
    Laya-backed structured decision provider.

    Laya makes the routing decision.
    Engram converts that decision into a RetrievalPolicy.
    Engram remains responsible for executing retrieval.
    """

    name = "laya"

    def __init__(
        self,
        *,
        router: Any,
        model: str | None = None,
    ) -> None:
        self._router = router
        self._model = model

    async def decide(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        start = perf_counter()

        state = self._build_state(request)

        result = self._router.decide(
            state,
            schema=RETRIEVAL_DECISION_SCHEMA,
            return_details=True,
            **self._model_kwargs(),
        )

        policy = self._build_policy(
            result,
            top_k=request.top_k,
        )

        confidence = self._extract_confidence(result)

        latency_ms = (perf_counter() - start) * 1000

        return DecisionResponse(
            decision=policy,
            confidence=confidence,
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
            reasons=self._build_reasons(result),
            metadata={
                "namespace": request.namespace,
                "decision_type": "retrieval_strategy",
            },
        )

    def _model_kwargs(self) -> dict[str, str]:
        if self._model is None:
            return {}

        return {"model": self._model}

    @staticmethod
    def _build_state(
        request: DecisionRequest,
    ) -> str:
        return (
            f"Query: {request.query}\n"
            f"Namespace: {request.namespace}\n"
            f"Requested top_k: {request.top_k}"
        )

    @staticmethod
    def _build_policy(
        result: Any,
        *,
        top_k: int,
    ) -> RetrievalPolicy:
        values = getattr(result, "values", None)

        if values is None and isinstance(result, dict):
            values = result

        if values is None:
            raise ValueError(
                "Laya decision result does not contain structured values."
            )

        strategy = values.get("strategy")

        if strategy == "vector":
            strategies = [
                RetrievalStrategyName.VECTOR,
            ]

        elif strategy == "sparse":
            strategies = [
                RetrievalStrategyName.SPARSE,
            ]

        elif strategy == "hybrid":
            strategies = [
                RetrievalStrategyName.VECTOR,
                RetrievalStrategyName.SPARSE,
            ]

        else:
            raise ValueError(
                f"Unsupported Laya retrieval strategy: {strategy!r}"
            )

        return RetrievalPolicy(
            strategies=strategies,
            fusion=FusionStrategyName.RRF,
            top_k=top_k,
        )

    @staticmethod
    def _extract_confidence(result: Any) -> float:
        confidence = getattr(result, "confidence", None)

        if isinstance(confidence, dict):
            confidence = confidence.get("strategy")

        if confidence is None and isinstance(result, dict):
            confidence = result.get("confidence")

        if confidence is None:
            return 0.0

        return max(
            0.0,
            min(1.0, float(confidence)),
        )

    @staticmethod
    def _build_reasons(
        result: Any,
    ) -> list[str]:
        strategy = None

        values = getattr(result, "values", None)

        if values is not None:
            strategy = values.get("strategy")

        if strategy is None and isinstance(result, dict):
            strategy = result.get("strategy")

        if strategy is None:
            return []

        return [
            f"Laya selected retrieval strategy: {strategy}.",
        ]