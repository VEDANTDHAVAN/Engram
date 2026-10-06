from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from packages.retrieval.policy import (
    FusionStrategyName,
    RetrievalPolicy,
    RetrievalStrategyName,
)

from .base import DecisionProvider
from .exceptions import LayaDecisionError
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

        try:
            result = await asyncio.to_thread(
                self._router.decide,
                state,
                schema=RETRIEVAL_DECISION_SCHEMA,
                return_details=True,
                **self._model_kwargs(),
            )

        except Exception as exc:
            raise LayaDecisionError(
                f"Laya decision failed: {exc}"
            ) from exc

        policy = self._build_policy(
            result,
            top_k=request.top_k,
        )

        confidence, confidence_source = self._extract_confidence(result)

        latency_ms = (perf_counter() - start) * 1000

        return DecisionResponse(
            decision=policy,
            confidence=confidence,
            provider=self.name,
            model=self._extract_model(result),
            latency_ms=latency_ms,
            reasons=self._build_reasons(result),
            metadata={
                "namespace": request.namespace,
                "decision_type": "retrieval_strategy",
                "confidence_source": confidence_source,
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
            raise LayaDecisionError(
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
            raise LayaDecisionError(
                f"Unsupported Laya retrieval strategy: {strategy!r}"
            )

        return RetrievalPolicy(
            strategies=strategies,
            fusion=FusionStrategyName.RRF,
            top_k=top_k,
        )

    @staticmethod
    def _extract_confidence(
        result: Any,
    ) -> tuple[float, str]:
        answer_confidence = getattr(result, "answer_confidence", None)

        if isinstance(answer_confidence, dict):
            value = answer_confidence.get("strategy")

            if value is not None:
                return (
                    max(0.0, min(1.0, float(value))),
                    "laya_answer_confidence",
                )

        confidence = getattr(result, "confidence", None)

        if isinstance(confidence, dict):
            value = confidence.get("strategy")

            if value is not None:
                return (
                    max(0.0, min(1.0, float(value))),
                    "laya_confidence",
                )

        if isinstance(result, dict):
            value = result.get("confidence")

            if value is not None:
                return (
                    max(0.0, min(1.0, float(value))),
                    "laya_confidence",
                )

        return 1.0, "unavailable"

    def _extract_model(
        self,
        result: Any,
    ) -> str | None:
        routing = getattr(result, "routing", None)

        if isinstance(routing, dict):
            model = routing.get("model")

            if isinstance(model, str) and model:
                return model

        return self._model

    @staticmethod
    def _build_reasons(
        result: Any,
    ) -> list[str]:
        values = getattr(result, "values", None)

        if values is None and isinstance(result, dict):
            values = result

        if values is None:
            return []

        strategy = values.get("strategy")

        if strategy == "vector":
            return [
                "Laya selected semantic/vector retrieval.",
            ]

        if strategy == "sparse":
            return [
                "Laya selected lexical/sparse retrieval.",
            ]

        if strategy == "hybrid":
            return [
                "Laya selected hybrid retrieval.",
            ]

        return []