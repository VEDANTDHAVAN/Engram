from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.decisions import (
    DecisionRequest,
    LayaDecisionProvider,
)
from packages.decisions.exceptions import LayaDecisionError
from packages.retrieval.policy import (
    RetrievalStrategyName,
)


@dataclass
class FakeDecisionResult:
    values: dict[str, Any]
    confidence: dict[str, float]
    answer_confidence: dict[str, float | None] = field(
        default_factory=dict,
    )
    routing: dict[str, Any] | None = None


class FakeLayaRouter:
    def __init__(
        self,
        strategy: str,
        confidence: float | None = 0.94,
        answer_confidence: float | None = 0.94,
        routing_model: str | None = "english",
        error: Exception | None = None,
    ) -> None:
        self.strategy = strategy
        self.confidence = confidence
        self.answer_confidence = answer_confidence
        self.routing_model = routing_model
        self.error = error

        self.calls = 0
        self.last_state: str | None = None
        self.last_schema: dict[str, Any] | None = None

    def decide(
        self,
        state: str,
        *,
        schema: dict[str, Any],
        return_details: bool,
        **kwargs: Any,
    ) -> FakeDecisionResult:
        self.calls += 1
        self.last_state = state
        self.last_schema = schema

        assert return_details is True

        if self.error is not None:
            raise self.error

        confidence: dict[str, float] = {}
        answer_confidence: dict[str, float | None] = {}

        if self.confidence is not None:
            confidence["strategy"] = self.confidence

        if self.answer_confidence is not None:
            answer_confidence["strategy"] = self.answer_confidence

        return FakeDecisionResult(
            values={
                "strategy": self.strategy,
            },
            confidence=confidence,
            answer_confidence=answer_confidence,
            routing=(
                {"model": self.routing_model}
                if self.routing_model is not None
                else None
            ),
        )


def make_provider(
    router: FakeLayaRouter,
    *,
    model: str | None = None,
) -> LayaDecisionProvider:
    return LayaDecisionProvider(
        router=router,
        model=model,
    )


async def test_laya_selects_vector() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="Why does semantic vector search work?",
            namespace="user_001",
        )
    )

    assert router.calls == 1

    assert response.decision.strategies == [
        RetrievalStrategyName.VECTOR,
    ]


async def test_laya_selects_sparse() -> None:
    router = FakeLayaRouter(
        strategy="sparse",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="Find ValueError in auth.py",
            namespace="user_001",
        )
    )

    assert response.decision.strategies == [
        RetrievalStrategyName.SPARSE,
    ]


async def test_laya_selects_hybrid() -> None:
    router = FakeLayaRouter(
        strategy="hybrid",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query=(
                "Explain why calculate_score raises ValueError "
                "and find the function."
            ),
            namespace="user_001",
        )
    )

    assert response.decision.strategies == [
        RetrievalStrategyName.VECTOR,
        RetrievalStrategyName.SPARSE,
    ]

    assert response.decision.fusion.value == "rrf"


async def test_top_k_propagation() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
            top_k=7,
        )
    )

    assert response.decision.top_k == 7


async def test_namespace_metadata() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.metadata["namespace"] == "user_001"


async def test_provider_name() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.provider == "laya"


async def test_latency_is_measured() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.latency_ms is not None
    assert response.latency_ms >= 0


async def test_model_from_routing() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        routing_model="english",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.model == "english"


async def test_model_falls_back_to_constructor() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        routing_model=None,
    )

    provider = make_provider(
        router,
        model="test-model",
    )

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.model == "test-model"


async def test_confidence_from_answer_confidence() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        answer_confidence=0.87,
        confidence=0.5,
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.confidence == 0.87
    assert response.metadata["confidence_source"] == (
        "laya_answer_confidence"
    )


async def test_confidence_falls_back_to_entropy() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        answer_confidence=None,
        confidence=0.62,
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.confidence == 0.62
    assert response.metadata["confidence_source"] == (
        "laya_confidence"
    )


async def test_confidence_unavailable_returns_one() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        answer_confidence=None,
        confidence=None,
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.confidence == 1.0
    assert response.metadata["confidence_source"] == "unavailable"


async def test_invalid_strategy_raises() -> None:
    router = FakeLayaRouter(
        strategy="invalid",
    )

    provider = make_provider(router)

    with pytest.raises(LayaDecisionError):
        await provider.decide(
            DecisionRequest(
                query="test",
                namespace="user_001",
            )
        )


async def test_provider_failure_raises() -> None:
    router = FakeLayaRouter(
        strategy="vector",
        error=RuntimeError("laya model unavailable"),
    )

    provider = make_provider(router)

    with pytest.raises(LayaDecisionError):
        await provider.decide(
            DecisionRequest(
                query="test",
                namespace="user_001",
            )
        )


async def test_reasons_are_deterministic() -> None:
    router = FakeLayaRouter(
        strategy="hybrid",
    )

    provider = make_provider(router)

    response = await provider.decide(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.reasons == [
        "Laya selected hybrid retrieval.",
    ]