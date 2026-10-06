from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.decisions import (
    DecisionRequest,
    LayaDecisionProvider,
)
from packages.retrieval.policy import (
    RetrievalStrategyName,
)


@dataclass
class FakeDecisionResult:
    values: dict[str, Any]
    confidence: dict[str, float]


class FakeLayaRouter:
    def __init__(
        self,
        strategy: str,
        confidence: float = 0.94,
    ) -> None:
        self.strategy = strategy
        self.confidence = confidence

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

        return FakeDecisionResult(
            values={
                "strategy": self.strategy,
            },
            confidence={
                "strategy": self.confidence,
            },
        )


async def test_laya_selects_vector() -> None:
    router = FakeLayaRouter(
        strategy="vector",
    )

    provider = LayaDecisionProvider(
        router=router,
        model="test-model",
    )

    response = await provider.decide(
        DecisionRequest(
            query="Why does distributed consensus work?",
            namespace="user_001",
            top_k=10,
        )
    )

    assert router.calls == 1

    assert response.provider == "laya"
    assert response.model == "test-model"
    assert response.confidence == 0.94

    assert response.decision.strategies == [
        RetrievalStrategyName.VECTOR,
    ]


async def test_laya_selects_sparse() -> None:
    router = FakeLayaRouter(
        strategy="sparse",
    )

    provider = LayaDecisionProvider(
        router=router,
    )

    response = await provider.decide(
        DecisionRequest(
            query="Find exception ValueError in auth.py",
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

    provider = LayaDecisionProvider(
        router=router,
    )

    response = await provider.decide(
        DecisionRequest(
            query="Why did authentication fail with this error?",
            namespace="user_001",
        )
    )

    assert response.decision.strategies == [
        RetrievalStrategyName.VECTOR,
        RetrievalStrategyName.SPARSE,
    ]

    assert response.decision.fusion.value == "rrf"


async def test_laya_receives_query_context() -> None:
    router = FakeLayaRouter(
        strategy="hybrid",
    )

    provider = LayaDecisionProvider(
        router=router,
    )

    request = DecisionRequest(
        query="distributed systems",
        namespace="tenant_123",
        top_k=7,
    )

    await provider.decide(request)

    assert router.last_state is not None
    assert "distributed systems" in router.last_state
    assert "tenant_123" in router.last_state
    assert "7" in router.last_state