from __future__ import annotations

from packages.decisions import (
    DecisionOrchestrator,
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
)
from packages.retrieval.models import (
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
)
from packages.retrieval.policy import (
    FusionStrategyName,
    RetrievalPolicy,
    RetrievalStrategyName,
)
from packages.retrieval.base import RetrievalStrategy
from packages.retrieval.hybrid import HybridRetrievalService


class FakeDecisionProvider(DecisionProvider):
    name = "fake"

    def __init__(
        self,
        policy: RetrievalPolicy,
    ) -> None:
        self.policy = policy
        self.calls = 0

    async def decide(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        self.calls += 1

        return DecisionResponse(
            decision=self.policy,
            confidence=0.95,
            provider=self.name,
        )


class FailingDecisionProvider(DecisionProvider):
    name = "failing"

    async def decide(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        raise RuntimeError("decision provider unavailable")


class FakeRetrievalStrategy(RetrievalStrategy):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        self.calls += 1

        return RetrievalResult(
            evidence=[
                RetrievalEvidence(
                    id=f"{self.name}-1",
                    content=f"{self.name} result",
                    score=1.0,
                    source=self.name,
                )
            ],
            strategy=self.name,
        )


def make_query() -> RetrievalQuery:
    return RetrievalQuery(
        query="distributed systems",
        namespace="user_001",
        top_k=5,
    )


def make_policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        strategies=[
            RetrievalStrategyName.VECTOR,
        ],
        fusion=FusionStrategyName.RRF,
        top_k=5,
    )

async def test_decision_is_used_to_execute_retrieval() -> None:
    provider = FakeDecisionProvider(
        make_policy(),
    )

    fallback = FakeDecisionProvider(
        make_policy(),
    )

    vector = FakeRetrievalStrategy("vector")

    retrieval = HybridRetrievalService(
        vector_strategy=vector,
    )

    orchestrator = DecisionOrchestrator(
        decision_provider=provider,
        fallback_provider=fallback,
        retrieval_service=retrieval,
    )

    result = await orchestrator.retrieve(
        make_query(),
    )

    assert provider.calls == 1
    assert fallback.calls == 0
    assert vector.calls == 1

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "vector-1"

async def test_provider_failure_falls_back_to_rules() -> None:
    primary = FailingDecisionProvider()

    fallback = FakeDecisionProvider(
        make_policy(),
    )

    vector = FakeRetrievalStrategy("vector")

    retrieval = HybridRetrievalService(
        vector_strategy=vector,
    )

    orchestrator = DecisionOrchestrator(
        decision_provider=primary,
        fallback_provider=fallback,
        retrieval_service=retrieval,
    )

    result = await orchestrator.retrieve(
        make_query(),
    )

    assert fallback.calls == 1
    assert vector.calls == 1

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "vector-1"

async def test_fallback_metadata_is_recorded() -> None:
    primary = FailingDecisionProvider()

    fallback = FakeDecisionProvider(
        make_policy(),
    )

    vector = FakeRetrievalStrategy("vector")

    retrieval = HybridRetrievalService(
        vector_strategy=vector,
    )

    orchestrator = DecisionOrchestrator(
        decision_provider=primary,
        fallback_provider=fallback,
        retrieval_service=retrieval,
    )

    response = await orchestrator._make_decision(
        DecisionRequest(
            query="test",
            namespace="user_001",
        )
    )

    assert response.metadata["fallback"] is True
    assert response.metadata["primary_provider"] == "failing"

    fallback_reason = response.metadata["fallback_reason"]

    assert isinstance(fallback_reason, str)
    assert "decision provider unavailable" in fallback_reason