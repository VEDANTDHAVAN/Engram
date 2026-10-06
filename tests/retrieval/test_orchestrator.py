from packages.decisions import (
    DecisionProvider, DecisionRequest, DecisionResponse,
)
from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.models import (
    RetrievalEvidence, RetrievalQuery, RetrievalResult,
)
from packages.retrieval.orchestrator import RetrievalOrchestrator
from packages.retrieval.policy import (
    FusionStrategyName, RetrievalPolicy, RetrievalStrategyName,
)
from packages.retrieval.base import RetrievalStrategy

class FakeDecisionProvider(DecisionProvider):
    name = "fake"

    def __init__(self, policy: RetrievalPolicy) -> None:
        self.policy = policy
        self.calls = 0
        self.last_request: DecisionRequest | None = None

    async def decide(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        self.calls += 1
        self.last_request = request

        return DecisionResponse(
            decision=self.policy,
            confidence=0.9,
            provider=self.name,
        )


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


async def test_orchestrator_executes_decision_policy() -> None:
    vector = FakeRetrievalStrategy("vector")
    sparse = FakeRetrievalStrategy("sparse")

    service = HybridRetrievalService(
        vector_strategy=vector,
        sparse_strategy=sparse,
    )

    decision_provider = FakeDecisionProvider(
        RetrievalPolicy(
            strategies=[
                RetrievalStrategyName.VECTOR,
            ],
            fusion=FusionStrategyName.RRF,
            top_k=5,
        )
    )

    orchestrator = RetrievalOrchestrator(
        decision_provider=decision_provider,
        retrieval_service=service,
    )

    result = await orchestrator.retrieve(make_query())

    assert decision_provider.calls == 1
    assert vector.calls == 1
    assert sparse.calls == 0

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "vector-1"


async def test_orchestrator_passes_query_context_to_decision_provider() -> None:
    vector = FakeRetrievalStrategy("vector")

    service = HybridRetrievalService(
        vector_strategy=vector,
    )

    decision_provider = FakeDecisionProvider(
        RetrievalPolicy(
            strategies=[
                RetrievalStrategyName.VECTOR,
            ],
            fusion=FusionStrategyName.RRF,
            top_k=5,
        )
    )

    orchestrator = RetrievalOrchestrator(
        decision_provider=decision_provider,
        retrieval_service=service,
    )

    query = make_query()

    await orchestrator.retrieve(query)

    assert decision_provider.last_request is not None
    assert decision_provider.last_request.query == query.query
    assert decision_provider.last_request.namespace == query.namespace
    assert decision_provider.last_request.top_k == query.top_k