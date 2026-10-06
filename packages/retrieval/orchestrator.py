from __future__ import annotations

from packages.decisions import DecisionProvider, DecisionRequest
from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.models import RetrievalQuery, RetrievalResult

class RetrievalOrchestrator:
    """
    Coordinates retrieval decisions and retrieval execution.

    The decision provider selects a RetrievalPolicy.
    The hybrid retrieval service executes that policy.

    This class deliberately does not know whether the decision came
    from rules, Jev, or another provider.
    """
    def __init__(
        self, *,
        decision_provider: DecisionProvider,
        retrieval_service: HybridRetrievalService,
    ) -> None:
        self._decision_provider = decision_provider
        self._retrieval_service = retrieval_service

    async def retrieve(
        self, query: RetrievalQuery,
    ) -> RetrievalResult:
        decision = await self._decision_provider.decide(
            DecisionRequest(
                query=query.query,
                namespace=query.namespace,
                top_k=query.top_k,
            )
        )

        return await self._retrieval_service.retrieve_with_policy(
            query, decision.decision,
        )