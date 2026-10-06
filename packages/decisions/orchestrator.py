from __future__ import annotations

from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.models import RetrievalQuery, RetrievalResult

from .base import DecisionProvider
from .models import DecisionRequest, DecisionResponse


class DecisionOrchestrator:
    """
    Coordinates structured decision-making with retrieval execution.

    Decision providers decide.
    Retrieval services execute.

    A provider failure falls back to the configured fallback provider.
    """

    def __init__(
        self,
        *,
        decision_provider: DecisionProvider,
        fallback_provider: DecisionProvider,
        retrieval_service: HybridRetrievalService,
    ) -> None:
        self._decision_provider = decision_provider
        self._fallback_provider = fallback_provider
        self._retrieval_service = retrieval_service

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        request = DecisionRequest(
            query=query.query,
            namespace=query.namespace,
            top_k=query.top_k,
        )

        response = await self._make_decision(request)

        return await self._retrieval_service.retrieve_with_policy(
            query,
            response.decision,
        )

    async def _make_decision(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        try:
            return await self._decision_provider.decide(request)

        except Exception as exc:
            fallback = await self._fallback_provider.decide(request)

            return fallback.model_copy(
                update={
                    "metadata": {
                        **fallback.metadata,
                        "fallback": True,
                        "fallback_reason": str(exc),
                        "primary_provider": (
                            self._decision_provider.name
                        ),
                    },
                },
            )