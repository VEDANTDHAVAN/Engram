from __future__ import annotations

from time import perf_counter

from packages.retrieval.base import RetrievalStrategy
from packages.retrieval.fusion import (
    CandidateFusion,
    ReciprocalRankFusion,
    WeightedCandidateFusion,
)
from packages.retrieval.models import RetrievalQuery, RetrievalResult
from packages.retrieval.policy import RetrievalPolicy

class HybridRetrievalService:
    """
    Orchestrates multiple retrieval strategies and fuses their results.

    This service is intentionally provider-agnostic. It operates only on
    RetrievalStrategy and CandidateFusion abstractions.
    """

    def __init__(
        self,
        *,
        vector_strategy: RetrievalStrategy | None = None,
        sparse_strategy: RetrievalStrategy | None = None,
        weighted_fusion: CandidateFusion | None = None,
        rrf_fusion: CandidateFusion | None = None,
    ) -> None:
        self._strategies: dict[str, RetrievalStrategy] = {}

        if vector_strategy is not None:
            self._strategies["vector"] = vector_strategy

        if sparse_strategy is not None:
            self._strategies["sparse"] = sparse_strategy

        self._fusions: dict[str, CandidateFusion] = {
            "weighted": (
                weighted_fusion
                if weighted_fusion is not None
                else WeightedCandidateFusion()
            ),
            "rrf": (
                rrf_fusion
                if rrf_fusion is not None
                else ReciprocalRankFusion()
            ),
        }

    async def retrieve(
        self, query: RetrievalQuery, *,
        strategies: list[str] | None = None,
        fusion: str = "rrf",
    ) -> RetrievalResult:
        """
        Execute the selected retrieval strategies and fuse their results.

        Args:
            query:
                Retrieval query containing the query text, namespace,
                top_k and filters.

            strategies:
                Strategy names to execute. Supported values are:
                "vector" and "sparse".

                If omitted, all configured strategies are executed.

            fusion:
                Fusion algorithm. Supported values are:
                "weighted" and "rrf".

        Returns:
            A single RetrievalResult containing the fused evidence.
        """
        start = perf_counter()

        selected_strategies = self._resolve_strategies(strategies)
        fusion_strategy = self._resolve_fusion(fusion)

        results: list[RetrievalResult] = []

        for strategy in selected_strategies:
            results.append(await strategy.retrieve(query))

        if not results:
            return RetrievalResult(
                evidence=[],
                strategy="hybrid",
                latency_ms=(perf_counter() - start) * 1000,
            )

        fused_candidates = fusion_strategy.fuse(results)

        evidence: list = [
            {
                "id": candidate.id,
                "content": candidate.content,
                "score": candidate.score,
                "source": ",".join(candidate.sources),
                "metadata": candidate.metadata,
            }
            for candidate in fused_candidates[: query.top_k]
        ]

        return RetrievalResult(
            evidence=evidence,
            strategy=self._build_strategy_name(
                selected_strategies,
                fusion,
            ),
            latency_ms=(perf_counter() - start) * 1000,
        )

    async def retrieve_with_policy(
        self, query: RetrievalQuery, policy: RetrievalPolicy,
    ) -> RetrievalResult:
        effective_query = query.model_copy(
            update={"top_k": policy.top_k}
        )
        return await self.retrieve(
            effective_query, strategies=[
                strategy.value for strategy in policy.strategies
            ], fusion=policy.fusion.value,
        )

    def _resolve_strategies(
        self,
        strategies: list[str] | None,
    ) -> list[RetrievalStrategy]:
        if strategies is None:
            names = list(self._strategies.keys())
        else:
            names = [name.strip().lower() for name in strategies]

        if not names:
            return []

        unknown = [
            name
            for name in names
            if name not in self._strategies
        ]

        if unknown:
            raise ValueError(
                f"Unsupported or unconfigured retrieval strategies: "
                f"{unknown}"
            )

        # Preserve caller ordering while avoiding duplicate execution.
        unique_names = list(dict.fromkeys(names))

        return [
            self._strategies[name]
            for name in unique_names
        ]

    def _resolve_fusion(
        self,
        fusion: str,
    ) -> CandidateFusion:
        fusion_name = fusion.strip().lower()

        try:
            return self._fusions[fusion_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported fusion strategy: {fusion_name!r}. "
                f"Supported values: {sorted(self._fusions)}"
            ) from exc

    @staticmethod
    def _build_strategy_name(
        strategies: list[RetrievalStrategy],
        fusion: str,
    ) -> str:
        names = [
            strategy.name
            for strategy in strategies
        ]

        return f"hybrid:{'+'.join(names)}:{fusion.strip().lower()}"