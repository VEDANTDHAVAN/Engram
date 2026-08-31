from __future__ import annotations

from time import perf_counter
from uuid import UUID

from packages.providers.memory import MemoryProvider
from packages.providers.sparse import SparseProvider
from packages.retrieval.exceptions import RetrievalError

from ..base import RetrievalStrategy
from ..models import RetrievalEvidence, RetrievalQuery, RetrievalResult

class SparseRetrievalStrategy(RetrievalStrategy):
    """
    Lexical/sparse retrieval strategy.

    The strategy depends only on SparseProvider and optionally MemoryProvider.
    It doesnot know which sparse backend is used.

    SparseProvider: Candidate generation

    MemoryProvider: Canonical memory hydration and namespace enforcement.
    """
    name = "sparse"

    def __init__(
        self, *, sparse_provider: SparseProvider,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        self.sparse_provider = sparse_provider
        self.memory_provider = memory_provider

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started = perf_counter()

        filters = dict(query.filters)

        if self.memory_provider is not None:
            authorized_ids = await self.memory_provider.list_ids(
                namespace=query.namespace,
                filters=query.filters,
            )
            filters["allowed_ids"] = [
                str(memory_id) for memory_id in authorized_ids
            ]

        try:
            results = await self.sparse_provider.search(
                namespace=query.namespace,
                query=query.query,
                top_k=query.top_k,
                filters=filters,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Sparse provider retrieval failed: {exc}"
            ) from exc

        evidence: list[RetrievalEvidence] = []

        for result in results:
            evidence_item = await self._build_evidence(
                result=result, namespace=query.namespace,
            )

            if evidence_item is not None:
                evidence.append(evidence_item)

        latency_ms = (perf_counter() - started) * 1000

        return RetrievalResult(
            evidence=evidence, strategy=self.name, latency_ms=latency_ms,
        )

    async def _build_evidence(
        self, *, result, namespace: str, 
    ) -> RetrievalEvidence | None:
        """
        Hydrate canonical memory when a MemoryProvider is available.

        Sparse indexes are retrieval indexes, not the source of truth.
        """
        if self.memory_provider is None:
            return RetrievalEvidence(
                id=result.id, score=result.score,
                source=self.name, metadata=result.metadata,
                content=result.content or "",
            )

        try:
            memory_id = UUID(result.id)
        except (ValueError, TypeError):
            # A sparse provider may contain stale/malformed IDs.
            # Such records must not enter the final CAML context.
            return None

        memory = await self.memory_provider.get(
            memory_id, namespace=namespace,
        )

        if memory is None:
            # Stale index entry or namespace mismatch.
            return None

        metadata = dict(memory.metadata)

        # Preserve retrieval/index metadata without allowing it to overwrite canonical memory metadata.
        for key, value in result.metadata.items():
            metadata.setdefault(key, value)

        return RetrievalEvidence(
            id=str(memory.id), 
            content=memory.content,
            score=result.score,
            source=self.name, metadata=metadata,
        )
