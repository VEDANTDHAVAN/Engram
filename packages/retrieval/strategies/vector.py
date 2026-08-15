from __future__ import annotations

from time import perf_counter
from uuid import UUID

from packages.embeddings.base import BaseEmbeddingProvider
from packages.providers.memory import MemoryProvider
from packages.providers.vector.base import VectorProvider

from ..base import RetrievalStrategy
from ..exceptions import RetrievalError
from ..models import (
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
)


class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Semantic retrieval using an embedding provider and a
    provider-neutral VectorProvider.

    VectorProvider is responsible only for dense candidate retrieval.

    MemoryProvider is responsible for hydrating the canonical memory
    records from durable storage.
    """

    name = "vector"

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_provider: VectorProvider,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_provider = vector_provider
        self.memory_provider = memory_provider

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        started = perf_counter()

        # --------------------------------------------------------------
        # 1. Generate query embedding
        # --------------------------------------------------------------
        embedding_result = await self.embedding_provider.embed(
            query.query
        )

        # --------------------------------------------------------------
        # 2. Retrieve dense candidates
        # --------------------------------------------------------------
        results = await self.vector_provider.search(
            namespace=query.namespace,
            embedding=embedding_result.embedding,
            top_k=query.top_k,
            filters=query.filters,
        )

        evidence: list[RetrievalEvidence] = []

        # --------------------------------------------------------------
        # 3. Hydrate canonical memories
        # --------------------------------------------------------------
        for result in results:
            if self.memory_provider is None:
                # Backward-compatible behavior for existing isolated
                # vector-strategy tests.
                evidence.append(
                    RetrievalEvidence(
                        id=result.id,
                        content=result.content or "",
                        score=result.score,
                        source=self.name,
                        metadata=result.metadata,
                    )
                )
                continue

            try:
                memory_id = UUID(result.id)
            except ValueError as exc:
                raise RetrievalError(
                    f"Vector provider returned an invalid memory ID: "
                    f"{result.id!r}"
                ) from exc

            memory = await self.memory_provider.get(
                memory_id,
                namespace=query.namespace,
            )

            # The vector index may contain a stale ID after a memory
            # deletion. Do not expose stale indexed content.
            if memory is None:
                continue

            metadata = dict(memory.metadata)

            metadata.setdefault(
                "memory_type",
                memory.memory_type,
            )

            evidence.append(
                RetrievalEvidence(
                    id=str(memory.id),
                    content=memory.content,
                    score=result.score,
                    source=self.name,
                    metadata=metadata,
                )
            )

        # --------------------------------------------------------------
        # 4. Measure retrieval latency
        # --------------------------------------------------------------
        latency_ms = (
            perf_counter() - started
        ) * 1000

        return RetrievalResult(
            evidence=evidence,
            strategy=self.name,
            latency_ms=latency_ms,
        )