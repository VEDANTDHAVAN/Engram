from __future__ import annotations
from time import perf_counter

from packages.embeddings.base import BaseEmbeddingProvider
from packages.providers.vector.base import VectorProvider

from ..base import RetrievalStrategy
from ..models import RetrievalEvidence, RetrievalQuery, RetrievalResult

class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Semantic retrieval using an embedding provider and a provider-neutral VectorProvider.
    """
    name = "vector"

    def __init__(
        self, *, embedding_provider: BaseEmbeddingProvider,
        vector_provider: VectorProvider,
    ):
        self.embedding_provider = embedding_provider
        self.vector_provider = vector_provider

    async def retrieve(
        self, query: RetrievalQuery,
    ) -> RetrievalResult:
        started = perf_counter()

        embedding = await self.embedding_provider.embed(query.query)

        results = await self.vector_provider.search(
            namespace=query.namespace, embedding=embedding.embedding,
            top_k=query.top_k, filters=query.filters,
        )

        evidence = [
            RetrievalEvidence(
                id=result.id, content=result.content or "",
                score=result.score, source=self.name, metadata=result.metadata,
            ) for result in results
        ]

        latency_ms = (perf_counter() - started) * 1000

        return RetrievalResult(
            evidence=evidence, strategy=self.name, latency_ms=latency_ms,
        )