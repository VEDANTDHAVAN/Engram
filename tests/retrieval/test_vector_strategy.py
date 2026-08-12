import pytest

from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.retrieval import RetrievalQuery, VectorRetrievalStrategy
from packages.providers.vector import VectorProvider
from packages.providers.vector import VectorRecord

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    async def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            dimensions=3, model="fake",
            provider="fake",
        )

class FakeVectorProvider(VectorProvider):
    async def upsert(
        self, *, namespace, record_id, embedding, content, metadata=None,
    ):
        pass

    async def search(
        self, *, namespace, embedding, top_k=10, filters=None,
    ):
        return [
            VectorRecord(
                id="memory-1", score=0.95,
                content="CAML is a context engineering platform.",
                metadata={"memory_type": "semantic"},
            ),
            VectorRecord(
                id="memory-2", score=0.82,
                content="CAML provides memory-aware retrieval.",
                metadata={"memory_type": "semantic"},
            ),
        ]

    async def delete(
        self, *, namespace, record_id,
    ):
        pass

@pytest.mark.asyncio
async def test_vector_retrieval():
    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=FakeVectorProvider(),
    )
    query = RetrievalQuery(
        query="What is CAML?",
        namespace="test",
        top_k=2,
    )

    result = await strategy.retrieve(query)

    assert result.strategy == "vector"

    assert len(result.evidence) == 2

    assert result.evidence[0].id == "memory-1"

    assert result.evidence[0].score == 0.95

    assert result.latency_ms is not None