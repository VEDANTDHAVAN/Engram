from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.providers.memory import (
    MemoryProvider,
    MemoryRecord,
)
from packages.providers.vector import (
    VectorProvider,
    VectorRecord,
)
from packages.retrieval import (
    RetrievalQuery,
)
from packages.retrieval.exceptions import RetrievalError
from packages.retrieval.strategies.vector import (
    VectorRetrievalStrategy,
)


class FakeEmbeddingProvider(
    BaseEmbeddingProvider
):
    def __init__(
        self,
        embedding: list[float] | None = None,
    ) -> None:
        self._embedding = embedding or [
            0.1,
            0.2,
            0.3,
        ]

    async def embed(
        self,
        text: str,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=self._embedding,
            dimensions=len(self._embedding),
            model="fake-model", provider="fake",
        )


class FakeVectorProvider(
    VectorProvider
):
    def __init__(
        self,
        results: list[VectorRecord],
    ) -> None:
        self.results = results
        self.last_namespace: str | None = None
        self.last_embedding: list[float] | None = None
        self.last_top_k: int | None = None
        self.last_filters: dict | None = None

    async def upsert(
        self,
        *,
        namespace: str,
        record_id: str,
        embedding: list[float],
        content: str,
        metadata: dict | None = None,
    ) -> None:
        pass

    async def search(
        self,
        *,
        namespace: str,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[VectorRecord]:
        self.last_namespace = namespace
        self.last_embedding = embedding
        self.last_top_k = top_k
        self.last_filters = filters

        return self.results[:top_k]

    async def delete(
        self,
        *,
        namespace: str,
        record_id: str,
    ) -> None:
        pass


class FakeMemoryProvider(
    MemoryProvider
):
    def __init__(
        self,
        memory: MemoryRecord | None,
    ) -> None:
        self.memory = memory
        self.requested_id: UUID | None = None
        self.requested_namespace: str | None = None

    async def create(
        self,
        *,
        namespace: str,
        content: str,
        metadata: dict | None = None,
        memory_type: str,
    ) -> MemoryRecord:
        raise NotImplementedError

    async def get(
        self,
        memory_id: UUID,
        *,
        namespace: str,
    ) -> MemoryRecord | None:
        self.requested_id = memory_id
        self.requested_namespace = namespace

        if self.memory is None:
            return None

        if self.memory.id != memory_id:
            return None

        if self.memory.namespace != namespace:
            return None

        return self.memory

    async def update(
        self,
        memory_id: UUID,
        *,
        namespace: str,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        raise NotImplementedError

    async def delete(
        self,
        memory_id: UUID,
        *,
        namespace: str,
    ) -> None:
        raise NotImplementedError


def make_memory(
    *,
    memory_id: UUID | None = None,
    namespace: str = "tenant-a",
    content: str = "CANONICAL MEMORY",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)

    return MemoryRecord(
        id=memory_id or uuid4(),
        namespace=namespace,
        content=content,
        memory_type="semantic",
        metadata={
            "source": "postgres",
        },
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_vector_strategy_retrieves_vector_results():
    """
    Preserve the original vector-strategy behavior when no
    MemoryProvider is supplied.
    """

    vector_result = VectorRecord(
        id="memory-1",
        score=0.95,
        content="vector content",
        metadata={
            "source": "turbovec",
        },
    )

    vector_provider = FakeVectorProvider(
        [vector_result]
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=vector_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="test query",
            namespace="tenant-a",
            top_k=5,
        )
    )

    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.id == "memory-1"
    assert evidence.content == "vector content"
    assert evidence.score == 0.95
    assert evidence.source == "vector"


@pytest.mark.asyncio
async def test_vector_strategy_passes_query_parameters():
    vector_provider = FakeVectorProvider([])

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(
            [0.5, 0.6, 0.7]
        ),
        vector_provider=vector_provider,
    )

    query = RetrievalQuery(
        query="database architecture",
        namespace="tenant-a",
        top_k=7,
        filters={
            "memory_type": "semantic",
        },
    )

    await strategy.retrieve(query)

    assert (
        vector_provider.last_namespace
        == "tenant-a"
    )

    assert (
        vector_provider.last_embedding
        == [0.5, 0.6, 0.7]
    )

    assert vector_provider.last_top_k == 7

    assert vector_provider.last_filters == {
        "memory_type": "semantic",
    }


@pytest.mark.asyncio
async def test_vector_strategy_hydrates_canonical_memory():
    """
    VectorProvider supplies the candidate.
    MemoryProvider supplies canonical content.
    """

    memory_id = uuid4()

    memory = make_memory(
        memory_id=memory_id,
        content="CANONICAL POSTGRES CONTENT",
    )

    vector_result = VectorRecord(
        id=str(memory_id),
        score=0.95,
        content="STALE VECTOR CONTENT",
        metadata={
            "source": "turbovec",
        },
    )

    vector_provider = FakeVectorProvider(
        [vector_result]
    )

    memory_provider = FakeMemoryProvider(
        memory
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=vector_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="test query",
            namespace="tenant-a",
            top_k=5,
        )
    )

    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.id == str(memory_id)

    # PostgreSQL is authoritative.
    assert (
        evidence.content
        == "CANONICAL POSTGRES CONTENT"
    )

    # Vector similarity is preserved.
    assert evidence.score == 0.95

    assert evidence.source == "vector"

    assert (
        evidence.metadata["source"]
        == "postgres"
    )

    assert (
        evidence.metadata["memory_type"]
        == "semantic"
    )

    assert (
        memory_provider.requested_id
        == memory_id
    )

    assert (
        memory_provider.requested_namespace
        == "tenant-a"
    )


@pytest.mark.asyncio
async def test_vector_strategy_discards_stale_vector_entry():
    """
    If TurboVec contains an ID whose canonical memory no longer
    exists, the stale candidate must not reach the LLM context.
    """

    memory_id = uuid4()

    vector_result = VectorRecord(
        id=str(memory_id),
        score=0.95,
        content="STALE CONTENT",
        metadata={},
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=FakeVectorProvider(
            [vector_result]
        ),
        memory_provider=FakeMemoryProvider(
            None
        ),
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="deleted memory",
            namespace="tenant-a",
        )
    )

    assert result.evidence == []


@pytest.mark.asyncio
async def test_vector_strategy_rejects_invalid_memory_id():
    vector_result = VectorRecord(
        id="not-a-uuid",
        score=0.90,
        content="invalid",
        metadata={},
    )

    memory_provider = FakeMemoryProvider(
        None
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=FakeVectorProvider(
            [vector_result]
        ),
        memory_provider=memory_provider,
    )

    with pytest.raises(
        RetrievalError,
        match="invalid memory ID",
    ):
        await strategy.retrieve(
            RetrievalQuery(
                query="test",
                namespace="tenant-a",
            )
        )

    assert (
        memory_provider.requested_id
        is None
    )


@pytest.mark.asyncio
async def test_vector_strategy_supports_empty_results():
    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=FakeVectorProvider([]),
        memory_provider=FakeMemoryProvider(
            None
        ),
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="nothing",
            namespace="tenant-a",
        )
    )

    assert result.evidence == []

    assert result.strategy == "vector"

    assert result.latency_ms is not None
    assert result.latency_ms >= 0