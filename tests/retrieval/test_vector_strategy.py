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
    name = "fake"
    dimensions = 3

    def __init__(
        self,
        embedding: list[float] | None = None,
    ) -> None:
        self._embedding = (
            list(embedding) if embedding is not None
            else [0.1, 0.2, 0.3]
        )

    async def embed(
        self,
        text: str,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=list(self._embedding),
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

    async def initialize(self) -> None:
        pass
    
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
        self.last_embedding = list(embedding)
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
        memories: list[MemoryRecord] | None = None,
    ) -> None:
        self.memories = memories or []
        self.requested_id: UUID | None = None
        self.requested_namespace: str | None = None
        self.last_filters: dict | None = None

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

        for memory in self.memories:
            if (
                memory.id == memory_id
                and memory.namespace == namespace
            ):
                return memory

        return None

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

    async def list_ids(
        self, *, namespace: str, filters: dict | None = None,
    ) -> list[UUID]:
        self.last_filters = filters

        authorized = [
            memory.id for memory in self.memories
            if memory.namespace == namespace
        ]

        if not filters:
            return authorized

        allowed_ids = filters.get("allowed_ids")
        if allowed_ids is None:
            return authorized

        allowed = {
            str(value) for value in allowed_ids
        }

        return [
            memory_id for memory_id in authorized
            if str(memory_id) in allowed
        ]


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
    memory_id = uuid4()

    memory = make_memory(
        memory_id=memory_id,
        namespace="tenant-a",
        content="vector content",
    )

    vector_result = VectorRecord(
        id=str(memory_id),
        score=0.95,
        content="stale vector content",
        metadata={
            "source": "turbovec",
        },
    )

    vector_provider = FakeVectorProvider(
        [vector_result]
    )

    memory_provider = FakeMemoryProvider(
        [memory]
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

    # Canonical memory wins over vector-index content.
    assert evidence.content == "vector content"

    assert evidence.score == 0.95
    assert evidence.source == "vector"

@pytest.mark.asyncio
async def test_vector_strategy_passes_query_parameters():
    memory_id = uuid4()

    memory = make_memory(
        memory_id=memory_id,
        namespace="tenant-a",
    )

    vector_provider = FakeVectorProvider([])

    embedding_provider = FakeEmbeddingProvider(
        [0.5, 0.6, 0.7]
    )

    memory_provider = FakeMemoryProvider(
        [memory]
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=embedding_provider,
        vector_provider=vector_provider,
        memory_provider=memory_provider,
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

    assert vector_provider.last_filters is not None

    assert (
        vector_provider.last_filters["memory_type"]
        == "semantic"
    )

    # The retrieval strategy must add the canonical
    # namespace-authorized ID allowlist.
    assert (
        vector_provider.last_filters["allowed_ids"]
        == [str(memory_id)]
    )


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
        [memory]
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

@pytest.mark.asyncio
async def test_vector_strategy_enforces_namespace_isolation():
    from datetime import datetime, timezone

    memory_a_id = uuid4()
    memory_b_id = uuid4()

    now = datetime.now(timezone.utc)

    memory_a = MemoryRecord(
        id=memory_a_id,
        namespace="tenant-a",
        content="Tenant A secret",
        memory_type="semantic",
        metadata={},
        created_at=now,
        updated_at=now,
    )

    vector_results = [
        VectorRecord(
            id=str(memory_a_id),
            score=0.95,
            content=None,
            metadata={},
        ),
        VectorRecord(
            id=str(memory_b_id),
            score=0.94,
            content=None,
            metadata={},
        ),
    ]

    vector_provider = FakeVectorProvider(
        vector_results
    )

    memory_provider = FakeMemoryProvider(
        [memory_a]
    )

    strategy = VectorRetrievalStrategy(
        embedding_provider=FakeEmbeddingProvider(),
        vector_provider=vector_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="secret information",
            namespace="tenant-a",
            top_k=10,
        )
    )

    assert len(result.evidence) == 1

    assert (
        result.evidence[0].id
        == str(memory_a_id)
    )

    assert (
        result.evidence[0].content
        == "Tenant A secret"
    )

    assert (
        vector_provider.last_filters
        is not None
    )

    allowed_ids = (
        vector_provider.last_filters[
            "allowed_ids"
        ]
    )

    assert allowed_ids == [
        str(memory_a_id)
    ]

    assert (
        str(memory_b_id)
        not in allowed_ids
    )