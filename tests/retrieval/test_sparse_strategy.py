from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from packages.providers.memory import MemoryProvider, MemoryRecord
from packages.providers.sparse import SparseProvider, SparseRecord
from packages.retrieval import RetrievalQuery
from packages.retrieval.strategies import SparseRetrievalStrategy

class FakeSparseProvider(SparseProvider):
    def __init__(
        self, results: list[SparseRecord] | None = None,
    ) -> None:
        self.results = results or []
        self.calls: list[dict] = []

    async def initialize(self) -> None:
        pass

    async def upsert(
        self, *, namespace: str, record_id: str,
        content: str, metadata: dict | None = None,
    ) -> None:
        pass

    async def search(self, *, namespace: str, query: str, top_k: int = 10, filters: dict | None = None) -> list[SparseRecord]:
        self.calls.append({
            "namespace": namespace, 
            "query": query, "top_k": top_k, 
            "filters": filters,
        })

        return self.results

    async def delete(
        self, *, namespace: str, record_id: str,
    ) -> None:
        pass

class FakeMemoryProvider(MemoryProvider):
    def __init__(
        self, memories: dict[UUID, MemoryRecord] | None = None,
    ) -> None:
        if memories is None:
            self.memories = {}
        else:
            self.memories = memories

        self.last_filters: dict | None = None

    async def create(self, *, namespace: str, content: str, metadata: dict | None = None, memory_type: str) -> MemoryRecord:
        raise NotImplementedError

    async def get(
        self, memory_id: UUID, *, namespace: str,
    ) -> MemoryRecord | None:
        memory = self.memories.get(memory_id)

        if memory is None:
            return None

        if memory.namespace != namespace:
            return None

        return memory

    async def update(
        self, memory_id: UUID, *, namespace: str,
        content: str | None = None, metadata: dict | None = None,
    ) -> MemoryRecord:
        raise NotImplementedError

    async def delete(
        self, memory_id: UUID, *, namespace: str,
    ) -> None:
        raise NotImplementedError

    async def list_ids(
        self, *, namespace: str, filters: dict | None = None,
    ) -> list[UUID]:
        self.last_filters = filters
        authorized = [
            memory_id
            for memory_id, memory in self.memories.items()
            if memory.namespace == namespace
        ]

        if not filters or filters.get("allowed_ids") is None:
            return authorized

        allowed_ids = {
            str(memory_id) for memory_id in filters["allowed_ids"]
        }
        return [
            memory_id
            for memory_id in authorized
            if str(memory_id) in allowed_ids
        ]

def make_memory(
    *, memory_id: UUID, namespace: str = "test",
    content: str = "Canonical memory",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id, namespace=namespace, content=content,
        memory_type="semantic", metadata={"source": "postgres"},
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )

@pytest.mark.asyncio
async def test_sparse_strategy_retrieves_results():
    memory_id = uuid4()

    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id=str(memory_id), score=2.5, content="Indexed content",
                metadata={"source": "bm25"},
            )
        ]
    )

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
    )

    query = RetrievalQuery(
        query="PostgreSQL", 
        namespace="test", top_k=5,
    )

    result = await strategy.retrieve(query)

    assert len(result.evidence) == 1
    assert result.evidence[0].id == str(memory_id)
    assert result.evidence[0].score == 2.5
    assert result.evidence[0].source == "sparse"
    assert result.strategy == "sparse"
    assert result.latency_ms is not None

@pytest.mark.asyncio
async def test_sparse_strategy_passes_query_parameters():
    sparse_provider = FakeSparseProvider()

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
    )

    query = RetrievalQuery(
        query="TurboVecProvider",
        namespace="tenant-a",
        top_k=7,
        filters={
            "allowed_ids": [
                "memory-1",
                "memory-2",
            ]
        },
    )

    await strategy.retrieve(query)

    assert sparse_provider.calls == [
        {
            "namespace": "tenant-a",
            "query": "TurboVecProvider",
            "top_k": 7,
            "filters": {
                "allowed_ids": [
                    "memory-1",
                    "memory-2",
                ]
            },
        }
    ]


@pytest.mark.asyncio
async def test_sparse_strategy_hydrates_canonical_memory():
    memory_id = uuid4()

    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id=str(memory_id),
                score=3.0,
                content="STALE INDEX CONTENT",
                metadata={
                    "source": "bm25",
                },
            )
        ]
    )

    memory = make_memory(
        memory_id=memory_id,
        content="CANONICAL POSTGRES CONTENT",
    )

    memory_provider = FakeMemoryProvider(
        {
            memory_id: memory,
        }
    )

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="PostgreSQL",
            namespace="test",
        )
    )

    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.content == (
        "CANONICAL POSTGRES CONTENT"
    )

    assert evidence.metadata["source"] == "postgres"
    assert evidence.id == str(memory_id)


@pytest.mark.asyncio
async def test_sparse_strategy_discards_stale_index_entry():
    memory_id = uuid4()

    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id=str(memory_id),
                score=2.0,
                content="Stale memory",
            )
        ]
    )

    memory_provider = FakeMemoryProvider()

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="PostgreSQL",
            namespace="test",
        )
    )

    assert result.evidence == []


@pytest.mark.asyncio
async def test_sparse_strategy_discards_invalid_memory_id():
    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id="not-a-uuid",
                score=1.5,
                content="Invalid ID",
            )
        ]
    )

    memory_provider = FakeMemoryProvider()

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="PostgreSQL",
            namespace="test",
        )
    )

    assert result.evidence == []


@pytest.mark.asyncio
async def test_sparse_strategy_enforces_namespace_isolation():
    memory_id = uuid4()

    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id=str(memory_id),
                score=2.0,
                content="Tenant B memory",
            )
        ]
    )

    memory_provider = FakeMemoryProvider(
        {
            memory_id: make_memory(
                memory_id=memory_id,
                namespace="tenant-b",
            )
        }
    )

    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="database",
            namespace="tenant-a",
        )
    )

    assert result.evidence == []


@pytest.mark.asyncio
async def test_sparse_strategy_authorizes_candidates_before_search():
    memory_id = uuid4()
    memory_provider = FakeMemoryProvider(
        {memory_id: make_memory(memory_id=memory_id)}
    )
    sparse_provider = FakeSparseProvider()
    strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    await strategy.retrieve(
        RetrievalQuery(query="database", namespace="test")
    )

    assert sparse_provider.calls[0]["filters"] == {
        "allowed_ids": [str(memory_id)],
    }


@pytest.mark.asyncio
async def test_sparse_strategy_supports_empty_results():
    strategy = SparseRetrievalStrategy(
        sparse_provider=FakeSparseProvider(),
    )

    result = await strategy.retrieve(
        RetrievalQuery(
            query="nothing",
            namespace="test",
        )
    )

    assert result.evidence == []
    assert result.strategy == "sparse"
    assert result.latency_ms is not None
