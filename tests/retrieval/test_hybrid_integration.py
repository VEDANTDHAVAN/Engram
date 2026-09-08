from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.providers.memory import MemoryProvider, MemoryRecord
from packages.providers.sparse import SparseProvider, SparseRecord
from packages.providers.vector import VectorProvider, VectorRecord
from packages.retrieval.fusion import (
    ReciprocalRankFusion,
    WeightedCandidateFusion,
)
from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.models import RetrievalQuery
from packages.retrieval.strategies.sparse import SparseRetrievalStrategy
from packages.retrieval.strategies.vector import VectorRetrievalStrategy


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    name = "fake"
    dimensions = 4

    async def embed(self, text: str) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=[1.0, 0.0, 0.0, 0.0],
            dimensions=4,
            model="fake",
            provider="fake",
        )


class FakeVectorProvider(VectorProvider):
    def __init__(self, records: list[VectorRecord]) -> None:
        self.records = records

    async def initialize(self) -> None:
        pass

    async def search(
        self,
        *,
        namespace: str,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[VectorRecord]:
        allowed_ids = None

        if filters:
            allowed_ids = filters.get("allowed_ids")

        records = self.records

        if allowed_ids is not None:
            records = [
                record
                for record in records
                if record.id in allowed_ids
            ]

        return records[:top_k]

    async def upsert(
        self,
        *,
        namespace: str,
        record_id: str,
        embedding: list[float],
        content: str,
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError

    async def delete(
        self,
        *,
        namespace: str,
        record_id: str,
    ) -> None:
        raise NotImplementedError


class FakeSparseProvider(SparseProvider):
    def __init__(self, records: list[SparseRecord]) -> None:
        self.records = records

    async def initialize(self) -> None:
        pass

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SparseRecord]:
        allowed_ids = None

        if filters:
            allowed_ids = filters.get("allowed_ids")

        records = self.records

        if allowed_ids is not None:
            records = [
                record
                for record in records
                if record.id in allowed_ids
            ]

        return records[:top_k]

    async def upsert(
        self,
        *,
        namespace: str,
        record_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError

    async def delete(
        self,
        *,
        namespace: str,
        record_id: str,
    ) -> None:
        raise NotImplementedError


class FakeMemoryProvider(MemoryProvider):
    def __init__(self, memories: dict[str, MemoryRecord]) -> None:
        self.memories = memories

    async def get(
        self,
        memory_id: UUID,
        *,
        namespace: str,
    ) -> MemoryRecord | None:
        memory = self.memories.get(str(memory_id))

        if memory is None:
            return None

        if memory.namespace != namespace:
            return None

        return memory

    async def create(
        self,
        *,
        namespace: str,
        content: str,
        metadata: dict | None = None,
        memory_type: str,
    ) -> MemoryRecord:
        raise NotImplementedError

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

    async def list_ids( self, *, namespace: str, filters: dict | None = None, ) -> list[UUID]: 
        memories = [ 
            memory for memory in self.memories.values() 
            if memory.namespace == namespace 
        ] 
        allowed_ids: set[str] | None = None
        if filters: allowed_ids = filters.get("allowed_ids") 
        if allowed_ids is not None: memories = [
            memory for memory in memories 
            if str(memory.id) in allowed_ids or memory.id in allowed_ids
        ] 

        return [memory.id for memory in memories]


def make_query(
    *,
    top_k: int = 10,
    namespace: str = "test",
) -> RetrievalQuery:
    return RetrievalQuery(
        query="distributed systems",
        namespace=namespace,
        top_k=top_k,
    )


def build_components():
    memory_id_1 = uuid4()
    memory_id_2 = uuid4()
    memory_id_3 = uuid4()

    now = datetime.now(timezone.utc)

    memories = {
        str(memory_id_1): MemoryRecord(
            id=memory_id_1,
            namespace="test",
            content="Vector and sparse shared result",
            memory_type="semantic",
            metadata={},
            created_at=now,
            updated_at=now,
        ),
        str(memory_id_2): MemoryRecord(
            id=memory_id_2,
            namespace="test",
            content="Vector-only result",
            memory_type="semantic",
            metadata={},
            created_at=now,
            updated_at=now,
        ),
        str(memory_id_3): MemoryRecord(
            id=memory_id_3,
            namespace="test",
            content="Sparse-only result",
            memory_type="semantic",
            metadata={},
            created_at=now,
            updated_at=now,
        ),
    }

    memory_provider = FakeMemoryProvider(memories)

    vector_provider = FakeVectorProvider(
        [
            VectorRecord(
                id=str(memory_id_1),
                score=0.95,
            ),
            VectorRecord(
                id=str(memory_id_2),
                score=0.85,
            ),
        ]
    )

    sparse_provider = FakeSparseProvider(
        [
            SparseRecord(
                id=str(memory_id_1),
                score=4.5,
            ),
            SparseRecord(
                id=str(memory_id_3),
                score=3.2,
            ),
        ]
    )

    embedding_provider = FakeEmbeddingProvider()

    vector_strategy = VectorRetrievalStrategy(
        embedding_provider=embedding_provider,
        vector_provider=vector_provider,
        memory_provider=memory_provider,
    )

    sparse_strategy = SparseRetrievalStrategy(
        sparse_provider=sparse_provider,
        memory_provider=memory_provider,
    )

    return (
        vector_strategy,
        sparse_strategy,
        memory_id_1,
        memory_id_2,
        memory_id_3,
    )


async def test_real_strategies_combine_vector_and_sparse_results():
    (
        vector_strategy,
        sparse_strategy,
        memory_id_1,
        memory_id_2,
        memory_id_3,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    ids = {evidence.id for evidence in result.evidence}

    assert str(memory_id_1) in ids
    assert str(memory_id_2) in ids
    assert str(memory_id_3) in ids

    assert result.strategy == "hybrid:vector+sparse:rrf"


async def test_shared_candidate_is_deduplicated_by_rrf():
    (
        vector_strategy,
        sparse_strategy,
        memory_id_1,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    matching = [
        evidence
        for evidence in result.evidence
        if evidence.id == str(memory_id_1)
    ]

    assert len(matching) == 1


async def test_shared_candidate_contains_both_sources():
    (
        vector_strategy,
        sparse_strategy,
        memory_id_1,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    shared = next(
        evidence
        for evidence in result.evidence
        if evidence.id == str(memory_id_1)
    )

    assert "vector" in shared.source
    assert "sparse" in shared.source


async def test_weighted_fusion_works_with_real_strategies():
    (
        vector_strategy,
        sparse_strategy,
        _,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
        weighted_fusion=WeightedCandidateFusion(
            weights={
                "vector": 0.5,
                "sparse": 0.5,
            }
        ),
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="weighted",
    )

    assert len(result.evidence) == 3
    assert result.strategy == "hybrid:vector+sparse:weighted"


async def test_rrf_works_with_real_strategies():
    (
        vector_strategy,
        sparse_strategy,
        _,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
        rrf_fusion=ReciprocalRankFusion(k=60),
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    assert len(result.evidence) == 3
    assert result.strategy == "hybrid:vector+sparse:rrf"


async def test_top_k_is_applied_after_real_fusion():
    (
        vector_strategy,
        sparse_strategy,
        _,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(top_k=2),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    assert len(result.evidence) == 2


async def test_canonical_memory_content_is_preserved():
    (
        vector_strategy,
        sparse_strategy,
        memory_id_1,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    evidence = next(
        evidence
        for evidence in result.evidence
        if evidence.id == str(memory_id_1)
    )

    assert evidence.content == "Vector and sparse shared result"


async def test_namespace_isolation_survives_hybrid_retrieval():
    (
        vector_strategy,
        sparse_strategy,
        _,
        _,
        _,
    ) = build_components()

    service = HybridRetrievalService(
        vector_strategy=vector_strategy,
        sparse_strategy=sparse_strategy,
    )

    result = await service.retrieve(
        make_query(namespace="different-namespace"),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    assert result.evidence == []
