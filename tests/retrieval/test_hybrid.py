from __future__ import annotations

from packages.retrieval.base import RetrievalStrategy
from packages.retrieval.fusion import (
    CandidateFusion,
    FusedCandidate,
)
from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.models import (
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
)


class FakeRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        name: str,
        evidence: list[RetrievalEvidence],
    ) -> None:
        self.name = name
        self.evidence = evidence
        self.calls = 0

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        self.calls += 1

        return RetrievalResult(
            evidence=self.evidence,
            strategy=self.name,
        )


class FakeFusion(CandidateFusion):
    def __init__(self) -> None:
        self.calls = 0
        self.received_results: list[RetrievalResult] = []

    def fuse(
        self,
        results: list[RetrievalResult],
    ) -> list[FusedCandidate]:
        self.calls += 1
        self.received_results = results

        if not results:
            return []

        return [
            FusedCandidate(
                id=evidence.id,
                content=evidence.content,
                score=evidence.score,
                sources=[result.strategy],
                metadata=evidence.metadata,
            )
            for result in results
            for evidence in result.evidence
        ]


def make_query() -> RetrievalQuery:
    return RetrievalQuery(
        query="distributed systems",
        namespace="test",
        top_k=10,
    )


def make_evidence(
    memory_id: str,
    content: str,
    score: float,
    source: str,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        id=memory_id,
        content=content,
        score=score,
        source=source,
    )


async def test_vector_only_retrieval() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [
            make_evidence(
                "memory-1",
                "Vector result",
                0.9,
                "vector",
            )
        ],
    )

    fusion = FakeFusion()

    service = HybridRetrievalService(
        vector_strategy=vector,
        weighted_fusion=fusion,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector"],
        fusion="weighted",
    )

    assert vector.calls == 1
    assert fusion.calls == 1

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "memory-1"
    assert result.strategy == "hybrid:vector:weighted"


async def test_sparse_only_retrieval() -> None:
    sparse = FakeRetrievalStrategy(
        "sparse",
        [
            make_evidence(
                "memory-2",
                "Sparse result",
                4.2,
                "sparse",
            )
        ],
    )

    fusion = FakeFusion()

    service = HybridRetrievalService(
        sparse_strategy=sparse,
        weighted_fusion=fusion,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["sparse"],
        fusion="weighted",
    )

    assert sparse.calls == 1
    assert fusion.calls == 1

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "memory-2"
    assert result.strategy == "hybrid:sparse:weighted"


async def test_vector_and_sparse_retrieval() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [
            make_evidence(
                "memory-1",
                "Vector result",
                0.9,
                "vector",
            )
        ],
    )

    sparse = FakeRetrievalStrategy(
        "sparse",
        [
            make_evidence(
                "memory-2",
                "Sparse result",
                3.5,
                "sparse",
            )
        ],
    )

    fusion = FakeFusion()

    service = HybridRetrievalService(
        vector_strategy=vector,
        sparse_strategy=sparse,
        rrf_fusion=fusion,
    )

    result = await service.retrieve(
        make_query(),
        strategies=["vector", "sparse"],
        fusion="rrf",
    )

    assert vector.calls == 1
    assert sparse.calls == 1
    assert fusion.calls == 1

    assert len(fusion.received_results) == 2
    assert len(result.evidence) == 2

    assert result.strategy == "hybrid:vector+sparse:rrf"


async def test_duplicate_strategy_is_executed_once() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [
            make_evidence(
                "memory-1",
                "Vector result",
                0.9,
                "vector",
            )
        ],
    )

    service = HybridRetrievalService(
        vector_strategy=vector,
    )

    await service.retrieve(
        make_query(),
        strategies=["vector", "vector"],
    )

    assert vector.calls == 1


async def test_all_configured_strategies_are_used_by_default() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [
            make_evidence(
                "memory-1",
                "Vector result",
                0.9,
                "vector",
            )
        ],
    )

    sparse = FakeRetrievalStrategy(
        "sparse",
        [
            make_evidence(
                "memory-2",
                "Sparse result",
                3.5,
                "sparse",
            )
        ],
    )

    fusion = FakeFusion()

    service = HybridRetrievalService(
        vector_strategy=vector,
        sparse_strategy=sparse,
        weighted_fusion=fusion,
    )

    await service.retrieve(
        make_query(),
        fusion="weighted",
    )

    assert vector.calls == 1
    assert sparse.calls == 1
    assert fusion.calls == 1


async def test_unknown_strategy_raises() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [],
    )

    service = HybridRetrievalService(
        vector_strategy=vector,
    )

    try:
        await service.retrieve(
            make_query(),
            strategies=["graph"],
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "graph" in str(exc)


async def test_unknown_fusion_raises() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [],
    )

    service = HybridRetrievalService(
        vector_strategy=vector,
    )

    try:
        await service.retrieve(
            make_query(),
            strategies=["vector"],
            fusion="unknown",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "unknown" in str(exc)


async def test_top_k_is_applied_after_fusion() -> None:
    vector = FakeRetrievalStrategy(
        "vector",
        [
            make_evidence(
                "memory-1",
                "one",
                0.9,
                "vector",
            ),
            make_evidence(
                "memory-2",
                "two",
                0.8,
                "vector",
            ),
            make_evidence(
                "memory-3",
                "three",
                0.7,
                "vector",
            ),
        ],
    )

    fusion = FakeFusion()

    service = HybridRetrievalService(
        vector_strategy=vector,
        weighted_fusion=fusion,
    )

    query = RetrievalQuery(
        query="test",
        namespace="test",
        top_k=2,
    )

    result = await service.retrieve(
        query,
        strategies=["vector"],
        fusion="weighted",
    )

    assert len(result.evidence) == 2