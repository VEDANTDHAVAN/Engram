from packages.retrieval.models import (
    RetrievalEvidence,
    RetrievalResult,
)
from packages.retrieval.fusion import (
    WeightedCandidateFusion,
)


def evidence(
    *,
    record_id: str,
    score: float,
    source: str,
    content: str = "memory",
) -> RetrievalEvidence:
    return RetrievalEvidence(
        id=record_id,
        content=content,
        score=score,
        source=source,
    )


def test_weighted_fusion_combines_results():
    fusion = WeightedCandidateFusion()

    vector_result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.9,
                source="vector",
            ),
        ],
    )

    sparse_result = RetrievalResult(
        strategy="sparse",
        evidence=[
            evidence(
                record_id="memory-1",
                score=5.0,
                source="sparse",
            ),
        ],
    )

    results = fusion.fuse(
        [
            vector_result,
            sparse_result,
        ]
    )

    assert len(results) == 1

    candidate = results[0]

    assert candidate.id == "memory-1"
    assert set(candidate.sources) == {
        "vector",
        "sparse",
    }


def test_weighted_fusion_applies_strategy_weights():
    fusion = WeightedCandidateFusion(
        weights={
            "vector": 2.0,
            "sparse": 1.0,
        }
    )

    vector_result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=1.0,
                source="vector",
            ),
        ],
    )

    sparse_result = RetrievalResult(
        strategy="sparse",
        evidence=[
            evidence(
                record_id="memory-1",
                score=1.0,
                source="sparse",
            ),
        ],
    )

    results = fusion.fuse(
        [
            vector_result,
            sparse_result,
        ]
    )

    assert results[0].score == 3.0


def test_weighted_fusion_deduplicates_candidates():
    fusion = WeightedCandidateFusion()

    result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.9,
                source="vector",
            ),
            evidence(
                record_id="memory-1",
                score=0.8,
                source="vector",
            ),
        ],
    )

    results = fusion.fuse([result])

    assert len(results) == 1


def test_weighted_fusion_ranks_candidates():
    fusion = WeightedCandidateFusion()

    result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.9,
                source="vector",
            ),
            evidence(
                record_id="memory-2",
                score=0.1,
                source="vector",
            ),
        ],
    )

    results = fusion.fuse([result])

    assert results[0].id == "memory-1"
    assert results[1].id == "memory-2"


def test_weighted_fusion_handles_empty_results():
    fusion = WeightedCandidateFusion()

    assert fusion.fuse([]) == []


def test_weighted_fusion_ignores_zero_weight_strategy():
    fusion = WeightedCandidateFusion(
        weights={
            "vector": 0.0,
            "sparse": 1.0,
        }
    )

    vector_result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="vector-memory",
                score=1.0,
                source="vector",
            )
        ],
    )

    sparse_result = RetrievalResult(
        strategy="sparse",
        evidence=[
            evidence(
                record_id="sparse-memory",
                score=1.0,
                source="sparse",
            )
        ],
    )

    results = fusion.fuse(
        [
            vector_result,
            sparse_result,
        ]
    )

    assert len(results) == 1
    assert results[0].id == "sparse-memory"