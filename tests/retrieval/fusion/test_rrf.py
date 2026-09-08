from packages.retrieval.fusion import ReciprocalRankFusion
from packages.retrieval.models import (
    RetrievalEvidence,
    RetrievalResult,
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


def test_rrf_combines_ranked_results():
    fusion = ReciprocalRankFusion(k=60)

    vector = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.9,
                source="vector",
            ),
            evidence(
                record_id="memory-2",
                score=0.8,
                source="vector",
            ),
        ],
    )

    sparse = RetrievalResult(
        strategy="sparse",
        evidence=[
            evidence(
                record_id="memory-2",
                score=5.0,
                source="sparse",
            ),
            evidence(
                record_id="memory-1",
                score=4.0,
                source="sparse",
            ),
        ],
    )

    results = fusion.fuse(
        [vector, sparse]
    )

    assert len(results) == 2

    # Both candidates appear in both ranked lists.
    assert {
        candidate.id
        for candidate in results
    } == {
        "memory-1",
        "memory-2",
    }


def test_rrf_score_is_based_on_rank():
    fusion = ReciprocalRankFusion(k=60)

    result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.1,
                source="vector",
            ),
            evidence(
                record_id="memory-2",
                score=0.9,
                source="vector",
            ),
        ],
    )

    results = fusion.fuse([result])

    assert results[0].id == "memory-1"

    assert results[0].score == 1 / 61
    assert results[1].score == 1 / 62


def test_rrf_rewards_cross_strategy_agreement():
    fusion = ReciprocalRankFusion(k=60)

    vector = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=0.9,
                source="vector",
            ),
            evidence(
                record_id="memory-2",
                score=0.8,
                source="vector",
            ),
        ],
    )

    sparse = RetrievalResult(
        strategy="sparse",
        evidence=[
            evidence(
                record_id="memory-2",
                score=5.0,
                source="sparse",
            ),
            evidence(
                record_id="memory-1",
                score=4.0,
                source="sparse",
            ),
        ],
    )

    results = fusion.fuse(
        [vector, sparse]
    )

    # Each candidate appears at ranks 1+2, so they tie.
    assert results[0].score == results[1].score


def test_rrf_preserves_sources():
    fusion = ReciprocalRankFusion()

    result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=1.0,
                source="vector",
            )
        ],
    )

    results = fusion.fuse([result])

    assert results[0].sources == ["vector"]


def test_rrf_deduplicates_candidates():
    fusion = ReciprocalRankFusion()

    result = RetrievalResult(
        strategy="vector",
        evidence=[
            evidence(
                record_id="memory-1",
                score=1.0,
                source="vector",
            ),
            evidence(
                record_id="memory-1",
                score=0.5,
                source="vector",
            ),
        ],
    )

    results = fusion.fuse([result])

    assert len(results) == 1

    # A candidate contributes only its first rank within
    # a retrieval strategy.
    assert results[0].score == 1 / 61


def test_rrf_handles_empty_results():
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([]) == []


def test_rrf_handles_empty_strategy_result():
    fusion = ReciprocalRankFusion()

    result = RetrievalResult(
        strategy="vector",
        evidence=[],
    )

    assert fusion.fuse([result]) == []


def test_rrf_rejects_invalid_k():
    try:
        ReciprocalRankFusion(k=0)
        assert False
    except ValueError as exc:
        assert "k" in str(exc)