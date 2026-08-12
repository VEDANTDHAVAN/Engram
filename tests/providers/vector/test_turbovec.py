from pathlib import Path

import pytest

from packages.providers.vector import TurboVecProvider


# Small dimension keeps provider unit tests fast.
# Production CAML currently uses 1024-dimensional Jina embeddings.
TEST_DIMENSION = 8


def make_vector(
    first: float,
    second: float = 0.0,
) -> list[float]:
    """
    Create a deterministic vector with the dimension expected
    by the TurboVec test index.
    """

    return [
        first,
        second,
        *([0.0] * (TEST_DIMENSION - 2)),
    ]


@pytest.fixture
async def provider(tmp_path: Path) -> TurboVecProvider:
    """
    Create and initialize an isolated TurboVec provider.

    Each test receives its own temporary index.
    """

    vector_provider = TurboVecProvider(
        dimension=TEST_DIMENSION,
        index_path=tmp_path / "semantic.tvim",
        bit_width=4,
    )

    await vector_provider.initialize()

    return vector_provider


@pytest.mark.asyncio
async def test_turbovec_initialize(
    provider: TurboVecProvider,
):
    """
    Provider should initialize a TurboVec index successfully.
    """

    assert provider._index is not None


@pytest.mark.asyncio
async def test_turbovec_upsert_and_search(
    provider: TurboVecProvider,
):
    """
    A vector inserted into TurboVec should be retrievable
    through the CAML VectorProvider interface.
    """

    embedding = make_vector(1.0)

    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        embedding=embedding,
        content="PostgreSQL is used.",
        metadata={
            "memory_type": "semantic",
        },
    )

    results = await provider.search(
        namespace="test",
        embedding=embedding,
        top_k=1,
    )

    assert len(results) == 1

    assert results[0].id == "memory-1"

    assert results[0].score >= 0.0


@pytest.mark.asyncio
async def test_turbovec_batch_add(
    provider: TurboVecProvider,
):
    """
    Multiple vectors should be inserted through the
    batch_add operation.
    """

    vector_a = make_vector(1.0)

    vector_b = make_vector(
        0.99,
        0.01,
    )

    await provider.batch_add(
        namespace="test",
        records=[
            ("memory-1", vector_a),
            ("memory-2", vector_b),
        ],
    )

    results = await provider.search(
        namespace="test",
        embedding=vector_a,
        top_k=2,
    )

    assert len(results) == 2

    result_ids = {
        result.id
        for result in results
    }

    assert result_ids == {
        "memory-1",
        "memory-2",
    }


@pytest.mark.asyncio
async def test_turbovec_allowlist_filter(
    provider: TurboVecProvider,
):
    """
    TurboVec's native allowlist should restrict the candidate
    set before vector search results are returned.
    """

    vector_a = make_vector(1.0)

    vector_b = make_vector(
        0.99,
        0.01,
    )

    await provider.batch_add(
        namespace="test",
        records=[
            ("memory-1", vector_a),
            ("memory-2", vector_b),
        ],
    )

    results = await provider.search(
        namespace="test",
        embedding=vector_a,
        top_k=10,
        filters={
            "allowed_ids": [
                "memory-2",
            ],
        },
    )

    assert len(results) == 1

    assert results[0].id == "memory-2"


@pytest.mark.asyncio
async def test_turbovec_delete(
    provider: TurboVecProvider,
):
    """
    Deleted vectors should no longer appear in retrieval results.
    """

    embedding = make_vector(1.0)

    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        embedding=embedding,
        content="Temporary memory.",
    )

    results_before = await provider.search(
        namespace="test",
        embedding=embedding,
        top_k=10,
    )

    assert len(results_before) == 1

    await provider.delete(
        namespace="test",
        record_id="memory-1",
    )

    results_after = await provider.search(
        namespace="test",
        embedding=embedding,
        top_k=10,
    )

    assert results_after == []


@pytest.mark.asyncio
async def test_turbovec_rejects_wrong_dimension(
    provider: TurboVecProvider,
):
    """
    Embeddings with a dimension different from the configured
    index dimension must be rejected before reaching TurboVec.
    """

    invalid_embedding = [
        1.0,
        0.0,
        0.0,
    ]

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        await provider.upsert(
            namespace="test",
            record_id="memory-1",
            embedding=invalid_embedding,
            content="Invalid vector.",
        )


@pytest.mark.asyncio
async def test_turbovec_rejects_non_finite_values(
    provider: TurboVecProvider,
):
    """
    NaN and infinity must never be inserted into the vector index.
    """

    invalid_embedding = make_vector(1.0)

    invalid_embedding[3] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        await provider.upsert(
            namespace="test",
            record_id="memory-1",
            embedding=invalid_embedding,
            content="Invalid vector.",
        )


@pytest.mark.asyncio
async def test_turbovec_update(
    provider: TurboVecProvider,
):
    """
    Updating a vector should replace its existing vector
    while preserving the CAML record ID.
    """

    original_embedding = make_vector(1.0)

    updated_embedding = make_vector(
        0.0,
        1.0,
    )

    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        embedding=original_embedding,
        content="Original memory.",
    )

    await provider.update(
        namespace="test",
        record_id="memory-1",
        embedding=updated_embedding,
    )

    results = await provider.search(
        namespace="test",
        embedding=updated_embedding,
        top_k=1,
    )

    assert len(results) == 1

    assert results[0].id == "memory-1"


@pytest.mark.asyncio
async def test_turbovec_empty_search(
    provider: TurboVecProvider,
):
    """
    Searching an empty index should return no evidence.
    """

    embedding = make_vector(1.0)

    results = await provider.search(
        namespace="test",
        embedding=embedding,
        top_k=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_turbovec_empty_batch_add(
    provider: TurboVecProvider,
):
    """
    Empty batch insertion should be a no-op.
    """

    await provider.batch_add(
        namespace="test",
        records=[],
    )

    embedding = make_vector(1.0)

    results = await provider.search(
        namespace="test",
        embedding=embedding,
        top_k=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_turbovec_persist(
    provider: TurboVecProvider,
    tmp_path: Path,
):
    """
    Persisting the index should create the configured index file.
    """

    embedding = make_vector(1.0)

    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        embedding=embedding,
        content="Persistent memory.",
    )

    await provider.persist()

    index_path = (
        tmp_path / "semantic.tvim"
    )

    assert index_path.exists()


@pytest.mark.asyncio
async def test_turbovec_requires_initialization(
    tmp_path: Path,
):
    """
    Operations before initialize() should fail explicitly rather
    than implicitly creating an index.
    """

    provider = TurboVecProvider(
        dimension=TEST_DIMENSION,
        index_path=tmp_path / "semantic.tvim",
    )

    embedding = make_vector(1.0)

    with pytest.raises(
        Exception,
        match="initialize",
    ):
        await provider.search(
            namespace="test",
            embedding=embedding,
            top_k=1,
        )