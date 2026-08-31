from pathlib import Path

import pytest

from packages.providers.sparse import BM25Provider

@pytest.fixture
async def provider() -> BM25Provider:
    provider = BM25Provider()

    await provider.initialize()

    return provider

@pytest.mark.asyncio
async def test_bm25_initialize(
    provider: BM25Provider,
):
    assert provider._initialized is True

@pytest.mark.asyncio
async def test_bm25_upsert_and_search(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test", record_id="memory-1",
        content=(
            "PostgreSQL is the canonical database for CAML."
        ),
    )

    results = await provider.search(
        namespace="test", top_k=5,
        query="PostgreSQL database",
    )

    assert len(results) == 1
    assert results[0].id == "memory-1"
    assert results[0].score > 0.0

@pytest.mark.asyncio
async def test_bm25_ranks_relevant_document_higher(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content=(
            "TurboVec provides dense vector retrieval."
        ),
    )

    await provider.upsert(
        namespace="test",
        record_id="memory-2",
        content=(
            "BM25 provides sparse lexical retrieval "
            "for exact technical terms."
        ),
    )

    results = await provider.search(
        namespace="test",
        query="BM25 sparse lexical retrieval",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].id == "memory-2"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_bm25_namespace_isolation(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="tenant-a",
        record_id="memory-a",
        content="PostgreSQL database architecture",
    )

    await provider.upsert(
        namespace="tenant-b",
        record_id="memory-b",
        content="PostgreSQL database architecture",
    )

    results = await provider.search(
        namespace="tenant-a",
        query="PostgreSQL database",
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].id == "memory-a"


@pytest.mark.asyncio
async def test_bm25_allowlist_filter(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="PostgreSQL database",
    )

    await provider.upsert(
        namespace="test",
        record_id="memory-2",
        content="PostgreSQL database",
    )

    results = await provider.search(
        namespace="test",
        query="PostgreSQL",
        top_k=10,
        filters={
            "allowed_ids": ["memory-2"],
        },
    )

    assert len(results) == 1
    assert results[0].id == "memory-2"


@pytest.mark.asyncio
async def test_bm25_upsert_replaces_existing_record(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="TurboVec vector retrieval",
    )

    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="PostgreSQL database retrieval",
    )

    results = await provider.search(
        namespace="test",
        query="PostgreSQL database",
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].id == "memory-1"
    assert results[0].content == (
        "PostgreSQL database retrieval"
    )


@pytest.mark.asyncio
async def test_bm25_delete(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="Temporary memory",
    )

    await provider.delete(
        namespace="test",
        record_id="memory-1",
    )

    results = await provider.search(
        namespace="test",
        query="Temporary memory",
        top_k=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_bm25_empty_search(
    provider: BM25Provider,
):
    results = await provider.search(
        namespace="test",
        query="PostgreSQL",
        top_k=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_bm25_empty_query(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="PostgreSQL database",
    )

    results = await provider.search(
        namespace="test",
        query="",
        top_k=10,
    )

    assert results == []


@pytest.mark.asyncio
async def test_bm25_requires_initialization():
    provider = BM25Provider()

    with pytest.raises(
        RuntimeError,
        match="initialized",
    ):
        await provider.search(
            namespace="test",
            query="PostgreSQL",
            top_k=10,
        )


@pytest.mark.asyncio
async def test_bm25_preserves_metadata(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="PostgreSQL database",
        metadata={
            "memory_type": "semantic",
            "source": "documentation",
        },
    )

    results = await provider.search(
        namespace="test",
        query="PostgreSQL",
        top_k=1,
    )

    assert results[0].metadata == {
        "memory_type": "semantic",
        "source": "documentation",
    }


@pytest.mark.asyncio
async def test_bm25_rejects_invalid_top_k(
    provider: BM25Provider,
):
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        await provider.search(
            namespace="test",
            query="PostgreSQL",
            top_k=0,
        )


@pytest.mark.asyncio
async def test_bm25_rejects_invalid_allowlist(
    provider: BM25Provider,
):
    await provider.upsert(
        namespace="test",
        record_id="memory-1",
        content="PostgreSQL database",
    )

    with pytest.raises(
        ValueError,
        match="allowed_ids",
    ):
        await provider.search(
            namespace="test",
            query="PostgreSQL",
            filters={
                "allowed_ids": "memory-1",
            },
        )