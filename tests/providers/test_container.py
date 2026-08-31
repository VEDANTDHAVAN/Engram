from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from packages.providers.container import ProviderContainer
from packages.providers.vector import TurboVecProvider
from packages.providers.memory import PostgresMemoryProvider, MemoryProvider, MemoryRecord
from packages.providers.sparse import BM25Provider, SparseProvider

class FakeMemoryProvider(MemoryProvider):
    async def create(self, *, namespace: str, content: str, metadata: dict | None = None, memory_type: str) -> MemoryRecord:
        raise NotImplementedError

    async def get(self, memory_id: UUID, *, namespace: str) -> MemoryRecord | None:
        return None

    async def update(self, memory_id: UUID, *, namespace: str, content: str | None = None, metadata: dict | None = None) -> MemoryRecord:
        raise NotImplementedError

    async def delete(self, memory_id: UUID, *, namespace: str) -> None:
        raise NotImplementedError

    async def list_ids(self, *, namespace: str, filters: dict | None = None) -> list[UUID]:
        return []

def make_container(**kwargs):
    return ProviderContainer(
        memory_provider=FakeMemoryProvider(),
        **kwargs,
    )

@pytest.mark.asyncio
async def test_container_creates_configured_vector_provider(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        "packages.providers.vector.factory.settings.VECTOR_PROVIDER",
        "turbovec",
    )

    monkeypatch.setattr(
        "packages.providers.vector.factory.settings.TURBOVEC_INDEX_PATH",
        str(tmp_path / "semantic.tvim"),
    )

    monkeypatch.setattr(
        "packages.providers.vector.factory.settings.EMBEDDING_DIMENSIONS",
        8,
    )

    container = make_container()

    provider = container.vector

    assert isinstance(
        provider,
        TurboVecProvider,
    )


@pytest.mark.asyncio
async def test_container_initializes_vector_provider(
    tmp_path: Path,
):
    provider = TurboVecProvider(
        dimension=8,
        index_path=tmp_path / "semantic.tvim",
    )

    container = make_container(
        vector_provider=provider,
    )

    assert provider._index is None
    assert container._initialized is False

    await container.initialize()

    assert provider._index is not None
    assert container._initialized is True


@pytest.mark.asyncio
async def test_container_initialization_is_idempotent(
    tmp_path: Path,
):
    provider = TurboVecProvider(
        dimension=8,
        index_path=tmp_path / "semantic.tvim",
    )

    container = make_container(
        vector_provider=provider,
    )

    await container.initialize()

    first_index = provider._index

    await container.initialize()

    second_index = provider._index

    assert first_index is second_index
    assert container._initialized is True


@pytest.mark.asyncio
async def test_container_can_be_shutdown(
    tmp_path: Path,
):
    provider = TurboVecProvider(
        dimension=8,
        index_path=tmp_path / "semantic.tvim",
    )

    container = make_container(
        vector_provider=provider,
    )

    await container.initialize()

    assert container._initialized is True

    await container.shutdown()

    assert container._initialized is False


@pytest.mark.asyncio
async def test_container_uses_injected_provider(
    tmp_path: Path,
):
    """
    Dependency injection must take precedence over factory
    construction.
    """

    provider = TurboVecProvider(
        dimension=8,
        index_path=tmp_path / "semantic.tvim",
    )

    container = make_container(
        vector_provider=provider,
    )

    assert container.vector is provider

def test_container_requires_session_for_memory_provider():
    container = ProviderContainer(
        memory_provider=None,
        session=None,
    )

    with pytest.raises(
        RuntimeError,
        match="AsyncSession is required",
    ):
        _ = container.memory


def test_container_uses_injected_memory_provider():
    memory_provider = MagicMock()

    container = ProviderContainer(
        memory_provider=memory_provider,
    )

    assert container.memory is memory_provider


def test_container_creates_postgres_memory_provider():
    session = MagicMock()

    container = ProviderContainer(
        memory_provider=None,
        session=session,
    )

    provider = container.memory

    assert isinstance(
        provider,
        PostgresMemoryProvider,
    )

    assert provider.session is session


def test_container_creates_embedding_provider():
    container = ProviderContainer(
        memory_provider=FakeMemoryProvider(),
    )

    provider = container.embedding

    assert provider is not None


def test_container_creates_default_sparse_provider():
    container = make_container()

    assert isinstance(container.sparse, BM25Provider)
    assert container.sparse is container.sparse


def test_container_uses_injected_sparse_provider():
    sparse_provider = MagicMock(spec=SparseProvider)
    container = make_container(sparse_provider=sparse_provider)

    assert container.sparse is sparse_provider


@pytest.mark.asyncio
async def test_container_initializes_sparse_provider():
    sparse_provider = MagicMock(spec=SparseProvider)
    sparse_provider.initialize = AsyncMock()
    vector_provider = MagicMock()
    vector_provider.initialize = AsyncMock()

    container = make_container(
        sparse_provider=sparse_provider,
        vector_provider=vector_provider,
    )

    await container.initialize()

    sparse_provider.initialize.assert_awaited_once()
