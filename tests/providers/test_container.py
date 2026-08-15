from pathlib import Path
from unittest.mock import MagicMock

import pytest

from packages.providers.container import ProviderContainer
from packages.providers.vector import TurboVecProvider
from packages.providers.memory import PostgresMemoryProvider

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

    container = ProviderContainer()

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

    container = ProviderContainer(
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

    container = ProviderContainer(
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

    container = ProviderContainer(
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

    container = ProviderContainer(
        vector_provider=provider,
    )

    assert container.vector is provider

def test_container_requires_session_for_memory_provider():
    container = ProviderContainer()

    with pytest.raises(
        RuntimeError, match="AsyncSession is required",
    ):
        container.memory

def test_container_uses_injected_memory_provider():
    memory_provider = MagicMock()

    container = ProviderContainer(
        memory_provider=memory_provider,
    )

    assert container.memory is memory_provider


def test_container_creates_postgres_memory_provider():
    session = MagicMock()

    container = ProviderContainer(
        session=session,
    )

    provider = container.memory

    assert isinstance(
        provider,
        PostgresMemoryProvider,
    )


def test_container_creates_embedding_provider():
    container = ProviderContainer()

    provider = container.embedding

    assert provider is not None