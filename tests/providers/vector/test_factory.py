import pytest

from uuid import UUID
from packages.core.config import settings
from packages.providers.vector import (
    TurboVecProvider,
    VectorFactory,
)
from packages.providers.memory import MemoryProvider, MemoryRecord
from packages.providers.container import ProviderContainer

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

def test_factory_creates_turbovec_provider():
    provider = VectorFactory.create(
        provider="turbovec",
    )

    assert isinstance(
        provider,
        TurboVecProvider,
    )


def test_factory_provider_name_is_case_insensitive():
    provider = VectorFactory.create(
        provider="TURBOVEC",
    )

    assert isinstance(
        provider,
        TurboVecProvider,
    )


def test_factory_provider_name_strips_whitespace():
    provider = VectorFactory.create(
        provider="  turbovec  ",
    )

    assert isinstance(
        provider,
        TurboVecProvider,
    )


def test_factory_rejects_unknown_provider():
    with pytest.raises(
        ValueError,
        match="Unsupported vector provider",
    ):
        VectorFactory.create(
            provider="unknown",
        )

def test_factory_uses_configured_provider(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "VECTOR_PROVIDER",
        "turbovec",
    )

    provider = VectorFactory.create()

    assert isinstance(
        provider,
        TurboVecProvider,
    )

def test_factory_explicit_provider_overrides_configuration(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "VECTOR_PROVIDER",
        "unknown",
    )

    provider = VectorFactory.create(
        provider="turbovec",
    )

    assert isinstance(
        provider,
        TurboVecProvider,
    )

@pytest.mark.asyncio
async def test_container_uses_factory_when_vector_provider_not_injected(
    monkeypatch,
    tmp_path,
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

    assert provider is container.vector