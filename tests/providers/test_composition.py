from unittest.mock import MagicMock

from packages.providers.container import ProviderContainer
from packages.providers.memory import PostgresMemoryProvider

def test_container_composes_caml_providers():
    session = MagicMock()

    container = ProviderContainer(session=session)

    embedding_provider = container.embedding
    memory_provider = container.memory
    vector_provider = container.vector

    assert embedding_provider is not None

    assert isinstance(
        memory_provider, PostgresMemoryProvider,
    )

    assert vector_provider is not None