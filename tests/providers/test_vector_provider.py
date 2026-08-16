import pytest
from packages.providers.vector import VectorProvider

class FakeVectorProvider(VectorProvider):
    async def initialize(self) -> None:
        pass
    
    async def upsert(
        self, *, namespace, record_id,
        embedding, content, metadata=None,
    ):
        return None

    async def search(
        self, *, namespace, embedding,
        top_k=10, filters=None,
    ):
        return []

    async def delete(
        self, *,
        namespace,
        record_id,
    ):
        return None


@pytest.mark.asyncio
async def test_vector_provider_contract():
    provider = FakeVectorProvider()

    await provider.upsert(
        namespace="test",
        record_id="1",
        embedding=[0.1, 0.2],
        content="hello",
    )

    results = await provider.search(
        namespace="test",
        embedding=[0.1, 0.2],
    )

    assert results == []

    await provider.delete(
        namespace="test",
        record_id="1",
    )