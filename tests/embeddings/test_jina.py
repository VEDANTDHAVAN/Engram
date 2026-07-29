import pytest

from packages.embeddings.factory import EmbeddingFactory
from packages.embeddings.types import EmbeddingProvider

@pytest.mark.asyncio
async def test_embedding_generation():
    provider = EmbeddingFactory.create(
        EmbeddingProvider.JINA
    )

    response = await provider.embed(
        "Engram builds lifelong memory systems."
    )

    assert len(response.embedding) > 0
    assert response.dimensions == len(response.embedding)