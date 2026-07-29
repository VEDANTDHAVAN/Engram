from packages.embeddings.factory import EmbeddingFactory
from packages.embeddings.providers.jina import JinaEmbeddingProvider
from packages.embeddings.types import EmbeddingProvider

def test_factory_returns_jina():
    provider = EmbeddingFactory.create(
        EmbeddingProvider.JINA
    )

    assert isinstance(provider, JinaEmbeddingProvider)