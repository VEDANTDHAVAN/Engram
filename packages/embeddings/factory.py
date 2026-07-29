from packages.embeddings.providers.jina import JinaEmbeddingProvider
from packages.embeddings.types import EmbeddingProvider

class EmbeddingFactory:
    _providers = {
        EmbeddingProvider.JINA: JinaEmbeddingProvider,
    }

    @classmethod
    def create(cls, provider: EmbeddingProvider):
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provisder: {provider}")

        return cls._providers[provider]()