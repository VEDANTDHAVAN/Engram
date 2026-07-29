from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    """Abstract interface implemented by all embedding providers."""
    name: str
    dimensions: int

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single string."""
        raise NotImplementedError

    async def embed_batch(
        self, texts: list[str],
    ) -> list[list[float]]:
        """Default batch implementation."""
        return [
            await self.embed(text) for text in texts
        ]