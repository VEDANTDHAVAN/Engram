from packages.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"
    dimensions = 1024

    async def embed(
        self,
        text: str,
    ) -> list[float]:
        raise NotImplementedError(
            "OpenAI provider not implemented yet."
        )