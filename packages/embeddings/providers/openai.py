from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"
    dimensions = 1024

    async def embed(
        self,
        text: str,
    ) -> EmbeddingResponse:
        raise NotImplementedError(
            "OpenAI provider not implemented yet."
        )