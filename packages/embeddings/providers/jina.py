from packages.core.config import settings
from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.client import get_jina_client
from packages.embeddings.schemas import EmbeddingResponse

class JinaEmbeddingProvider(BaseEmbeddingProvider):
    name = "jina"

    def __init__(self):
        self.client = get_jina_client()

    async def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        response = await self.client.post(
            "/embeddings", json={
                "model": settings.EMBEDDING_MODEL,
                "input": [text],
            },
        )

        response.raise_for_status()

        data = response.json()

        vector = data["data"][0]["embedding"]

        return EmbeddingResponse(
            embedding=vector, dimensions=len(vector),
            model=settings.EMBEDDING_MODEL, provider=self.name,
        )