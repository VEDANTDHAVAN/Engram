import asyncio
  
from packages.embeddings import (
    EmbeddingFactory, EmbeddingProvider,
)

async def main():
    provider = EmbeddingFactory.create(EmbeddingProvider.JINA)

    result = await provider.embed("Hello from Engram.")

    print(result.dimensions)
    print(result.embedding[:10])


asyncio.run(main())