from __future__ import annotations

from uuid import uuid4

from packages.embeddings.base import BaseEmbeddingProvider
from packages.providers.memory import MemoryProvider
from packages.providers.vector import VectorProvider

from .pipeline import IngestionPipeline
from .schemas import RawDocument, ProcessedChunk, ServiceIngestionResult

class IngestionService:
    """
    Application service responsible for turning raw content into persistent CAML memories.

    The service knows about provider interfaces, not concrete storage implementations.
    """
    def __init__(
        self, *, pipeline: IngestionPipeline,
        embedding_provider: BaseEmbeddingProvider,
        memory_provider: MemoryProvider, vector_provider: VectorProvider,
    ):
        self.pipeline = pipeline
        self.embedding_provider = embedding_provider
        self.memory_provider = memory_provider
        self.vector_provider = vector_provider

    async def ingest(
        self, document: RawDocument,
        *, namespace: str,
    ) -> ServiceIngestionResult:
        result = self.pipeline.run(document)
        document_id = document.id or uuid4()

        processed_chunks: list[ProcessedChunk] = []

        for chunk in result.chunks:
            # 1. Generate embedding
            embedding_result = await self.embedding_provider.embed(chunk.text)

            # 2. Persist memory
            memory = await self.memory_provider.create(
                namespace=namespace, content=chunk.text,
                memory_type="semantic", metadata={
                    "source": document.source, "url": document.url,
                    "title": document.title, "chunk_index": chunk.index,
                    "token_count": chunk.token_count,
                },
            )

            # 3. Persist vector
            vector_id = str(memory.id)

            await self.vector_provider.upsert(
                namespace=namespace, record_id=vector_id,
                embedding=embedding_result.embedding,
                content=chunk.text, metadata={
                    "memory_id": str(memory.id),
                    "source": document.source,
                    "chunk_index": chunk.index,
                },
            )

            processed_chunks.append(
                ProcessedChunk(
                    chunk_index=chunk.index, memory_id=memory.id,
                    vector_id=vector_id, token_count=chunk.token_count,
                )
            )

        return ServiceIngestionResult(
            document_id=document_id, namespace=namespace,
            chunks=processed_chunks, total_chunks=len(processed_chunks),
            total_embeddings=len(processed_chunks),
        )