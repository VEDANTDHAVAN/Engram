from uuid import UUID, uuid4
from datetime import datetime, timezone

import pytest

from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.ingestion import IngestionPipeline, IngestionService, RawDocument
from packages.providers.memory import MemoryProvider, MemoryRecord
from packages.providers.vector import VectorProvider, VectorRecord

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    async def embed(self, text: str, **kwargs):
        return EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3], dimensions=3,
            model="fake", provider="fake",
        )

class FakeMemoryProvider(MemoryProvider):
    def __init__(self):
        self.records = {}

    async def create(
        self, *, namespace, content,
        memory_type, metadata=None,
    ):
        memory_id = uuid4()
        now = datetime.now(timezone.utc)

        record = MemoryRecord(
            id=memory_id, namespace=namespace, content=content,
            memory_type=memory_type, metadata=metadata or {},
            created_at=now, updated_at=now,
        )

        self.records[memory_id] = record

        return record

    async def get(
        self, memory_id: UUID, *, namespace: str,
    ):
        return self.records.get(memory_id)

    async def update(
        self, memory_id: UUID, *, namespace: str,
        content: None, metadata=None,
    ):
        record = self.records[memory_id]

        if content is not None:
            record.content = content

        if metadata is not None:
            record.metadata = metadata

        return record

    async def delete(
        self, memory_id: UUID, *, namespace: str,
    ):
        self.records.pop(memory_id, None)

    async def list_ids(self, *, namespace: str, filters: dict | None = None) -> list[UUID]:
        return []

class FakeVectorProvider(VectorProvider):
    def __init__(self):
        self.records = {}

    async def initialize(self) -> None:
        pass

    async def upsert(
        self, *, namespace, record_id,
        embedding, content, metadata=None,
    ):
        self.records[record_id] = VectorRecord(
            id=record_id, score=1.0, content=content,
            metadata=metadata or {},
        )

    async def search(
        self, *, namespace, embedding,
        top_k=10, filters=None,
    ):
        return list(self.records.values())[:top_k]

    async def delete(
        self, *, namespace, record_id,
    ):
        self.records.pop(record_id, None)

@pytest.mark.asyncio
async def test_ingestion_service():
    memory_provider = FakeMemoryProvider()
    vector_provider = FakeVectorProvider()
    embedding_provider = FakeEmbeddingProvider()

    service = IngestionService(
        pipeline=IngestionPipeline(),
        embedding_provider=embedding_provider,
        memory_provider=memory_provider,
        vector_provider=vector_provider,
    )

    document = RawDocument(
        source="test", content="CAML provides context engineering for AI Agents.",
    )

    result = await service.ingest(
        document, namespace="test-tenant",
    )

    assert result.total_chunks == 1
    assert result.total_embeddings == 1

    assert len(memory_provider.records) == 1
    assert len(vector_provider.records) == 1