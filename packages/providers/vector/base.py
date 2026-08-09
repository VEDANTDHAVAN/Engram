from __future__ import annotations

from abc import ABC, abstractmethod

from .types import VectorRecord

class VectorProvider(ABC):
    """
    Provider-neutral vector search interface.
    
    Implementations can use pgvector, Qdrant, Milvus, Pinecone, etc.
    """
    @abstractmethod
    async def upsert(
        self, *, namespace: str, record_id: str,
        embedding: list[float], content: str,
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, *, namespace: str,
        embedding: list[float], top_k: int = 10,
        filters: dict | None = None,
    ) -> list[VectorRecord]:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self, *,
        namespace: str,
        record_id: str,
    ) -> None:
        raise NotImplementedError