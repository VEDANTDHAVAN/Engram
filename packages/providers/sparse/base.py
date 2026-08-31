from __future__ import annotations
from abc import ABC, abstractmethod
from .types import SparseRecord

class SparseProvider(ABC):
    """
    Provider-neutral sparse retrieval interface.

    Implementations may use BM25, Tantivy, Elasticsearch, OpenSearch, or another lexical retrieval backend.

    PostgreSQL remains the canonical source of durable memory.
    """
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the sparse retrieval backend."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self, *, namespace: str, record_id: str,
        content: str, metadata: dict | None = None,
    ) -> None:
        """Insert or replace a searchable document."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, *, namespace: str, query: str,
        top_k: int = 10, filters: dict | None = None,
    ) -> list[SparseRecord]:
        """Perform lexical/sparse retrieval."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self, *, namespace: str, record_id: str,
    ) -> None:
        """Remove a document from the sparse index."""
        raise NotImplementedError