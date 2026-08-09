from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from .types import MemoryRecord

class MemoryProvider(ABC):
    """
    Storage abstraction for persistent memories.

    Implementations may use PostgreSQL, another relational database,
    or a future storage backend.
    """
    @abstractmethod
    async def create(
        self, *, namespace: str, content: str, 
        metadata: dict | None = None, memory_type: str,  
    ) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    async def get(self, memory_id: UUID, *, namespace: str) -> MemoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def update(
        self, memory_id: UUID, *, namespace: str,
        content: str | None = None, metadata: dict | None = None,
    ) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, memory_id: UUID, *, namespace: str) -> None:
        raise NotImplementedError