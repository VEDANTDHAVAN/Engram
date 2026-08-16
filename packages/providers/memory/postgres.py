from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from packages.database.models.memory import Memory
from packages.database.repositories.memory import MemoryRepository
from packages.providers.memory.base import MemoryProvider
from packages.providers.memory.exceptions import MemoryNotFoundError
from packages.providers.memory.types import MemoryRecord


class PostgresMemoryProvider(MemoryProvider):
    """
    PostgreSQL adapter for CAML's MemoryProvider.

    PostgreSQL is the canonical source of durable memory.

    Vector retrieval is intentionally handled by VectorProvider
    implementations such as TurboVecProvider.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session
        self.repository = MemoryRepository(session)

    async def create(
        self,
        *,
        namespace: str,
        content: str,
        metadata: dict | None = None,
        memory_type: str,
    ) -> MemoryRecord:
        """
        Create a durable memory.

        The memory type is currently stored inside the existing
        metadata JSON rather than introducing another database
        column.
        """

        memory_metadata = dict(metadata or {})

        memory_metadata.setdefault(
            "memory_type",
            memory_type,
        )

        memory = Memory(
            namespace=namespace,
            content=content,
            metadata_=memory_metadata,
        )

        await self.repository.create(memory)

        await self.session.commit()
        await self.session.refresh(memory)

        return self._to_record(memory)

    async def get(
        self,
        memory_id: UUID,
        *,
        namespace: str,
    ) -> MemoryRecord | None:
        """
        Retrieve a memory while enforcing namespace isolation.
        """

        memory = await self.repository.get(memory_id)

        if memory is None:
            return None

        if memory.namespace != namespace:
            return None

        return self._to_record(memory)

    async def update(
        self,
        memory_id: UUID,
        *,
        namespace: str,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """
        Update canonical memory state.

        Vector index synchronization is deliberately handled
        outside this persistence adapter.
        """

        memory = await self.repository.get(memory_id)

        if memory is None:
            raise MemoryNotFoundError(
                f"Memory {memory_id} was not found."
            )

        if memory.namespace != namespace:
            raise MemoryNotFoundError(
                f"Memory {memory_id} was not found."
            )

        if content is not None:
            memory.content = content

        if metadata is not None:
            current_metadata = dict(
                memory.metadata_ or {}
            )

            current_metadata.update(metadata)

            memory.metadata_ = current_metadata

        memory.updated_at = datetime.now(
            timezone.utc
        )

        await self.session.commit()
        await self.session.refresh(memory)

        return self._to_record(memory)

    async def delete(
        self,
        memory_id: UUID,
        *,
        namespace: str,
    ) -> None:
        """
        Delete canonical memory state.

        Vector-index deletion is intentionally performed by the
        higher-level lifecycle/orchestration layer.
        """

        memory = await self.repository.get(memory_id)

        if memory is None:
            return

        if memory.namespace != namespace:
            return

        await self.session.delete(memory)
        await self.session.commit()

    async def list_ids(
        self, *, namespace: str, 
        filters: dict | None = None,
    ) -> list[UUID]:
        """
        Return IDs of memories authorized for retrieval.

        PostgreSQL is the canonical source of namespace membership.

        For this milestone, only explicit `allowed_ids` filtering is supported. 
        Arbitrary metadata predicates are intentionally deferred until CAML has a formal provider-neutral filter model.
        """
        memories = await self.repository.by_namespace(namespace)

        if not filters:
            return [
                memory.id for memory in memories
            ]

        allowed_ids = filters.get("allowed_ids")

        if allowed_ids is None:
            return [
                memory.id for memory in memories
            ]

        allowed_uuid_ids: set[UUID] = set()

        for value in allowed_ids:
            try:
                allowed_uuid_ids.add(UUID(str(value)))
            except (ValueError, AttributeError):
                continue

        return [
            memory.id for memory in memories
            if memory.id in allowed_uuid_ids
        ]

    @staticmethod
    def _to_record(
        memory: Memory,
    ) -> MemoryRecord:
        """
        Convert the SQLAlchemy model into the provider-neutral
        MemoryRecord representation.
        """

        metadata = dict(
            memory.metadata_ or {}
        )

        memory_type = metadata.pop(
            "memory_type",
            "semantic",
        )

        return MemoryRecord(
            id=memory.id,
            namespace=memory.namespace,
            content=memory.content,
            memory_type=memory_type,
            metadata=metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )