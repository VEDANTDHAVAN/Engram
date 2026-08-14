from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4, UUID

import pytest

from packages.providers.memory import (
    PostgresMemoryProvider,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FakeMemory:
    id: UUID
    namespace: str
    content: str
    metadata_: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def __init__(
        self,
        *,
        memory_id: UUID | None = None,
        namespace: str = "tenant-a",
        content: str = "test memory",
        metadata_: dict[str, Any] | None = None,
    ) -> None:
        self.id = memory_id or uuid4()
        self.namespace = namespace
        self.content = content
        self.metadata_ = dict(metadata_ or {})

        self.embedding = None
        self.state = None
        self.confidence = 1.0
        self.importance = 0.5
        self.version = 1
        self.parent_memory_id = None
        self.valid_from = datetime.now(timezone.utc)
        self.valid_until = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

class FakeRepository:
    memory: FakeMemory | None

    def __init__(self) -> None:
        self.memory = None

    async def create(self, memory: FakeMemory) -> None:
        self.memory = memory

        # Simulate SQLAlchemy/DB-generated primary key.
        if getattr(memory, "id", None) is None:
            setattr(memory, "id", uuid4())

    async def get(self, memory_id: UUID) -> object | None:
        memory = self.memory

        if memory is None:
            return None

        if getattr(memory, "id", None) != memory_id:
            return None

        return memory

class FakeSession:
    async def commit(self) -> None:
        pass

    async def refresh(self, obj: object) -> None:
        """
        Simulate PostgreSQL server defaults being populated
        after INSERT + refresh.
        """

        if getattr(obj, "created_at", None) is None:
            setattr(
                obj,
                "created_at",
                utc_now(),
            )

        if getattr(obj, "updated_at", None) is None:
            setattr(
                obj,
                "updated_at",
                utc_now(),
            )

    async def delete(self, obj: object) -> None:
        pass

@pytest.mark.asyncio
async def test_create_returns_memory_record():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    repository = FakeRepository()
    provider.repository = cast(Any, repository)

    result = await provider.create(
        namespace="tenant-a",
        content="PostgreSQL is the source of truth.",
        memory_type="semantic",
        metadata={
            "source": "test",
        },
    )

    assert result.namespace == "tenant-a"

    assert (
        result.content
        == "PostgreSQL is the source of truth."
    )

    assert result.memory_type == "semantic"

    assert result.metadata["source"] == "test"

    assert result.created_at is not None


@pytest.mark.asyncio
async def test_memory_type_is_stored_in_metadata():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    repository = FakeRepository()
    provider.repository = cast(Any, repository)

    result = await provider.create(
        namespace="tenant-a",
        content="Semantic memory",
        memory_type="semantic",
    )

    assert repository.memory is not None

    assert (
        repository.memory.metadata_["memory_type"]
        == "semantic"
    )

    assert result.memory_type == "semantic"


@pytest.mark.asyncio
async def test_get_enforces_namespace():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    repository = FakeRepository()
    provider.repository = cast(Any, repository)

    memory = FakeMemory(
        namespace="tenant-a",
        metadata_={
            "memory_type": "semantic",
        },
    )

    repository.memory = memory

    result = await provider.get(
        memory.id,
        namespace="tenant-a",
    )

    assert result is not None
    assert result.id == memory.id

    unauthorized = await provider.get(
        memory.id,
        namespace="tenant-b",
    )

    assert unauthorized is None


@pytest.mark.asyncio
async def test_get_missing_memory_returns_none():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    provider.repository = cast(Any, FakeRepository())

    result = await provider.get(
        uuid4(),
        namespace="tenant-a",
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_memory():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    repository = FakeRepository()
    provider.repository = cast(Any, repository)

    memory = FakeMemory(
        namespace="tenant-a",
        content="old content",
        metadata_={
            "memory_type": "semantic",
            "source": "original",
        },
    )

    repository.memory = memory

    result = await provider.update(
        memory.id,
        namespace="tenant-a",
        content="new content",
        metadata={
            "updated": True,
        },
    )

    assert result.content == "new content"

    assert result.metadata["updated"] is True

    assert result.memory_type == "semantic"

    assert result.metadata["source"] == "original"


@pytest.mark.asyncio
async def test_delete_respects_namespace():
    provider = PostgresMemoryProvider(
        cast(Any, FakeSession())
    )

    repository = FakeRepository()
    provider.repository = cast(Any, repository)

    memory = FakeMemory(
        namespace="tenant-a",
    )

    repository.memory = memory

    await provider.delete(
        memory.id,
        namespace="tenant-b",
    )

    assert repository.memory is memory