import pytest

from packages.context.in_memory_state import (
    InMemoryContextStateStore,
)


@pytest.mark.asyncio
async def test_add_and_get_ids() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="user_001",
        artifact_id="tool_002",
    )

    assert await store.get_ids(
        namespace="user_001",
    ) == [
        "tool_001",
        "tool_002",
    ]


@pytest.mark.asyncio
async def test_add_is_idempotent() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    assert await store.get_ids(
        namespace="user_001",
    ) == ["tool_001"]


@pytest.mark.asyncio
async def test_remove() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="user_001",
        artifact_id="tool_002",
    )

    await store.remove(
        namespace="user_001",
        artifact_id="tool_001",
    )

    assert await store.get_ids(
        namespace="user_001",
    ) == ["tool_002"]


@pytest.mark.asyncio
async def test_exists() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    assert await store.exists(
        namespace="user_001",
        artifact_id="tool_001",
    )

    assert not await store.exists(
        namespace="user_001",
        artifact_id="tool_002",
    )


@pytest.mark.asyncio
async def test_namespace_isolation() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="namespace_a",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="namespace_b",
        artifact_id="tool_002",
    )

    assert await store.get_ids(
        namespace="namespace_a",
    ) == ["tool_001"]

    assert await store.get_ids(
        namespace="namespace_b",
    ) == ["tool_002"]


@pytest.mark.asyncio
async def test_clear() -> None:
    store = InMemoryContextStateStore()

    await store.add(
        namespace="user_001",
        artifact_id="tool_001",
    )

    await store.clear(
        namespace="user_001",
    )

    assert await store.get_ids(
        namespace="user_001",
    ) == []