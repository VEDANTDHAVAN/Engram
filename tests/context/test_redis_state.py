from __future__ import annotations

from typing import Any

import pytest

from packages.context.redis_state import RedisContextStateStore
from packages.providers.cache import CacheProvider


class FakeCacheProvider(CacheProvider):
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self.data.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
    ) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self.data


@pytest.mark.asyncio
async def test_redis_state_add_and_get() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_002",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == [
        "tool_001",
        "tool_002",
    ]


@pytest.mark.asyncio
async def test_redis_state_add_is_idempotent() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == ["tool_001"]


@pytest.mark.asyncio
async def test_redis_state_remove() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_002",
    )

    await store.remove(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == ["tool_002"]


@pytest.mark.asyncio
async def test_redis_state_removes_key_when_empty() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.remove(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == []

    assert not await cache.exists(
        "engram:context:tenant_a:active"
    )


@pytest.mark.asyncio
async def test_redis_state_namespace_isolation() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.add(
        namespace="tenant_b",
        artifact_id="tool_002",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == ["tool_001"]

    assert await store.get_ids(
        namespace="tenant_b",
    ) == ["tool_002"]


@pytest.mark.asyncio
async def test_redis_state_exists() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    assert await store.exists(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    assert not await store.exists(
        namespace="tenant_a",
        artifact_id="tool_002",
    )


@pytest.mark.asyncio
async def test_redis_state_clear() -> None:
    cache = FakeCacheProvider()
    store = RedisContextStateStore(cache)

    await store.add(
        namespace="tenant_a",
        artifact_id="tool_001",
    )

    await store.clear(
        namespace="tenant_a",
    )

    assert await store.get_ids(
        namespace="tenant_a",
    ) == []