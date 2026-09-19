from __future__ import annotations

from datetime import datetime, timezone

import pytest

from packages.context.cache_store import CacheContextArtifactStore
from packages.context.exceptions import (
    ContextArtifactDeleted,
    ContextArtifactNotFound,
    InvalidContextStateTransition,
)
from packages.context.models import (
    ContextArtifact,
    ContextArtifactState,
)
from packages.providers.cache import CacheProvider


class FakeCacheProvider(CacheProvider):
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    async def get(self, key: str):
        return self.data.get(key)

    async def set(
        self,
        key: str,
        value,
        *,
        ttl: int | None = None,
    ) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self.data


def make_artifact(
    *,
    artifact_id: str = "tool_007",
    namespace: str = "test",
) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        content='{"complete": "original output"}',
        token_count=1200,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="filesystem",
        namespace=namespace,
    )


@pytest.fixture
def store():
    return CacheContextArtifactStore(
        FakeCacheProvider()
    )


async def test_store_and_get(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    result = await store.get(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert result is not None
    assert result.id == artifact.id
    assert result.content == artifact.content
    assert result.token_count == 1200
    assert result.state == ContextArtifactState.ACTIVE


async def test_hide_preserves_original_content(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    hidden = await store.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert hidden.state == ContextArtifactState.HIDDEN
    assert hidden.content == artifact.content
    assert hidden.token_count == artifact.token_count


async def test_restore_returns_artifact_to_active(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    await store.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    restored = await store.restore(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert restored.state == ContextArtifactState.ACTIVE
    assert restored.content == artifact.content


async def test_hide_then_get_remains_recoverable(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    await store.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    result = await store.get(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert result is not None
    assert result.state == ContextArtifactState.HIDDEN
    assert result.content == artifact.content


async def test_delete_removes_artifact(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    await store.delete(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert not await store.exists(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert await store.get(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    ) is None


async def test_get_missing_artifact_returns_none(store):
    result = await store.get(
        artifact_id="missing",
        namespace="test",
    )

    assert result is None


async def test_hide_missing_artifact_raises(store):
    with pytest.raises(ContextArtifactNotFound):
        await store.hide(
            artifact_id="missing",
            namespace="test",
        )


async def test_restore_missing_artifact_raises(store):
    with pytest.raises(ContextArtifactNotFound):
        await store.restore(
            artifact_id="missing",
            namespace="test",
        )


async def test_namespace_isolation(store):
    artifact = make_artifact(
        namespace="tenant-a",
    )

    await store.store(artifact=artifact)

    result = await store.get(
        artifact_id=artifact.id,
        namespace="tenant-b",
    )

    assert result is None


async def test_namespace_isolation_for_exists(store):
    artifact = make_artifact(
        namespace="tenant-a",
    )

    await store.store(artifact=artifact)

    assert not await store.exists(
        artifact_id=artifact.id,
        namespace="tenant-b",
    )


async def test_cannot_restore_active_artifact(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    with pytest.raises(InvalidContextStateTransition):
        # Restore is only valid after HIDDEN.
        await store.restore(
            artifact_id=artifact.id,
            namespace=artifact.namespace,
        )


async def test_cannot_hide_deleted_artifact(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    await store.delete(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    with pytest.raises(ContextArtifactNotFound):
        await store.hide(
            artifact_id=artifact.id,
            namespace=artifact.namespace,
        )


async def test_delete_hidden_artifact(store):
    artifact = make_artifact()

    await store.store(artifact=artifact)

    await store.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    await store.delete(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert not await store.exists(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )