from datetime import datetime, timezone

import pytest

from packages.context.cache_store import CacheContextArtifactStore
from packages.context.manager import ContextManager
from packages.context.models import (
    ContextArtifact,
    ContextArtifactState,
)
from packages.context.exceptions import ContextArtifactNotFound
from tests.context.test_store import FakeCacheProvider
from packages.context.in_memory_state import InMemoryContextStateStore

def make_artifact(
    *,
    artifact_id: str = "tool_001",
    namespace: str = "user_001",
    content: str = "original tool output",
) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        content=content,
        token_count=100,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="tool",
        namespace=namespace,
        metadata={"tool": "filesystem"},
    )


@pytest.mark.asyncio
async def test_add_artifact_persists_and_adds_to_context() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager( 
        store, state_store,
    )

    artifact = make_artifact()

    entry = await manager.add_artifact(artifact=artifact)

    assert entry.artifact_id == artifact.id
    assert entry.namespace == artifact.namespace
    assert entry.token_count == artifact.token_count
    assert entry.state == ContextArtifactState.ACTIVE
    assert entry.is_visible is True
    assert entry.is_hidden is False

    context = await manager.get_active_context(
        namespace=artifact.namespace,
    )

    assert len(context) == 1
    assert context[0].artifact_id == artifact.id


@pytest.mark.asyncio
async def test_active_context_does_not_contain_full_content() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    artifact = make_artifact(
        content="VERY LARGE TOOL OUTPUT",
    )

    await manager.add_artifact(artifact=artifact)

    context = await manager.get_active_context(
        namespace=artifact.namespace,
    )

    entry = context[0]

    assert not hasattr(entry, "content")
    assert entry.token_count == artifact.token_count


@pytest.mark.asyncio
async def test_hide_preserves_original_artifact() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    original_content = "complete original tool output"

    artifact = make_artifact(
        content=original_content,
    )

    await manager.add_artifact(artifact=artifact)

    hidden = await manager.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert hidden.state == ContextArtifactState.HIDDEN
    assert hidden.is_hidden is True

    stored = await manager.get_artifact(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert stored.content == original_content
    assert stored.state == ContextArtifactState.HIDDEN


@pytest.mark.asyncio
async def test_hidden_artifact_remains_represented_in_context() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    artifact = make_artifact()

    await manager.add_artifact(artifact=artifact)

    await manager.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    context = await manager.get_active_context(
        namespace=artifact.namespace,
    )

    assert len(context) == 1
    assert context[0].artifact_id == artifact.id
    assert context[0].state == ContextArtifactState.HIDDEN
    assert context[0].is_hidden is True


@pytest.mark.asyncio
async def test_restore_returns_original_artifact_reference() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    original_content = "exact original output"

    artifact = make_artifact(
        content=original_content,
    )

    await manager.add_artifact(artifact=artifact)

    await manager.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    restored_entry = await manager.restore(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert restored_entry.state == ContextArtifactState.ACTIVE

    restored = await manager.get_artifact(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    assert restored.content == original_content


@pytest.mark.asyncio
async def test_restore_does_not_duplicate_context_entry() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    artifact = make_artifact()

    await manager.add_artifact(artifact=artifact)

    await manager.hide(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    await manager.restore(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    context = await manager.get_active_context(
        namespace=artifact.namespace,
    )

    assert len(context) == 1
    assert context[0].artifact_id == artifact.id
    assert context[0].state == ContextArtifactState.ACTIVE

@pytest.mark.asyncio
async def test_namespace_isolation() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    artifact_a = make_artifact(
        artifact_id="tool_a",
        namespace="namespace_a",
    )

    artifact_b = make_artifact(
        artifact_id="tool_b",
        namespace="namespace_b",
    )

    await manager.add_artifact(artifact=artifact_a)
    await manager.add_artifact(artifact=artifact_b)

    context_a = await manager.get_active_context(
        namespace="namespace_a",
    )

    context_b = await manager.get_active_context(
        namespace="namespace_b",
    )

    assert [entry.artifact_id for entry in context_a] == ["tool_a"]
    assert [entry.artifact_id for entry in context_b] == ["tool_b"]


@pytest.mark.asyncio
async def test_remove_deletes_artifact_and_context_entry() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )

    artifact = make_artifact()

    await manager.add_artifact(artifact=artifact)

    await manager.remove(
        artifact_id=artifact.id,
        namespace=artifact.namespace,
    )

    context = await manager.get_active_context(
        namespace=artifact.namespace,
    )

    assert context == []

    with pytest.raises(ContextArtifactNotFound):
        await manager.get_artifact(
            artifact_id=artifact.id,
            namespace=artifact.namespace,
        )

@pytest.mark.asyncio
async def test_context_state_is_owned_by_state_store() -> None:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    state_store = InMemoryContextStateStore()

    manager = ContextManager(
        store, state_store,
    )
    artifact = make_artifact()

    await manager.add_artifact(artifact=artifact)

    assert await state_store.exists(
        namespace=artifact.namespace, 
        artifact_id=artifact.id,
    )