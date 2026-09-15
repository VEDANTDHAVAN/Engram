from datetime import datetime, timezone

import pytest

from packages.context.cache_store import CacheContextArtifactStore
from packages.context.dtoc import DTOCEngine, DTOCPolicy
from packages.context.manager import ContextManager
from packages.context.models import (
    ContextArtifact,
    ContextArtifactState,
)
from tests.context.test_store import FakeCacheProvider


def make_artifact(
    artifact_id: str,
    token_count: int,
    namespace: str = "user_001",
) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        content=f"original content for {artifact_id}",
        token_count=token_count,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="tool",
        namespace=namespace,
    )

async def make_manager() -> ContextManager:
    cache = FakeCacheProvider()
    store = CacheContextArtifactStore(cache)
    return ContextManager(store)


@pytest.mark.asyncio
async def test_dtoc_does_nothing_when_context_is_within_budget() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    await manager.add_artifact(
       artifact=make_artifact(
            artifact_id="tool_001",
            token_count=100,
        )
    )

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="tool_002",
            token_count=150,
        )
    )

    result = await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=300,
        ),
    )

    assert result.tokens_before == 250
    assert result.tokens_after == 250
    assert result.tokens_saved == 0
    assert result.hidden_artifact_ids == []

    assert all(
        entry.state == ContextArtifactState.ACTIVE
        for entry in result.context
    )


@pytest.mark.asyncio
async def test_dtoc_hides_artifacts_until_budget_is_met() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="tool_001",
            token_count=100,
        )
    )

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="tool_002",
            token_count=150,
        )
    )

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="tool_003",
            token_count=200,
        )
    )

    result = await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=250,
        ),
    )

    assert result.tokens_before == 450
    assert result.tokens_after == 200

    assert result.hidden_artifact_ids == [
        "tool_001",
        "tool_002",
    ]

    active = [
        entry.artifact_id
        for entry in result.context
        if entry.state == ContextArtifactState.ACTIVE
    ]

    hidden = [
        entry.artifact_id
        for entry in result.context
        if entry.state == ContextArtifactState.HIDDEN
    ]

    assert active == ["tool_003"]
    assert hidden == ["tool_001", "tool_002"]


@pytest.mark.asyncio
async def test_dtoc_preserves_hidden_artifact_as_reference() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    artifact = make_artifact(
        artifact_id="tool_001",
        token_count=500,
    )

    await manager.add_artifact(artifact=artifact)

    result = await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=100,
        ),
    )

    assert result.hidden_artifact_ids == ["tool_001"]

    hidden_entry = result.context[0]

    assert hidden_entry.artifact_id == "tool_001"
    assert hidden_entry.state == ContextArtifactState.HIDDEN
    assert hidden_entry.token_count == 500


@pytest.mark.asyncio
async def test_dtoc_is_lossless() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    original_content = (
        "This is the complete original tool output. "
        "It must remain unchanged after DTOC."
    )

    artifact = ContextArtifact(
        id="tool_001",
        content=original_content,
        token_count=500,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="filesystem",
        namespace="user_001",
    )

    await manager.add_artifact(artifact=artifact)

    await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=100,
        ),
    )

    stored = await manager.get_artifact(
        artifact_id="tool_001",
        namespace="user_001",
    )

    assert stored.content == original_content
    assert stored.token_count == 500
    assert stored.state == ContextArtifactState.HIDDEN


@pytest.mark.asyncio
async def test_dtoc_protects_selected_artifacts() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="important",
            token_count=400,
        )
    )

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="tool_002",
            token_count=300,
        )
    )

    result = await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=400,
            protected_artifact_ids={"important"},
        ),
    )

    assert result.hidden_artifact_ids == ["tool_002"]

    context = {
        entry.artifact_id: entry
        for entry in result.context
    }

    assert context["important"].state == ContextArtifactState.ACTIVE
    assert context["tool_002"].state == ContextArtifactState.HIDDEN


@pytest.mark.asyncio
async def test_dtoc_respects_minimum_token_threshold() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="small",
            token_count=50,
        )
    )

    await manager.add_artifact(
        artifact=make_artifact(
            artifact_id="large",
            token_count=500,
        )
    )

    result = await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=100,
            min_tokens_to_hide=100,
        ),
    )

    assert result.hidden_artifact_ids == ["large"]

    context = {
        entry.artifact_id: entry
        for entry in result.context
    }

    assert context["small"].state == ContextArtifactState.ACTIVE
    assert context["large"].state == ContextArtifactState.HIDDEN


@pytest.mark.asyncio
async def test_dtoc_does_not_delete_artifacts() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    artifact = make_artifact(
        artifact_id="tool_001",
        token_count=500,
    )

    await manager.add_artifact(artifact=artifact)

    await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=100,
        ),
    )

    stored = await manager.get_artifact(
        artifact_id="tool_001",
        namespace="user_001",
    )

    assert stored.content == artifact.content
    assert stored.state == ContextArtifactState.HIDDEN


@pytest.mark.asyncio
async def test_dtoc_can_restore_hidden_artifact() -> None:
    manager = await make_manager()
    engine = DTOCEngine(manager)

    original_content = "complete tool output"

    artifact = ContextArtifact(
        id="tool_001",
        content=original_content,
        token_count=500,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="tool",
        namespace="user_001",
    )

    await manager.add_artifact(artifact=artifact)

    await engine.optimize(
        namespace="user_001",
        policy=DTOCPolicy(
            max_active_tokens=100,
        ),
    )

    restored = await manager.restore(
        artifact_id="tool_001",
        namespace="user_001",
    )

    assert restored.state == ContextArtifactState.ACTIVE

    complete = await manager.get_artifact(
        artifact_id="tool_001",
        namespace="user_001",
    )

    assert complete.content == original_content