from datetime import datetime, timezone

import pytest

from packages.context import (
    ContextArtifact,
    ContextArtifactReference,
    ContextArtifactState,
    can_transition,
)


def make_artifact(
    *,
    state: ContextArtifactState = ContextArtifactState.ACTIVE,
) -> ContextArtifact:
    return ContextArtifact(
        id="tool_007",
        content='{"result": "complete original output"}',
        token_count=1200,
        state=state,
        created_at=datetime.now(timezone.utc),
        source="filesystem",
        namespace="test",
    )


def test_artifact_preserves_complete_content():
    artifact = make_artifact()

    assert artifact.content == (
        '{"result": "complete original output"}'
    )


def test_default_metadata_is_empty():
    artifact = make_artifact()

    assert artifact.metadata == {}


def test_artifact_starts_active():
    artifact = make_artifact()

    assert artifact.state == ContextArtifactState.ACTIVE


def test_hidden_artifact_remains_represented():
    artifact = make_artifact(
        state=ContextArtifactState.HIDDEN,
    )

    assert artifact.state == ContextArtifactState.HIDDEN
    assert artifact.content


def test_hidden_does_not_delete_content():
    artifact = make_artifact()

    original_content = artifact.content

    hidden = artifact.model_copy(
        update={
            "state": ContextArtifactState.HIDDEN,
        }
    )

    assert hidden.content == original_content
    assert hidden.state == ContextArtifactState.HIDDEN


def test_reference_does_not_contain_original_content():
    reference = ContextArtifactReference(
        id="tool_007",
        token_count=1200,
        state=ContextArtifactState.HIDDEN,
        source="filesystem",
        namespace="test",
    )

    assert not hasattr(reference, "content")


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (
            ContextArtifactState.ACTIVE,
            ContextArtifactState.HIDDEN,
        ),
        (
            ContextArtifactState.ACTIVE,
            ContextArtifactState.DELETED,
        ),
        (
            ContextArtifactState.HIDDEN,
            ContextArtifactState.DELETED,
        ),
    ],
)
def test_valid_state_transitions(current, target):
    assert can_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (
            ContextArtifactState.DELETED,
            ContextArtifactState.ACTIVE,
        ),
        (
            ContextArtifactState.DELETED,
            ContextArtifactState.HIDDEN,
        ),
    ],
)
def test_invalid_state_transitions(current, target):
    assert not can_transition(current, target)


def test_deleted_is_terminal():
    for state in ContextArtifactState:
        assert not can_transition(
            ContextArtifactState.DELETED,
            state,
        )


def test_token_count_cannot_be_negative():
    with pytest.raises(ValueError):
        ContextArtifact( 
            id="tool_007", content="content", 
            token_count=-1, state=ContextArtifactState.ACTIVE, 
            created_at=datetime.now(timezone.utc), 
            source="filesystem", namespace="test", 
        )


def test_empty_id_is_rejected():
    with pytest.raises(ValueError):
        ContextArtifact(
            id="",
            content="content",
            token_count=1,
            created_at=datetime.now(timezone.utc),
            source="test",
            namespace="test",
        )