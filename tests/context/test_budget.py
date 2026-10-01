from datetime import datetime, timezone

import pytest

from packages.context.budget import (
    ContextBudget, ContextUsage,
    TokenBudgetManager,
)
from packages.context.models import ContextArtifact, ContextArtifactState


def make_artifact(artifact_id: str, token_count: int) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        content=f"original content for {artifact_id}",
        token_count=token_count,
        state=ContextArtifactState.ACTIVE,
        created_at=datetime.now(timezone.utc),
        source="tool",
        namespace="user_001",
    )


def test_available_context_tokens() -> None:
    budget = ContextBudget(
        context_window_tokens=128_000,
        reserved_output_tokens=16_000,
        safety_margin_tokens=2_000,
    )

    assert budget.available_context_tokens == 110_000


def test_budget_rejects_negative_context_window() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            context_window_tokens=-1,
        )


def test_budget_rejects_negative_output_reservation() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            context_window_tokens=100,
            reserved_output_tokens=-1,
        )


def test_budget_rejects_negative_safety_margin() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            context_window_tokens=100,
            safety_margin_tokens=-1,
        )


def test_budget_rejects_reservations_larger_than_window() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            context_window_tokens=100,
            reserved_output_tokens=80,
            safety_margin_tokens=30,
        )


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError):
        ContextUsage(
            active_tokens=-1,
        )


def test_usage_within_budget() -> None:
    budget = ContextBudget(
        context_window_tokens=1_000,
        reserved_output_tokens=200,
    )

    usage = ContextUsage(
        active_tokens=700,
    )

    assert not usage.exceeds(budget)
    assert usage.remaining_tokens(budget) == 100
    assert usage.overflow_tokens(budget) == 0


def test_usage_exceeds_budget() -> None:
    budget = ContextBudget(
        context_window_tokens=1_000,
        reserved_output_tokens=200,
    )

    usage = ContextUsage(
        active_tokens=900,
    )

    assert usage.exceeds(budget)
    assert usage.remaining_tokens(budget) == 0
    assert usage.overflow_tokens(budget) == 100


def test_manager_fits() -> None:
    manager = TokenBudgetManager(
        ContextBudget(
            context_window_tokens=1_000,
            reserved_output_tokens=200,
        )
    )

    assert manager.fits(
        active_tokens=800,
    )

    assert not manager.fits(
        active_tokens=801,
    )


def test_manager_remaining() -> None:
    manager = TokenBudgetManager(
        ContextBudget(
            context_window_tokens=1_000,
            reserved_output_tokens=200,
        )
    )

    assert manager.remaining(
        active_tokens=600,
    ) == 200


def test_manager_overflow() -> None:
    manager = TokenBudgetManager(
        ContextBudget(
            context_window_tokens=1_000,
            reserved_output_tokens=200,
        )
    )

    assert manager.overflow(
        active_tokens=900,
    ) == 100

def test_active_token_calculation() -> None:
    artifacts = [
        make_artifact(
            artifact_id="tool_001",
            token_count=400,
        ),
        make_artifact(
            artifact_id="tool_002",
            token_count=600,
        ),
        make_artifact(
            artifact_id="tool_003",
            token_count=1_000,
        ),
    ]

    manager = TokenBudgetManager(
        ContextBudget(context_window_tokens=2_000),
    )

    assert manager.calculate_active_tokens(artifacts) == 2_000