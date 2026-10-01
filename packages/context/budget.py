from __future__ import annotations

from dataclasses import dataclass
from packages.context.models import ContextArtifact

@dataclass(frozen=True)
class ContextBudget:
    """
    Token budget available for active context.

    context_window_tokens:
        Maximum model context size.

    reserved_output_tokens:
        Tokens reserved for the model's response.

    safety_margin_tokens:
        Additional deterministic buffer preventing us from
        consuming the entire available context window.
    """
    context_window_tokens: int
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0

    def __post_init__(self) -> None:
        if self.context_window_tokens < 0:
            raise ValueError(
                "context_window_tokens cannot be negative"
            )

        if self.reserved_output_tokens < 0:
            raise ValueError(
                "reserved_output_tokens cannot be negative"
            )

        if self.safety_margin_tokens < 0:
            raise ValueError(
                "safety_margin_tokens cannot be negative"
            )

        if self.reserved_output_tokens + self.safety_margin_tokens > self.context_window_tokens:
            raise ValueError(
                "Reserved output and safety margin exceed context window"
            )

    @property
    def available_context_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )

@dataclass(frozen=True)
class ContextUsage:
    """
    Deterministic snapshot of context token usage.
    """
    active_tokens: int

    def exceeds(self, budget: ContextBudget) -> bool:
        return self.active_tokens > budget.available_context_tokens

    def remaining_tokens(
        self,
        budget: ContextBudget,
    ) -> int:
        return max(
            budget.available_context_tokens
            - self.active_tokens,
            0,
        )

    def overflow_tokens(
        self,
        budget: ContextBudget,
    ) -> int:
        return max(
            self.active_tokens
            - budget.available_context_tokens,
            0,
        )

    def __post_init__(self) -> None:
        if self.active_tokens < 0:
            raise ValueError(
                "active_tokens cannot be negative"
            )

class TokenBudgetManager:
    """
    Deterministic context-budget accounting.

    This component does not decide which artifacts to hide.
    It only determines whether the current context fits.
    """
    def __init__(
        self,
        budget: ContextBudget,
    ) -> None:
        self.budget = budget

    def usage(
        self, *,
        active_tokens: int,
    ) -> ContextUsage:
        return ContextUsage(
            active_tokens=active_tokens,
        )

    def fits(
        self, *,
        active_tokens: int,
    ) -> bool:
        return active_tokens <= self.budget.available_context_tokens

    def remaining(
        self, *,
        active_tokens: int,
    ) -> int:
        return max(
            self.budget.available_context_tokens - active_tokens, 0,
        )

    def overflow(
        self, *,
        active_tokens: int,
    ) -> int:
        return max(
            active_tokens - self.budget.available_context_tokens, 0,
        )

    def calculate_active_tokens(
        self,
        artifacts: list[ContextArtifact],
    ) -> int:
        return sum(
            artifact.token_count for artifact in artifacts
        )