from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from packages.context.manager import ContextManager
from packages.context.models import ActiveContextEntry, ContextArtifactState

class DTOCPolicy(BaseModel):
    """
    Deterministic policy controlling dynamic context visibility.

    DTOC does not delete artifacts. It only changes whether an 
    artifact is active in the current context.
    """
    model_config = ConfigDict(extra="forbid")

    max_active_tokens: int = Field(ge=1)
    min_tokens_to_hide: int = Field(default=1, ge=1)
    protected_artifact_ids: set[str] = Field(default_factory=set)

class DTOCResult(BaseModel):
    """Result of a DTOC optimization pass."""
    model_config = ConfigDict(extra="forbid")

    context: list[ActiveContextEntry]
    tokens_before: int = Field(ge=0)
    tokens_after: int = Field(ge=0)
    hidden_artifact_ids: list[str] = Field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

class DTOCEngine:
    """
    Deterministic Dynamic Tool Output Compression engine.

    DTOC changes context visibility only. Complete artifact content
    remains unavailable through ContextManager -> ContextArtifactStore.
    """
    def __init__(
        self, manager: ContextManager,
    ) -> None:
        self._manager = manager

    async def optimize(
        self, *, namespace: str,
        policy: DTOCPolicy,
    ) -> DTOCResult:
        context = await self._manager.get_active_context(
            namespace=namespace,
        )

        tokens_before = self._active_tokens(context)

        if tokens_before <= policy.max_active_tokens:
            return DTOCResult(
                context=context, tokens_before=tokens_before,
                tokens_after=tokens_before, hidden_artifact_ids=[],
            )

        eligible = [
            entry for entry in context if (
                entry.state == ContextArtifactState.ACTIVE and
                entry.artifact_id not in policy.protected_artifact_ids and
                entry.token_count >= policy.min_tokens_to_hide
            )
        ]
        # Deterministic ordering:
        # 
        # 1. Older/insertion-order entries are considered first.
        # 2. artifact_id provides deterministic behaviour if the underlying ordering ever changes.
        #
        # ContextManager already returns insertion order, so we preserve that order here.
        hidden_artifact_ids: list[str] = []
        tokens_after = tokens_before

        for entry in eligible:
            if tokens_after <= policy.max_active_tokens:
                break

            await self._manager.hide(
                artifact_id=entry.artifact_id,
                namespace=namespace,
            )

            hidden_artifact_ids.append(entry.artifact_id)
            tokens_after -= entry.token_count

        optimized_context = await self._manager.get_active_context(
            namespace=namespace,
        )

        return DTOCResult(
            context=optimized_context, tokens_before=tokens_before,
            tokens_after=self._active_tokens(optimized_context),
            hidden_artifact_ids=hidden_artifact_ids,
        )

    @staticmethod
    def _active_tokens(
        context: list[ActiveContextEntry],
    ) -> int:
        return sum(
            entry.token_count for entry in context
            if entry.state == ContextArtifactState.ACTIVE
        )

""" 
 Notice that tokens_after is calculated from ACTIVE entries
 A hidden artifact can still appear in: get_active_context()
 because we deliberately preserve its lightweight reference: tool_001 | 1,200 tokens | HIDDEN
 
 But its 1,200 tokens must not count against the active LLM context budget.
 That distinction is fundamental:
 Stored tokens ≠ Active context tokens
"""