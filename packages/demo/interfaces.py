"""Common interface implemented by every memory system under comparison.

The evaluator depends only on ``MemorySystem``. It never branches on which
system it is driving, so adding a fourth architecture cannot require an
evaluator change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter


def count_tokens(text: str) -> int:
    """Deterministic token estimate shared by every system.

    A single counting rule is used for prompt, memory and artifact accounting so
    that the context columns in the benchmark are directly comparable.
    """
    return len(text.split())


class SystemName(str, Enum):
    """Canonical system identifiers."""

    NO_MEMORY = "no_memory"
    CONVENTIONAL_MEMORY = "conventional_memory"
    ENGRAM = "engram"


@dataclass(slots=True)
class MemoryTrace:
    """Everything the benchmark measures about one system's execution.

    A system fills in only the fields its architecture actually has. A blank
    field therefore reports "not applicable to this architecture" rather than a
    value silently defaulted to zero.
    """

    # --- Retrieval ------------------------------------------------------
    strategies: list[str] = field(default_factory=list)
    fusion: str | None = None
    decision_provider: str | None = None
    decision_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    fusion_latency_ms: float | None = None

    # --- Evidence --------------------------------------------------------
    retrieved_memory_ids: list[str] = field(default_factory=list)
    retrieved_memory_count: int = 0
    retrieved_memory_tokens: int = 0
    superseded_memory_count: int = 0

    # --- Tool output -----------------------------------------------------
    tool_output_tokens: int = 0
    exposed_tool_tokens: int = 0
    artifact_id: str | None = None
    artifact_state: str | None = None
    artifact_restorable: bool | None = None
    artifact_store_latency_ms: float | None = None

    # --- Context ---------------------------------------------------------
    context_optimization_latency_ms: float | None = None
    final_context_tokens: int = 0
    tokens_saved: int | None = None

    # --- LLM -------------------------------------------------------------
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # --- Promotion -------------------------------------------------------
    evidence_promoted: bool | None = None
    recall_rank: int | None = None


@dataclass(slots=True)
class SystemResponse:
    """One answered query from one memory system."""

    system: SystemName
    answer: str
    prompt: str
    latency_ms: float
    trace: MemoryTrace
    corrected: bool = False


class MemorySystem(ABC):
    """A memory architecture the evaluator can drive identically."""

    name: SystemName

    @abstractmethod
    async def answer(
        self,
        query: str,
        *,
        namespace: str,
        tool_output: str | None = None,
        artifact_id: str | None = None,
        restore_artifact_id: str | None = None,
    ) -> SystemResponse:
        """Answer ``query`` using this system's memory architecture."""

    @property
    @abstractmethod
    def uses_memory(self) -> bool:
        """Whether this architecture retrieves prior memory at all."""

    @property
    @abstractmethod
    def uses_externalization(self) -> bool:
        """Whether this architecture externalizes content into artifacts."""

    async def __call__(self, query: str, **kwargs) -> SystemResponse:
        return await self.answer(query, **kwargs)


class TimedCall:
    """Small helper that records elapsed milliseconds on a trace field."""

    __slots__ = ("trace", "attribute", "started")

    def __init__(self, trace: MemoryTrace, attribute: str) -> None:
        self.trace = trace
        self.attribute = attribute

    def __enter__(self) -> TimedCall:
        self.started = perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        setattr(
            self.trace,
            self.attribute,
            (perf_counter() - self.started) * 1000,
        )