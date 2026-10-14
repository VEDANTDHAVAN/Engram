"""Shared experimental conditions for the three-way memory comparison.

Every memory system under comparison is constructed from one ``DemoConfig``.
There are no per-system overrides: the only intentional difference between
``NoMemorySystem``, ``ConventionalMemorySystem`` and ``EngramSystem`` is the
memory architecture itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "demo"


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Explicit, configurable token pricing.

    Prices are never invented. When a price is left as ``None`` the metric is
    reported as unavailable rather than silently estimated.
    """

    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None

    @property
    def is_configured(self) -> bool:
        return (
            self.input_cost_per_million_tokens is not None
            and self.output_cost_per_million_tokens is not None
        )

    def describe(self) -> str:
        if not self.is_configured:
            return "token_usage_unavailable"
        return (
            f"input=${self.input_cost_per_million_tokens}/1M, "
            f"output=${self.output_cost_per_million_tokens}/1M"
        )


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Conditions held constant across every system under comparison."""

    # --- Model conditions -------------------------------------------------
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt: str = (
        "You are the TaskFlow Engineering Agent, a precise AI engineering "
        "assistant. Answer using only the supplied conversation, tool output, "
        "and memory evidence. If the requested information is not present in "
        "that material, say that you do not have it. Do not guess."
    )

    # --- Data conditions --------------------------------------------------
    data_directory: Path = DEFAULT_DATA_DIRECTORY
    memories_filename: str = "memories.json"
    evaluation_filename: str = "evaluation_cases.json"

    # --- Retrieval conditions ---------------------------------------------
    namespace: str = "taskflow-benchmark"
    top_k: int = 5

    # --- Context conditions -----------------------------------------------
    context_window_tokens: int = 32768
    reserved_output_tokens: int = 1024
    safety_margin_tokens: int = 512
    max_active_artifact_tokens: int = 128

    # --- Tool conditions --------------------------------------------------
    tool_name: str = "search_codebase"
    artifact_id: str = "tool_search_codebase_001"

    cost: CostConfig = field(default_factory=CostConfig)

    @property
    def memories_path(self) -> Path:
        return self.data_directory / self.memories_filename

    @property
    def evaluation_path(self) -> Path:
        return self.data_directory / self.evaluation_filename

    @property
    def available_context_tokens(self) -> int:
        """Active tokens available before the budget is considered exceeded."""
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )