"""Metric computation for the three-way memory-system benchmark.

Every metric is derived from observed execution. Nothing is estimated: when a
value cannot be measured (notably LLM token usage and pricing) it is reported as
unavailable instead of silently approximated.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .config import CostConfig
from .interfaces import SystemName, SystemResponse


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation runs for robust term matching."""
    lowered = text.lower()
    cleaned = [
        character if character.isalnum() else " " for character in lowered
    ]
    return " ".join("".join(cleaned).split())


@dataclass(slots=True)
class CaseOutcome:
    """One evaluation case answered by one system."""

    case_id: str
    category: str
    system: SystemName
    question: str
    answer: str
    expected_answer: str
    correct: bool
    requires_memory: bool
    abstention_correct: bool
    forbidden_hit: str | None
    latency_ms: float
    retrieved_memory_ids: list[str]
    recall_hit: bool
    reciprocal_rank: float
    context_tokens: int
    response: SystemResponse = field(repr=False)

    @property
    def status(self) -> str:
        return "PASS" if self.correct else "FAIL"


def contains_all_terms(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return all(normalize(term) in normalized for term in terms)


def first_forbidden_hit(text: str, terms: list[str]) -> str | None:
    normalized = normalize(text)
    for term in terms:
        if normalize(term) in normalized:
            return term
    return None


def reciprocal_rank_of(
    retrieved: list[str], supporting: list[str],
) -> float:
    """1/rank of the first supporting memory, or 0.0 when absent."""
    if not supporting:
        return 0.0
    supporting_set = set(supporting)
    for position, memory_id in enumerate(retrieved, start=1):
        if memory_id in supporting_set:
            return 1.0 / position
    return 0.0


def looks_like_abstention(answer: str) -> bool:
    """Detect a genuine "I have no information" refusal.

    ``normalize`` turns apostrophes into spaces, so "don't" becomes "don t".
    The markers therefore use the space form.
    """
    normalized = normalize(answer)
    markers = (
        "do not have",
        "don t have",
        "dont have",
        "no information",
        "not have that",
        "not have any",
        "not present",
        "not in the",
        "cannot answer",
        "cannot determine",
        "unable to",
        "insufficient information",
        "not specified",
        "no such",
        "i m sorry",
        "i am sorry",
    )
    return any(marker in normalized for marker in markers)


@dataclass(slots=True)
class SystemReport:
    """Aggregate metrics for one system."""

    system: SystemName
    outcomes: list[CaseOutcome]

    # -- Retrieval ---------------------------------------------------
    @property
    def retrieval_cases(self) -> list[CaseOutcome]:
        return [item for item in self.outcomes if item.retrieved_memory_ids]

    @property
    def uses_memory(self) -> bool:
        return any(item.retrieved_memory_ids for item in self.outcomes)

    def recall_at_k(self) -> float | None:
        candidates = self.retrieval_cases
        if not candidates:
            return None
        hits = sum(1 for item in candidates if item.recall_hit)
        return hits / len(candidates)

    def mrr(self) -> float | None:
        candidates = self.retrieval_cases
        if not candidates:
            return None
        return sum(item.reciprocal_rank for item in candidates) / len(
            candidates
        )

    # -- Answers -----------------------------------------------------
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for item in self.outcomes if item.correct) / len(
            self.outcomes
        )

    def abstention_rate(self) -> float:
        """Share of memory-required cases the system declined to answer.

        High for the stateless baseline, which honestly reports that it has no
        information; near zero for the memory systems, which answer from
        retrieved evidence. A high rate on a memory system would signal a
        retrieval failure, not honesty.
        """
        cases = [item for item in self.outcomes if item.requires_memory]
        if not cases:
            return 0.0
        return sum(1 for item in cases if item.abstention_correct) / len(
            cases
        )

    def memory_grounded_accuracy(self) -> float | None:
        """Share of correct answers that cite retrieved evidence.

        A correct guess by the stateless baseline is NOT evidence of memory
        retrieval: it is correct but ungrounded. This metric separates "the
        answer is right" from "the answer came from memory".
        """
        cases = [item for item in self.outcomes if item.requires_memory]
        if not cases:
            return None
        grounded = sum(
            1
            for item in cases
            if item.correct and item.retrieved_memory_ids
        )
        return grounded / len(cases)

    def unsupported_answer_rate(self) -> float | None:
        """Share of memory-required cases answered without retrieved evidence.

        The stateless baseline answers from parametric knowledge alone, so its
        unsupported rate is high even when the answer happens to be correct.
        """
        cases = [item for item in self.outcomes if item.requires_memory]
        if not cases:
            return None
        unsupported = sum(
            1
            for item in cases
            if not item.abstention_correct and not item.retrieved_memory_ids
        )
        return unsupported / len(cases)

    def category_accuracy(self, category: str) -> float | None:
        subset = [
            item for item in self.outcomes if item.category == category
        ]
        if not subset:
            return None
        return sum(1 for item in subset if item.correct) / len(subset)

    def temporal_accuracy(self) -> float | None:
        return self._mean_categories(
            {"temporal_retrieval", "historical_retrieval"}
        )

    def conflict_accuracy(self) -> float | None:
        return self.category_accuracy("conflict_resolution")

    def _mean_categories(self, categories: set[str]) -> float | None:
        values = [
            value
            for category in categories
            if (value := self.category_accuracy(category)) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    # -- Context -----------------------------------------------------
    @property
    def average_memory_tokens(self) -> float:
        return self._mean(lambda item: item.response.trace.retrieved_memory_tokens)

    @property
    def average_tool_tokens(self) -> float:
        return self._mean(lambda item: item.response.trace.tool_output_tokens)

    @property
    def average_exposed_tool_tokens(self) -> float:
        return self._mean(lambda item: item.response.trace.exposed_tool_tokens)

    @property
    def average_final_context_tokens(self) -> float:
        return self._mean(lambda item: item.response.trace.final_context_tokens)

    def _mean(self, getter) -> float:
        values = [float(getter(item)) for item in self.outcomes]
        return statistics.fmean(values) if values else 0.0

    # -- Latency -----------------------------------------------------
    @property
    def average_latency_ms(self) -> float:
        return self._mean(lambda item: item.latency_ms)

    @property
    def p95_latency_ms(self) -> float:
        values = sorted(item.latency_ms for item in self.outcomes)
        if not values:
            return 0.0
        index = max(0, min(len(values) - 1, int(round(0.95 * len(values))) - 1))
        return values[index]

    def _mean_trace(self, attribute: str) -> float | None:
        values = [
            getattr(item.response.trace, attribute)
            for item in self.outcomes
            if getattr(item.response.trace, attribute, None) is not None
        ]
        return statistics.fmean(values) if values else None

    def phase_latency(self, attribute: str) -> float | None:
        return self._mean_trace(attribute)

    # -- DTOC --------------------------------------------------------
    def artifact_externalization_rate(self) -> float | None:
        applicable = [
            item for item in self.outcomes
            if item.response.trace.tool_output_tokens > 0
        ]
        if not applicable:
            return None
        hidden = [
            item for item in applicable
            if item.response.trace.artifact_state == "hidden"
        ]
        return len(hidden) / len(applicable)

    def restoration_success_rate(self) -> float | None:
        applicable = [
            item for item in self.outcomes
            if item.response.trace.artifact_restorable
        ]
        if not applicable:
            return None
        return sum(
            1 for item in applicable if item.response.trace.artifact_restorable
        ) / len(applicable)

    def tokens_saved(self) -> int | None:
        values = [
            item.response.trace.tokens_saved
            for item in self.outcomes
            if item.response.trace.tokens_saved is not None
        ]
        return sum(values) if values else None

    # -- Tokens / cost -----------------------------------------------
    @property
    def token_usage_unavailable(self) -> bool:
        """True when the LLM provider reports no token usage."""
        return all(item.response.trace.total_tokens is None
                   for item in self.outcomes)

    def total_cost(self, cost: CostConfig) -> float | None:
        if not cost.is_configured:
            return None
        total = 0.0
        for item in self.outcomes:
            trace = item.response.trace
            if trace.input_tokens is None or trace.output_tokens is None:
                return None
            total += (
                trace.input_tokens
                / 1_000_000
                * float(cost.input_cost_per_million_tokens)
            )
            total += (
                trace.output_tokens
                / 1_000_000
                * float(cost.output_cost_per_million_tokens)
            )
        return total

    def cost_per_successful_answer(self, cost: CostConfig) -> float | None:
        total = self.total_cost(cost)
        if total is None:
            return None
        successes = sum(1 for item in self.outcomes if item.correct)
        if successes == 0:
            return None
        return total / successes


def format_optional(value: float | None, *, percent: bool = False) -> str:
    """Render a metric that may legitimately not exist."""
    if value is None:
        return "--"
    if percent:
        return f"{value * 100:.0f}%"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.2f}"