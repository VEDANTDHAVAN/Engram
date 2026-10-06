from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .base import DecisionProvider
from .models import DecisionRequest
from packages.retrieval.policy import RetrievalStrategyName


@dataclass(frozen=True)
class RoutingEvaluationCase:
    query: str
    expected: frozenset[RetrievalStrategyName]
    namespace: str = "evaluation"
    top_k: int = 10


@dataclass(frozen=True)
class RoutingEvaluationResult:
    query: str
    expected: frozenset[RetrievalStrategyName]
    predicted: frozenset[RetrievalStrategyName]
    correct: bool
    confidence: float
    latency_ms: float
    provider: str


@dataclass(frozen=True)
class RoutingEvaluationSummary:
    provider: str
    total_cases: int
    correct_cases: int
    accuracy: float
    average_latency_ms: float
    p95_latency_ms: float
    average_confidence: float
    false_high_confidence_count: int

class DecisionRoutingEvaluator:
    def __init__(
        self,
        provider: DecisionProvider,
    ) -> None:
        self._provider = provider

    async def evaluate(
        self,
        cases: list[RoutingEvaluationCase],
    ) -> tuple[
        list[RoutingEvaluationResult],
        RoutingEvaluationSummary,
    ]:
        results: list[RoutingEvaluationResult] = []

        for case in cases:
            request = DecisionRequest(
                query=case.query,
                namespace=case.namespace,
                top_k=case.top_k,
            )

            started = perf_counter()

            response = await self._provider.decide(request)

            measured_latency_ms = (
                perf_counter() - started
            ) * 1000

            predicted = frozenset(
                response.decision.strategies
            )

            results.append(
                RoutingEvaluationResult(
                    query=case.query,
                    expected=case.expected,
                    predicted=predicted,
                    correct=predicted == case.expected,
                    confidence=response.confidence,
                    latency_ms=measured_latency_ms,
                    provider=response.provider,
                )
            )

        return results, self._summarize(results)

    def _summarize(
        self,
        results: list[RoutingEvaluationResult],
    ) -> RoutingEvaluationSummary:
        if not results:
            return RoutingEvaluationSummary(
                provider=self._provider.name,
                total_cases=0,
                correct_cases=0,
                accuracy=0.0,
                average_latency_ms=0.0,
                p95_latency_ms=0.0,
                average_confidence=0.0,
                false_high_confidence_count=0,
            )

        latencies = sorted(
            result.latency_ms
            for result in results
        )

        correct = sum(
            result.correct
            for result in results
        )

        p95_index = min(
            len(latencies) - 1,
            max(0, int(len(latencies) * 0.95) - 1),
        )

        return RoutingEvaluationSummary(
            provider=self._provider.name,
            total_cases=len(results),
            correct_cases=correct,
            accuracy=correct / len(results),
            average_latency_ms=(
                sum(latencies) / len(latencies)
            ),
            p95_latency_ms=latencies[p95_index],
            average_confidence=(
                sum(result.confidence for result in results)
                / len(results)
            ),
            false_high_confidence_count=len(results) - correct,
        )