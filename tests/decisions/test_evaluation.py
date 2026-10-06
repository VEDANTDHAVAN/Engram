import pytest

from packages.decisions import (
    RuleBasedDecisionProvider,
)
from packages.decisions.evaluation import (
    DecisionRoutingEvaluator,
)

from .evaluation_cases import CASES


@pytest.mark.asyncio
async def test_rule_provider_evaluation() -> None:
    evaluator = DecisionRoutingEvaluator(
        RuleBasedDecisionProvider(),
    )

    results, summary = await evaluator.evaluate(
        CASES,
    )

    assert len(results) == len(CASES)

    assert summary.total_cases == 10
    assert summary.correct_cases == 10

    assert summary.accuracy == 1.0

    assert summary.average_latency_ms >= 0.0
    assert summary.p95_latency_ms >= 0.0

    assert 0.0 <= summary.average_confidence <= 1.0

    assert (
        summary.false_high_confidence_count == 0
    )


@pytest.mark.asyncio
async def test_evaluation_is_deterministic() -> None:
    provider = RuleBasedDecisionProvider()

    evaluator = DecisionRoutingEvaluator(
        provider,
    )

    first_results, first_summary = (
        await evaluator.evaluate(CASES)
    )

    second_results, second_summary = (
        await evaluator.evaluate(CASES)
    )

    first_predictions = [
        (
            result.query,
            result.predicted,
            result.correct,
        )
        for result in first_results
    ]

    second_predictions = [
        (
            result.query,
            result.predicted,
            result.correct,
        )
        for result in second_results
    ]

    assert first_predictions == second_predictions

    assert (
        first_summary.total_cases
        == second_summary.total_cases
    )

    assert (
        first_summary.correct_cases
        == second_summary.correct_cases
    )


@pytest.mark.asyncio
async def test_evaluation_preserves_case_order() -> None:
    evaluator = DecisionRoutingEvaluator(
        RuleBasedDecisionProvider(),
    )

    results, _ = await evaluator.evaluate(CASES)

    assert [
        result.query
        for result in results
    ] == [
        case.query
        for case in CASES
    ]