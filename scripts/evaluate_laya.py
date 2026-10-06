from __future__ import annotations

import asyncio

from laya import Router

from packages.decisions import (
    DecisionRoutingEvaluator,
    LayaDecisionProvider,
)

from tests.decisions.evaluation_cases import CASES


async def main() -> None:
    router = Router()

    provider = LayaDecisionProvider(
        router=router,
    )

    evaluator = DecisionRoutingEvaluator(
        provider,
    )

    results, summary = await evaluator.evaluate(
        CASES,
    )

    print()
    print("=" * 70)
    print("LAYA DECISION ROUTING EVALUATION")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        expected = ", ".join(
            strategy.value
            for strategy in sorted(
                result.expected,
                key=lambda item: item.value,
            )
        )

        predicted = ", ".join(
            strategy.value
            for strategy in sorted(
                result.predicted,
                key=lambda item: item.value,
            )
        )

        print()
        print(f"Case {index}")
        print(f"Query:      {result.query}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {predicted}")
        print(f"Correct:    {result.correct}")
        print(
            f"Confidence: {result.confidence:.2%}"
        )
        print(
            f"Latency:    {result.latency_ms:.3f} ms"
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Provider:       {summary.provider}")
    print(f"Cases:          {summary.total_cases}")
    print(f"Correct:        {summary.correct_cases}")
    print(f"Accuracy:       {summary.accuracy:.2%}")
    print(
        f"Avg latency:    "
        f"{summary.average_latency_ms:.3f} ms"
    )
    print(
        f"P95 latency:    "
        f"{summary.p95_latency_ms:.3f} ms"
    )
    print(
        f"Avg confidence: "
        f"{summary.average_confidence:.2%}"
    )
    print(
        "False high-confidence: "
        f"{summary.false_high_confidence_count}"
    )


if __name__ == "__main__":
    asyncio.run(main())