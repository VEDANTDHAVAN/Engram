import asyncio

from packages.decisions import (
    DecisionRoutingEvaluator,
    RuleBasedDecisionProvider,
)
from tests.decisions.evaluation_cases import CASES


async def main() -> None:
    provider = RuleBasedDecisionProvider()

    evaluator = DecisionRoutingEvaluator(provider)

    _, summary = await evaluator.evaluate(CASES)

    print()
    print("Decision Routing Evaluation")
    print("---------------------------")
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


if __name__ == "__main__":
    asyncio.run(main())