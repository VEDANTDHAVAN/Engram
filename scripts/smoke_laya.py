from __future__ import annotations

import asyncio

from laya import Router

from packages.decisions import (
    DecisionRequest,
    LayaDecisionProvider,
)


async def main() -> None:
    router = Router()

    provider = LayaDecisionProvider(
        router=router,
    )

    queries = [
        (
            "Why are embeddings useful for semantic retrieval?",
            "vector",
        ),
        (
            "Find ValueError in auth.py",
            "sparse",
        ),
    ]

    print("Laya smoke test")
    print("---------------")

    for query, expected in queries:
        print()
        print(f"Query:")
        print(query)

        response = await provider.decide(
            DecisionRequest(
                query=query,
                namespace="smoke",
            )
        )

        print()
        print(f"Provider:")
        print(response.provider)

        print()
        print(f"Decision:")
        print(
            ", ".join(
                strategy.value
                for strategy in response.decision.strategies
            )
        )

        print()
        print(f"Expected:")
        print(expected)

        print()
        print(f"Confidence:")
        print(f"{response.confidence:.4f}")

        print()
        print(f"Model:")
        print(response.model)

        print()
        print(f"Latency:")
        print(f"{response.latency_ms:.3f} ms")

        print()
        print(f"Reasons:")
        for reason in response.reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    asyncio.run(main())