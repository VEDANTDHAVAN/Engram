"""Run the three-way memory-system benchmark and print the full report.

Usage:
    python -m scripts.evaluate_memory_systems [--llm echo|groq] [--no-cases]

The evaluator drives all three systems through the ``MemorySystem`` interface.
No per-system logic lives here; the report is rendered from the metrics the
evaluator produced.
"""

from __future__ import annotations

import argparse
import asyncio

from packages.demo import (
    DemoConfig,
    MemorySystemEvaluator,
    build_memory_systems,
    create_benchmark_runtime,
    create_llm_provider,
    load_evaluation_cases,
    render_all,
)


async def run(args: argparse.Namespace) -> None:
    config = DemoConfig()
    llm_provider = create_llm_provider(live=args.llm == "groq")

    runtime = await create_benchmark_runtime(
        config=config,
        llm_provider=llm_provider,
        deterministic_embeddings=args.llm != "groq",
    )
    try:
        systems = build_memory_systems(runtime, config)
        cases = load_evaluation_cases(config.evaluation_path)
        evaluator = MemorySystemEvaluator(
            config=config, concurrency=args.concurrency,
        )
        result = await evaluator.run(systems, cases)
    finally:
        runtime.close()

    print(
        render_all(
            result,
            config=config,
            cases=list(cases),
            show_cases=not args.no_cases,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way memory-system benchmark (No Memory vs "
        "Conventional Memory vs Engram).",
    )
    parser.add_argument(
        "--llm",
        choices=("echo", "groq"),
        default="echo",
        help="LLM provider: 'echo' (deterministic, default) or 'groq' (live).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent LLM calls per system (default 4).",
    )
    parser.add_argument(
        "--no-cases",
        action="store_true",
        help="Skip the per-case section (aggregate only).",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()