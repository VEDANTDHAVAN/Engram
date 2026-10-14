"""Live three-way demonstration: the same question, three architectures.

Usage:
    python -m scripts.demo_comparison [--llm echo|groq]

Runs a curated sequence of questions side-by-side across No Memory,
Conventional Memory and Engram, then finishes with the aggregate benchmark.
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
    load_tool_output,
    render_all,
)

# Curated live sequence (a subset of the benchmark cases, in narrative order).
LIVE_SEQUENCE = [
    "case_01_knowledge",
    "case_02_exact_identifier",
    "case_04_semantic",
    "case_05_temporal_current",
    "case_07_conflict",
    "case_09_large_tool_output",
    "case_10_artifact_restoration",
]


def _box(question: str) -> str:
    width = 60
    top = "+" + "-" * width + "+"
    bottom = "+" + "-" * width + "+"
    middle = "+" + "-" * width + "+"
    inner = f" QUESTION".ljust(width)
    return "\n".join([top, inner, middle, question[:width].ljust(width), bottom])


async def run(
    args: argparse.Namespace,
    tool_output: str | None,
) -> None:
    config = DemoConfig()
    llm_provider = create_llm_provider(live=args.llm == "groq")

    runtime = await create_benchmark_runtime(
        config=config,
        llm_provider=llm_provider,
        deterministic_embeddings=args.llm != "groq",
    )
    try:
        systems = build_memory_systems(runtime, config)
        cases = list(load_evaluation_cases(config.evaluation_path))
        by_id = {case["id"]: case for case in cases}
        live_cases = [by_id[cid] for cid in LIVE_SEQUENCE if cid in by_id]

        print("\n" + "=" * 60)
        print("ENGRAM — THREE-WAY LIVE COMPARISON".center(60))
        print("=" * 60)
        print("Same Agent + Same LLM + Same Tools + Same Dataset")
        print("Different memory architecture.\n")

        for index, case in enumerate(live_cases, start=1):
            print(_box(case["question"]))
            print(f"\n[step {index}/{len(live_cases)}]")
            for system in systems:
                response = await system.answer(
                    case["question"],
                    namespace=config.namespace,
                    tool_output=(
                        tool_output if case.get("tool") else None
                    ),
                    artifact_id=(
                        config.artifact_id if case.get("tool") else None
                    ),
                    restore_artifact_id=case.get("requires_artifact"),
                )
                trace = response.trace
                print(f"\n{system.name.value.upper()}")
                print(f"  -> {response.answer.strip()[:200]}")
                if trace.strategies:
                    print(
                        f"  Retrieval: {' + '.join(trace.strategies)}"
                        + (f"  Fusion: {trace.fusion}" if trace.fusion else "")
                    )
                if trace.artifact_state:
                    print(
                        f"  Artifact: {trace.artifact_id} "
                        f"state={trace.artifact_state} "
                        f"exposed={trace.exposed_tool_tokens}tok"
                    )
                print(f"  Context: {trace.final_context_tokens} tokens")
            print()

        # Finish with the full aggregate benchmark.
        evaluator = MemorySystemEvaluator(
            config=config, concurrency=args.concurrency,
        )
        result = await evaluator.run(systems, cases)
        print(render_all(result, config=config, cases=cases, show_cases=False))
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live three-way memory-system comparison.",
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
        help="Concurrent LLM calls for the final benchmark (default 4).",
    )
    args = parser.parse_args()
    tool_output = load_tool_output()
    asyncio.run(run(args, tool_output))


if __name__ == "__main__":
    main()