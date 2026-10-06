"""Run the same TaskFlow prompts through baseline and Engram agent modes."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from packages.demo import (
    AgentMode,
    DemoEchoLLMProvider,
    EngramEngineeringAgent,
    create_demo_runtime,
)
from packages.llm import LLMFactory, LLMProvider
from packages.llm.base import BaseLLMProvider


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    query: str
    expected_terms: tuple[str, ...]
    category: str


CASES = (
    EvaluationCase("What database is the source of truth for TaskFlow?", ("postgresql",), "architecture"),
    EvaluationCase("Why did we move to cursor pagination?", ("100,000", "cursor"), "memory"),
    EvaluationCase("What authentication-token policy did we choose?", ("httponly", "short-lived"), "security"),
    EvaluationCase("What pagination approach is current?", ("cursor",), "temporal"),
    EvaluationCase("What pagination approach did TaskFlow originally use?", ("offset",), "temporal"),
    EvaluationCase("Should Redis replace PostgreSQL?", ("not", "canonical"), "architecture"),
    EvaluationCase("Analyze the recent TaskFlow backend changes and identify whether any authentication-related changes could violate our security requirements.", ("httponly", "json"), "security"),
)


async def run_mode(
    mode: AgentMode,
    llm_provider: BaseLLMProvider,
) -> dict[str, object]:
    runtime = create_demo_runtime(llm_provider=llm_provider)
    await runtime.initialize()
    await runtime.seed_taskflow()
    agent = EngramEngineeringAgent(runtime, mode=mode)
    outcomes = []
    try:
        for index, case in enumerate(CASES):
            response = await agent.ask(session_id=f"case-{index}", query=case.query)
            answer = response.content.lower()
            correct = all(term.lower() in answer for term in case.expected_terms)
            outcomes.append((case, response, correct))
    finally:
        runtime.close()

    return {
        "correct": sum(correct for _, _, correct in outcomes),
        "architecture": sum(correct for case, _, correct in outcomes if case.category == "architecture"),
        "architecture_total": sum(case.category == "architecture" for case, _, _ in outcomes),
        "temporal": sum(correct for case, _, correct in outcomes if case.category == "temporal"),
        "temporal_total": sum(case.category == "temporal" for case, _, _ in outcomes),
        "security": sum(correct for case, _, correct in outcomes if case.category == "security"),
        "security_total": sum(case.category == "security" for case, _, _ in outcomes),
        "average_context_tokens": round(sum(response.trace.context_tokens for _, response, _ in outcomes) / len(outcomes)),
        "average_latency_ms": round(sum(response.latency_ms for _, response, _ in outcomes) / len(outcomes), 1),
        "artifact_retained": any(response.trace.artifact_restorable for _, response, _ in outcomes),
        "outcomes": outcomes,
    }


def _ratio(value: int, total: int) -> str:
    return f"{value}/{total}"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engram A/B evaluation (baseline vs Engram).",
    )
    parser.add_argument(
        "--llm",
        choices=("echo", "groq"),
        default="echo",
        help="LLM provider: 'echo' (deterministic, default) or 'groq' (requires GROQ_API_KEY).",
    )
    args = parser.parse_args()

    llm_provider: BaseLLMProvider
    if args.llm == "groq":
        llm_provider = LLMFactory.create(LLMProvider.GROQ)
    else:
        llm_provider = DemoEchoLLMProvider()

    baseline, engram = await asyncio.gather(
        run_mode(AgentMode.BASELINE, llm_provider),
        run_mode(AgentMode.ENGRAM, llm_provider),
    )
    print("=" * 60)
    print("ENGRAM A/B EVALUATION")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'Engram':>12}")
    print("-" * 60)
    print(f"{'Memory questions correct':<30} {_ratio(baseline['correct'], len(CASES)):>12} {_ratio(engram['correct'], len(CASES)):>12}")
    print(f"{'Architecture consistency':<30} {_ratio(baseline['architecture'], baseline['architecture_total']):>12} {_ratio(engram['architecture'], engram['architecture_total']):>12}")
    print(f"{'Temporal correctness':<30} {_ratio(baseline['temporal'], baseline['temporal_total']):>12} {_ratio(engram['temporal'], engram['temporal_total']):>12}")
    print(f"{'Security consistency':<30} {_ratio(baseline['security'], baseline['security_total']):>12} {_ratio(engram['security'], engram['security_total']):>12}")
    print(f"{'Average context tokens':<30} {baseline['average_context_tokens']:>12} {engram['average_context_tokens']:>12}")
    print(f"{'Tool output retained':<30} {'No':>12} {'Yes' if engram['artifact_retained'] else 'No':>12}")
    print(f"{'Artifact restoration':<30} {'No':>12} {'Yes' if engram['artifact_retained'] else 'No':>12}")
    print(f"{'Average latency (ms)':<30} {baseline['average_latency_ms']:>12} {engram['average_latency_ms']:>12}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
