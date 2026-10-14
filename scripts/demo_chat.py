"""Interactive three-way memory architecture demo.

Run:

    uv run python scripts/demo_chat.py

Menu:
    1. No Memory
    2. Conventional Memory
    3. Engram
    4. Compare All
    5. Presentation Demo
    6. Metrics
    0. Exit

Slash commands inside a chat session:

    /help        show this help
    /architecture  describe the three architectures
    /metrics     show the last response's metrics
    /history     show the current session's questions
    /scenario    pick a canned scenario (Exact ID, Semantic, Temporal, ...)
    /clear       clear the session history
    /back        return to the main menu
    /exit        quit

Everything renders with plain ASCII so it is readable in PowerShell, Windows
Terminal, VS Code, and Linux/macOS terminals alike.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow ``python scripts/demo_chat.py`` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.demo.benchmark import (
    build_memory_systems,
    create_benchmark_runtime,
    create_llm_provider,
)
from packages.demo.config import DemoConfig
from packages.demo.dataset_loader import (
    load_evaluation_cases,
    load_tool_output,
)
from packages.demo.interfaces import MemorySystem, SystemName
from packages.demo.metrics import SystemReport
from packages.demo.report import (
    render_accuracy,
    render_context,
    render_dtoc,
    render_latency,
    render_systems,
)

WIDTH = 60


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _title(text: str) -> str:
    return f"\n{_rule('=')}\n{text.center(WIDTH)}\n{_rule('=')}"


def _box(text: str) -> str:
    lines = text.splitlines()
    top = "+" + "-" * (WIDTH - 2) + "+"
    out = [top]
    for line in lines:
        out.append("| " + line.ljust(WIDTH - 4) + " |")
    out.append(top)
    return "\n".join(out)


def _print_metrics(report: SystemReport) -> None:
    print(f"  Answer Accuracy:      {report.accuracy():.0%}")
    print(f"  Memory-Grounded:      {report.memory_grounded_accuracy() or 0.0:.0%}")
    print(f"  Unsupported:          {report.unsupported_answer_rate() or 0.0:.0%}")
    print(f"  Temporal Accuracy:    {report.temporal_accuracy() or 0.0:.0%}")
    print(f"  Conflict Accuracy:    {report.conflict_accuracy() or 0.0:.0%}")
    print(f"  Abstention Rate:      {report.abstention_rate():.0%}")
    print(f"  Avg Latency (ms):     {report.average_latency_ms:,.0f}")
    print(f"  Avg Context Tokens:   {report.average_final_context_tokens:,.0f}")


@dataclass
class ChatSession:
    """One interactive session against the three systems."""

    systems: tuple[MemorySystem, ...]
    config: DemoConfig
    history: list[str] = field(default_factory=list)

    @property
    def by_name(self) -> dict[SystemName, MemorySystem]:
        return {system.name: system for system in self.systems}

    async def ask(self, system: MemorySystem, question: str) -> None:
        response = await system.answer(
            question, namespace=self.config.namespace,
        )
        print(f"\n[{system.name.value.upper()}]")
        print(f"  {response.answer.strip()}")
        trace = response.trace
        if trace.strategies:
            print(f"  retrieval: {', '.join(trace.strategies)}")
        if trace.fusion:
            print(f"  fusion: {trace.fusion}")
        if trace.decision_provider:
            print(f"  decision: {trace.decision_provider}")
        if trace.artifact_state:
            print(
                f"  artifact: {trace.artifact_state} "
                f"(restorable={trace.artifact_restorable})"
            )
        print(
            f"  latency: {response.latency_ms:,.0f}ms  "
            f"context tokens: {trace.final_context_tokens}"
        )

    async def compare_all(self, question: str) -> None:
        print(_title("COMPARE ALL"))
        print(f"Question: {question}\n")
        for system in self.systems:
            await self.ask(system, question)
            print()
        self.history.append(question)

    async def run_scenario(self, index: int) -> None:
        cases = load_evaluation_cases()
        mapping = {
            1: "exact_identifier",
            2: "semantic_retrieval",
            3: "temporal_retrieval",
            4: "conflict_resolution",
            5: "large_tool_output",
            6: "hybrid_retrieval",
        }
        category = mapping.get(index)
        if category is None:
            print("Unknown scenario.")
            return
        case = next(
            (c for c in cases if c["category"] == category), None,
        )
        if case is None:
            print(f"No case for category {category!r}.")
            return
        question = case["question"]
        print(_title(f"SCENARIO {index} - {category}"))
        print(f"Question: {question}\n")
        for system in self.systems:
            if case.get("tool") == "search_codebase":
                response = await system.answer(
                    question,
                    namespace=self.config.namespace,
                    tool_output=load_tool_output(),
                    artifact_id=self.config.artifact_id,
                )
                print(f"\n[{system.name.value.upper()}]")
                print(f"  {response.answer.strip()}")
                trace = response.trace
                print(
                    f"  tool tokens: {trace.tool_output_tokens}  "
                    f"exposed: {trace.exposed_tool_tokens}  "
                    f"state: {trace.artifact_state}"
                )
            else:
                await self.ask(system, question)
            print()
        self.history.append(question)

    async def presentation(self) -> None:
        """Automated walkthrough: Exact ID -> Semantic -> Temporal -> Conflict
        -> Large Tool / DTOC -> Summary."""
        print(_title("PRESENTATION DEMO"))
        cases = load_evaluation_cases()
        order = [
            "exact_identifier",
            "semantic_retrieval",
            "temporal_retrieval",
            "conflict_resolution",
            "large_tool_output",
        ]
        for category in order:
            case = next(
                (c for c in cases if c["category"] == category), None,
            )
            if case is None:
                continue
            print(f"\n{_rule('=')}")
            print(f"STEP: {category}")
            print(f"Question: {case['question']}")
            print(_rule("="))
            for system in self.systems:
                if case.get("tool") == "search_codebase":
                    response = await system.answer(
                        case["question"],
                        namespace=self.config.namespace,
                        tool_output=load_tool_output(),
                        artifact_id=self.config.artifact_id,
                    )
                    print(f"\n[{system.name.value.upper()}]")
                    print(f"  {response.answer.strip()}")
                    trace = response.trace
                    print(
                        f"  tool tokens: {trace.tool_output_tokens}  "
                        f"exposed: {trace.exposed_tool_tokens}  "
                        f"state: {trace.artifact_state}"
                    )
                else:
                    await self.ask(system, case["question"])
                print()
            self.history.append(case["question"])

        print(_title("SUMMARY"))
        for system in self.systems:
            print(f"- {system.name.value.upper()}: {system.uses_memory=} {system.uses_externalization=}")
        print("\nPresentation complete.")


def _menu() -> None:
    print(_title("ENGRAM - THREE-WAY MEMORY DEMO"))
    print("1. No Memory")
    print("2. Conventional Memory")
    print("3. Engram")
    print("4. Compare All")
    print("5. Presentation Demo")
    print("6. Metrics")
    print("0. Exit")


async def _run() -> None:
    config = DemoConfig()
    llm = create_llm_provider(live=False)
    runtime = await create_benchmark_runtime(
        config=config,
        llm_provider=llm,
        deterministic_embeddings=True,
    )
    systems = build_memory_systems(runtime, config)
    session = ChatSession(systems=systems, config=config)

    while True:
        _menu()
        choice = input("\n> ").strip()
        if choice == "0":
            print("Bye.")
            break
        if choice == "1":
            await session.ask(session.by_name[SystemName.NO_MEMORY], input("Question: ").strip())
        elif choice == "2":
            await session.ask(session.by_name[SystemName.CONVENTIONAL_MEMORY], input("Question: ").strip())
        elif choice == "3":
            await session.ask(session.by_name[SystemName.ENGRAM], input("Question: ").strip())
        elif choice == "4":
            question = input("Question: ").strip() or "What pagination strategy does TaskFlow currently use?"
            await session.compare_all(question)
        elif choice == "5":
            await session.presentation()
        elif choice == "6":
            print(_title("METRICS"))
            print("Run the benchmark for full metrics:")
            print("  uv run python -m scripts.evaluate_memory_systems")
            print("  uv run python -m scripts.demo_comparison")
        else:
            print("Unknown choice.")


def main() -> None:
    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()