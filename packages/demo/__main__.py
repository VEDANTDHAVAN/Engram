"""Run the interactive Engram engineering-agent demo."""

from __future__ import annotations

import asyncio

from .agent import AgentMode, EngramEngineeringAgent
from .runtime import create_demo_runtime


def _print_trace(response) -> None:
    trace = response.trace
    print("\n" + "=" * 60)
    print("ENGRAM TRACE")
    print("=" * 60)
    print(f"Decision: {trace.decision_provider or 'not used'}")
    print(f"Strategies: {', '.join(trace.strategies) or 'not used'}")
    print(f"Retrieved memories: {trace.retrieved_memory_count}")
    print(f"Context tokens: {trace.context_tokens}")
    if trace.artifact_id:
        print(f"Artifact: {trace.artifact_id}")
        print(f"Raw tool tokens: {trace.raw_tool_tokens}")
        print(f"Tokens exposed to LLM: {trace.exposed_tool_tokens}")
        print(f"Artifact recoverable: {'YES' if trace.artifact_restorable else 'NO'}")
    print("=" * 60)


async def main() -> None:
    print("╔════════════════════════════════════════════╗")
    print("║ ENGRAM — HALF-WAY PRODUCTION VERTICAL SLICE ║")
    print("╚════════════════════════════════════════════╝")
    selected = input("Mode [baseline/engram] (engram): ").strip().lower() or "engram"
    mode = AgentMode(selected)
    runtime = create_demo_runtime()
    await runtime.initialize()
    await runtime.seed_taskflow()
    agent = EngramEngineeringAgent(runtime, mode=mode)
    session_id = "interactive"
    print("Type /new to end the session, /quit to exit.")
    try:
        while True:
            query = input("\nUser > ").strip()
            if query == "/quit":
                return
            if query == "/new":
                agent.end_session(session_id)
                session_id = "interactive-next"
                print("Started a new session.")
                continue
            if not query:
                continue
            response = await agent.ask(session_id=session_id, query=query)
            print(f"\nAgent > {response.content}")
            _print_trace(response)
    finally:
        runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
