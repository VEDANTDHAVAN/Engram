from __future__ import annotations

from packages.demo import AgentMode, EngramEngineeringAgent, create_demo_runtime
from packages.llm.base import BaseLLMProvider
from packages.llm.schemas import ChatResponse


class PromptEchoLLM(BaseLLMProvider):
    """Deterministic test provider that exposes exactly what the agent supplied."""

    name = "prompt-echo"

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ChatResponse:
        del system_prompt, kwargs
        return ChatResponse(content=prompt, model="prompt-echo", provider=self.name)

    async def stream(self, prompt: str, system_prompt: str | None = None, **kwargs):
        del prompt, system_prompt, kwargs
        if False:
            yield ""


async def test_engram_retrieves_cross_session_memory_while_baseline_does_not():
    runtime = create_demo_runtime(llm_provider=PromptEchoLLM())
    await runtime.initialize()
    try:
        await runtime.seed_taskflow()
        baseline = EngramEngineeringAgent(runtime, mode=AgentMode.BASELINE)
        engram = EngramEngineeringAgent(runtime, mode=AgentMode.ENGRAM)
        lesson = (
            "TaskFlow uses cursor pagination because the system has more than "
            "100,000 tasks. Remember these decisions."
        )

        await baseline.ask(session_id="baseline-teach", query=lesson)
        await engram.ask(session_id="engram-teach", query=lesson)
        baseline.end_session("baseline-teach")
        engram.end_session("engram-teach")

        question = "Why did we choose cursor pagination for TaskFlow?"
        baseline_response = await baseline.ask(
            session_id="baseline-new", query=question,
        )
        engram_response = await engram.ask(
            session_id="engram-new", query=question,
        )

        assert "100,000" not in baseline_response.content
        assert "100,000" in engram_response.content
        assert engram_response.trace.retrieved_memory_count > 0
        assert engram_response.trace.decision_provider == "rules"
    finally:
        runtime.close()


async def test_large_tool_output_is_hidden_and_restorable_with_engram():
    runtime = create_demo_runtime(llm_provider=PromptEchoLLM())
    await runtime.initialize()
    try:
        await runtime.seed_taskflow()
        agent = EngramEngineeringAgent(runtime, mode=AgentMode.ENGRAM)
        response = await agent.ask(
            session_id="tool-session",
            query=(
                "Analyze the recent TaskFlow backend changes and identify whether "
                "any authentication-related changes could violate our security requirements."
            ),
        )

        assert response.trace.artifact_id == "tool_search_codebase_001"
        assert response.trace.raw_tool_tokens > response.trace.exposed_tool_tokens
        assert response.trace.artifact_restorable is True
        assert "Tool output stored as artifact" in response.content
        artifact = await runtime.artifact_manager.get_artifact(
            namespace=runtime.namespace,
            artifact_id=response.trace.artifact_id,
        )
        assert "refresh_token" in artifact.content
    finally:
        runtime.close()
