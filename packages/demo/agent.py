"""A small agent loop that demonstrates Engram as middleware, not an agent replacement."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter

from packages.context import ContextArtifact, DTOCEngine, DTOCPolicy
from packages.decisions import DecisionRequest, RuleBasedDecisionProvider
from packages.retrieval.models import RetrievalEvidence, RetrievalQuery

from .dataset import large_codebase_output, relevant_tool_evidence
from .runtime import DemoRuntime


SYSTEM_PROMPT = """You are the Engram Engineering Agent, a precise AI engineering assistant.
Answer only from the supplied conversation, tool output, and Engram evidence. If the information is absent, say so. Cite Engram evidence by its [memory:<id>] marker when available. Explain conflicts by preferring evidence marked active over superseded evidence."""


class AgentMode(str, Enum):
    BASELINE = "baseline"
    ENGRAM = "engram"


@dataclass(slots=True)
class EngramTrace:
    decision_provider: str | None = None
    decision_latency_ms: float | None = None
    strategies: list[str] = field(default_factory=list)
    fusion: str | None = None
    retrieved_memory_ids: list[str] = field(default_factory=list)
    retrieved_memory_count: int = 0
    context_tokens: int = 0
    raw_tool_tokens: int = 0
    exposed_tool_tokens: int = 0
    artifact_id: str | None = None
    artifact_restorable: bool | None = None


@dataclass(slots=True)
class AgentResponse:
    content: str
    mode: AgentMode
    trace: EngramTrace
    latency_ms: float


def _tokens(content: str) -> int:
    return len(content.split())


class EngramEngineeringAgent:
    """One controlled agent loop shared by baseline and Engram modes."""

    def __init__(self, runtime: DemoRuntime, *, mode: AgentMode) -> None:
        self._runtime = runtime
        self._mode = mode
        self._conversations: dict[str, list[str]] = {}

    async def ask(self, *, session_id: str, query: str) -> AgentResponse:
        started = perf_counter()
        trace = EngramTrace()
        conversation = self._conversations.setdefault(session_id, [])
        engram_context = ""
        tool_context = ""

        if self._mode == AgentMode.ENGRAM:
            engram_context = await self._retrieve_context(query, trace)
            if "remember" in query.lower():
                await self._runtime.store_memory(
                    content=query,
                    memory_type="episodic",
                    metadata={"topic": "session-decision", "state": "active"},
                )

        if self._needs_codebase_search(query):
            tool_context = await self._run_search_codebase(trace)

        prompt = self._build_prompt(
            query=query,
            conversation=conversation,
            engram_context=engram_context,
            tool_context=tool_context,
        )
        trace.context_tokens = _tokens(prompt)
        response = await self._runtime.llm_provider.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
        )
        conversation.extend([f"User: {query}", f"Agent: {response.content}"])
        return AgentResponse(
            content=response.content,
            mode=self._mode,
            trace=trace,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def end_session(self, session_id: str) -> None:
        """Clear only agent-local conversation; Engram durable memory remains intact."""
        self._conversations.pop(session_id, None)

    async def _retrieve_context(self, query: str, trace: EngramTrace) -> str:
        decision_started = perf_counter()
        try:
            decision = await self._runtime.decision_provider.decide(
                DecisionRequest(query=query, namespace=self._runtime.namespace, top_k=6)
            )
        except Exception:
            decision = await RuleBasedDecisionProvider().decide(
                DecisionRequest(query=query, namespace=self._runtime.namespace, top_k=6)
            )
        trace.decision_latency_ms = (perf_counter() - decision_started) * 1000
        trace.decision_provider = decision.provider
        trace.strategies = [strategy.value for strategy in decision.decision.strategies]
        trace.fusion = decision.decision.fusion.value
        result = await self._runtime.retrieval.retrieve_with_policy(
            RetrievalQuery(query=query, namespace=self._runtime.namespace, top_k=6),
            decision.decision,
        )
        trace.retrieved_memory_ids = [evidence.id for evidence in result.evidence]
        trace.retrieved_memory_count = len(result.evidence)
        return self._format_evidence(result.evidence)

    async def _run_search_codebase(self, trace: EngramTrace) -> str:
        raw_output = large_codebase_output()
        trace.raw_tool_tokens = _tokens(raw_output)
        if self._mode == AgentMode.BASELINE:
            trace.exposed_tool_tokens = trace.raw_tool_tokens
            return f"\nTOOL search_codebase:\n{raw_output}\n"

        artifact_id = "tool_search_codebase_001"
        artifact = ContextArtifact(
            id=artifact_id,
            content=raw_output,
            token_count=trace.raw_tool_tokens,
            created_at=datetime.now(timezone.utc),
            source="search_codebase",
            namespace=self._runtime.namespace,
            metadata={"query": "authentication backend changes"},
        )
        await self._runtime.artifact_manager.add_artifact(artifact=artifact)
        await DTOCEngine(self._runtime.artifact_manager).optimize(
            namespace=self._runtime.namespace,
            policy=DTOCPolicy(max_active_tokens=128),
        )
        await self._runtime.artifact_manager.restore(
            namespace=self._runtime.namespace,
            artifact_id=artifact_id,
        )
        restored = await self._runtime.artifact_manager.get_artifact(
            namespace=self._runtime.namespace,
            artifact_id=artifact_id,
        )
        await self._runtime.artifact_manager.hide(
            namespace=self._runtime.namespace,
            artifact_id=artifact_id,
        )
        trace.artifact_id = artifact_id
        trace.artifact_restorable = restored.content == raw_output
        reference = (
            f"[Tool output stored as artifact {artifact_id}; "
            f"tokens: {trace.raw_tool_tokens}; state: hidden; source: search_codebase]"
        )
        evidence = relevant_tool_evidence()
        trace.exposed_tool_tokens = _tokens(reference) + _tokens(evidence)
        return f"\nTOOL REFERENCE: {reference}\nRELEVANT TOOL EVIDENCE: {evidence}\n"

    @staticmethod
    def _needs_codebase_search(query: str) -> bool:
        return "recent" in query.lower() and "backend" in query.lower()

    @staticmethod
    def _format_evidence(evidence: list[RetrievalEvidence]) -> str:
        if not evidence:
            return ""
        lines = ["\nENGRAM EVIDENCE:"]
        for item in evidence:
            state = str(item.metadata.get("state", "active"))
            lines.append(
                f"[memory:{item.id}; state:{state}; source:{item.source}] {item.content}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_prompt(
        *,
        query: str,
        conversation: list[str],
        engram_context: str,
        tool_context: str,
    ) -> str:
        conversation_text = "\n".join(conversation) or "(new session)"
        return (
            f"CURRENT CONVERSATION:\n{conversation_text}\n"
            f"{engram_context}{tool_context}\n"
            f"CURRENT USER REQUEST:\n{query}\n"
        )
