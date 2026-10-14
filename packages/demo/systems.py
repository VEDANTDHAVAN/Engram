"""The three memory architectures under comparison.

All three share one ``DemoRuntime`` (same memories, same namespace, same
embedding model, same LLM, same tools). Only the memory path differs:

``NoMemorySystem``
    Query -> LLM. No retrieval, no memory, no externalization.

``ConventionalMemorySystem``
    Dense similarity search -> fixed top-K -> direct prompt injection.

``EngramSystem``
    Decision -> retrieval policy -> dense + sparse -> fusion -> canonical
    memory -> context management -> DTOC -> lightweight context package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from packages.context.models import ContextArtifact, ContextArtifactState
from packages.context.dtoc import DTOCEngine, DTOCPolicy
from packages.decisions.base import DecisionProvider
from packages.llm.base import BaseLLMProvider
from packages.providers.memory import MemoryRecord
from packages.retrieval.models import RetrievalEvidence, RetrievalQuery

from .config import DemoConfig
from .interfaces import (
    MemorySystem,
    MemoryTrace,
    SystemName,
    SystemResponse,
    TimedCall,
    count_tokens,
)


SYSTEM_BANNER = "=" * 58


def format_memory_block(
    memories: list[MemoryRecord],
) -> str:
    """Render retrieved memories as plainly labelled evidence."""
    if not memories:
        return "(no memory evidence available)"
    lines = ["RETRIEVED MEMORY:"]
    for memory in memories:
        lines.append(f"- {memory.content}")
    return "\n".join(lines)


def format_tool_block(
    tool_name: str,
    output: str,
) -> str:
    """Render full tool output for the architectures that do not externalize."""
    return f"TOOL OUTPUT ({tool_name}):\n{output}"


class NoMemorySystem(MemorySystem):
    """Level 1: a genuinely stateless agent.

    It receives the current query and the same system prompt as the other two
    systems, plus the same tool output. Nothing is retrieved and nothing is
    hardcoded. It is not made artificially worse.
    """

    name = SystemName.NO_MEMORY

    def __init__(
        self,
        *,
        runtime: object,
        llm_provider: BaseLLMProvider,
        config: DemoConfig,
    ) -> None:
        self._runtime = runtime
        self._llm = llm_provider
        self._config = config

    @property
    def uses_memory(self) -> bool:
        return False

    @property
    def uses_externalization(self) -> bool:
        return False

    async def answer(
        self,
        query: str,
        *,
        namespace: str,
        tool_output: str | None = None,
        artifact_id: str | None = None,
        restore_artifact_id: str | None = None,
    ) -> SystemResponse:
        del namespace, artifact_id, restore_artifact_id

        started = perf_counter()
        trace = MemoryTrace()

        sections: list[str] = []
        if tool_output:
            sections.append(
                format_tool_block(self._config.tool_name, tool_output)
            )
        sections.append(f"QUESTION:\n{query}")

        prompt = "\n\n".join(sections)

        trace.tool_output_tokens = (
            count_tokens(tool_output) if tool_output else 0
        )
        trace.exposed_tool_tokens = trace.tool_output_tokens
        trace.final_context_tokens = count_tokens(prompt)

        response = await self._llm.generate(
            prompt,
            system_prompt=self._config.system_prompt,
            temperature=self._config.temperature,
        )

        return SystemResponse(
            system=self.name,
            answer=response.content,
            prompt=prompt,
            latency_ms=(perf_counter() - started) * 1000,
            trace=trace,
        )


class ConventionalMemorySystem(MemorySystem):
    """Level 2: semantic memory with fixed retrieval.

    A reasonable production-style baseline: persistent memories, one dense
    retrieval mechanism, a fixed top-K, and direct prompt injection. It has no
    decision layer, no sparse retrieval, no fusion, no temporal handling and no
    context optimization.
    """

    name = SystemName.CONVENTIONAL_MEMORY

    def __init__(
        self,
        *,
        runtime: object,
        llm_provider: BaseLLMProvider,
        config: DemoConfig,
    ) -> None:
        self._runtime = runtime
        self._llm = llm_provider
        self._config = config

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_externalization(self) -> bool:
        return False

    async def answer(
        self,
        query: str,
        *,
        namespace: str,
        tool_output: str | None = None,
        artifact_id: str | None = None,
        restore_artifact_id: str | None = None,
    ) -> SystemResponse:
        del artifact_id, restore_artifact_id

        started = perf_counter()
        trace = MemoryTrace()

        retrieval_query = RetrievalQuery(
            query=query, namespace=namespace, top_k=self._config.top_k,
        )

        with TimedCall(trace, "retrieval_latency_ms"):
            result = await self._runtime.retrieval.retrieve(
                retrieval_query, strategies=["vector"], fusion="rrf",
            )

        evidence = result.evidence
        memories = await self._hydrate(evidence, namespace)

        trace.strategies = ["vector"]
        trace.fusion = None
        trace.decision_provider = None
        trace.retrieved_memory_ids = [
            memory.metadata.get("dataset_id", str(memory.id))
            for memory in memories
        ]
        trace.retrieved_memory_count = len(memories)
        trace.retrieved_memory_tokens = sum(
            count_tokens(memory.content) for memory in memories
        )

        sections: list[str] = []
        if tool_output:
            sections.append(
                format_tool_block(self._config.tool_name, tool_output)
            )
        sections.append(format_memory_block(memories))
        sections.append(f"QUESTION:\n{query}")

        prompt = "\n\n".join(sections)

        trace.tool_output_tokens = (
            count_tokens(tool_output) if tool_output else 0
        )
        trace.exposed_tool_tokens = trace.tool_output_tokens
        trace.final_context_tokens = count_tokens(prompt)

        response = await self._llm.generate(
            prompt,
            system_prompt=self._config.system_prompt,
            temperature=self._config.temperature,
        )

        return SystemResponse(
            system=self.name,
            answer=response.content,
            prompt=prompt,
            latency_ms=(perf_counter() - started) * 1000,
            trace=trace,
        )

    async def _hydrate(
        self,
        evidence: list[RetrievalEvidence],
        namespace: str,
    ) -> list[MemoryRecord]:
        from uuid import UUID

        memories: list[MemoryRecord] = []
        for item in evidence:
            try:
                memory_id = UUID(str(item.id))
            except (ValueError, TypeError):
                continue
            memory = await self._runtime.memory.get(
                memory_id, namespace=namespace,
            )
            if memory is not None:
                memories.append(memory)
        return memories


class EngramSystem(MemorySystem):
    """Level 3: memory and context actively managed.

    Decision -> retrieval policy -> dense + sparse -> fusion -> canonical
    memory hydration -> temporal/lifecycle ranking -> context management ->
    DTOC -> lightweight context package.
    """

    name = SystemName.ENGRAM

    def __init__(
        self,
        *,
        runtime: object,
        llm_provider: BaseLLMProvider,
        config: DemoConfig,
        decision_provider: DecisionProvider,
    ) -> None:
        self._runtime = runtime
        self._llm = llm_provider
        self._config = config
        self._decision_provider = decision_provider
        self._dtoc = DTOCEngine(runtime.artifact_manager)

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_externalization(self) -> bool:
        return True

    @property
    def decision_provider_name(self) -> str:
        return self._decision_provider.name

    async def answer(
        self,
        query: str,
        *,
        namespace: str,
        tool_output: str | None = None,
        artifact_id: str | None = None,
        restore_artifact_id: str | None = None,
    ) -> SystemResponse:
        started = perf_counter()
        trace = MemoryTrace()

        sections: list[str] = []

        # ---- 1. Decision ------------------------------------------------
        decision = await self._decide(query, namespace, trace)
        trace.strategies = [
            strategy.value for strategy in decision.decision.strategies
        ]
        trace.fusion = decision.decision.fusion.value

        # ---- 2. Restore on explicit request -----------------------------
        restored_output: str | None = None
        if restore_artifact_id:
            restored_output = await self._restore(
                restore_artifact_id, namespace, trace
            )
            sections.append(restored_output)

        # ---- 3. Retrieval + fusion --------------------------------------
        memories = await self._retrieve(query, namespace, decision, trace)

        # ---- 4. Temporal / lifecycle ranking ----------------------------
        ranked = self._apply_lifecycle(memories, trace)
        trace.retrieved_memory_ids = [
            memory.metadata.get("dataset_id", str(memory.id))
            for memory in ranked
        ]
        trace.retrieved_memory_count = len(ranked)
        trace.retrieved_memory_tokens = sum(
            count_tokens(memory.content) for memory in ranked
        )

        # ---- 5. Tool output externalization -----------------------------
        exposed_tool_output = tool_output
        if tool_output:
            exposed_tool_output = await self._externalize(
                tool_output, namespace, artifact_id, trace
            )

        # ---- 6. Context packaging ---------------------------------------
        if exposed_tool_output:
            sections.append(exposed_tool_output)
        sections.append(format_memory_block(ranked))
        if not restore_artifact_id:
            sections.append(f"QUESTION:\n{query}")
        else:
            sections.append(format_memory_block(ranked))

        prompt = "\n\n".join(sections)
        trace.final_context_tokens = count_tokens(prompt)

        response = await self._llm.generate(
            prompt,
            system_prompt=self._config.system_prompt,
            temperature=self._config.temperature,
        )

        return SystemResponse(
            system=self.name,
            answer=response.content,
            prompt=prompt,
            latency_ms=(perf_counter() - started) * 1000,
            trace=trace,
        )

    # -- internals ------------------------------------------------------

    async def _decide(self, query: str, namespace: str, trace: MemoryTrace):
        from packages.decisions.models import DecisionRequest

        request = DecisionRequest(
            query=query,
            namespace=namespace,
            top_k=self._config.top_k,
        )
        with TimedCall(trace, "decision_latency_ms"):
            return await self._decision_provider.decide(request)

    async def _retrieve(
        self,
        query: str,
        namespace: str,
        decision,
        trace: MemoryTrace,
    ) -> list[MemoryRecord]:
        from uuid import UUID

        retrieval_query = RetrievalQuery(
            query=query,
            namespace=namespace,
            top_k=decision.decision.top_k,
        )

        with TimedCall(trace, "retrieval_latency_ms"):
            result = await self._runtime.retrieval.retrieve_with_policy(
                retrieval_query, decision.decision,
            )

        memories: list[MemoryRecord] = []
        for item in result.evidence:
            try:
                memory_id = UUID(str(item.id))
            except (ValueError, TypeError):
                continue
            memory = await self._runtime.memory.get(
                memory_id, namespace=namespace,
            )
            if memory is not None:
                memories.append(memory)
        return memories

    @staticmethod
    def _apply_lifecycle(
        memories: list[MemoryRecord],
        trace: MemoryTrace,
    ) -> list[MemoryRecord]:
        """Prefer currently-valid memories over superseded ones.

        This is the only temporal/lifecycle capability that exists in the
        codebase today; ``packages/temporal`` and ``packages/lifecycle`` are
        empty. Superseded evidence is demoted rather than deleted, so a
        historical question can still surface it.
        """
        def rank(memory: MemoryRecord) -> tuple[int, int]:
            state = str(memory.metadata.get("state", "active"))
            return (0 if state == "superseded" else 1, 0)

        return sorted(memories, key=rank, reverse=True)

    async def _externalize(
        self,
        tool_output: str,
        namespace: str,
        artifact_id: str | None,
        trace: MemoryTrace,
    ) -> str:
        identifier = artifact_id or self._config.artifact_id
        token_count = count_tokens(tool_output)
        now = datetime.now(timezone.utc)

        artifact = ContextArtifact(
            id=identifier,
            content=tool_output,
            token_count=token_count,
            source=self._config.tool_name,
            namespace=namespace,
            created_at=now,
            metadata={"dataset_id": identifier},
        )

        with TimedCall(trace, "artifact_store_latency_ms"):
            await self._runtime.artifact_manager.add_artifact(
                artifact=artifact,
            )

        # DTOC decides whether the full body must leave the active context.
        with TimedCall(trace, "context_optimization_latency_ms"):
            policy = DTOCPolicy(
                max_active_tokens=min(
                    self._config.max_active_artifact_tokens,
                    self._config.available_context_tokens,
                ),
                min_tokens_to_hide=1,
            )
            dtoc_result = await self._dtoc.optimize(
                namespace=namespace, policy=policy,
            )

        trace.tool_output_tokens = token_count
        trace.artifact_id = identifier
        trace.artifact_state = (
            ContextArtifactState.HIDDEN.value
            if identifier in dtoc_result.hidden_artifact_ids
            else ContextArtifactState.ACTIVE.value
        )
        trace.artifact_restorable = True
        trace.tokens_saved = dtoc_result.tokens_saved

        if trace.artifact_state == ContextArtifactState.HIDDEN.value:
            exposed = (
                f"Tool output stored as artifact:\n"
                f"artifact_id = {identifier}\n"
                f"token_count = {token_count}\n"
                f"state = hidden\n"
                f"Retrieve it with: retrieve_artifact({identifier!r})"
            )
        else:
            exposed = format_tool_block(
                self._config.tool_name, tool_output
            )

        trace.exposed_tool_tokens = count_tokens(exposed)
        return exposed

    async def _restore(
        self,
        identifier: str,
        namespace: str,
        trace: MemoryTrace,
    ) -> str:
        """Restore the complete original artifact.

        Restoration reads the canonical content back from the store. It is
        never regenerated or summarized, so ``hidden != deleted``.
        """
        with TimedCall(trace, "artifact_store_latency_ms"):
            artifact = await self._runtime.artifact_manager.get_artifact(
                artifact_id=identifier, namespace=namespace,
            )

        trace.artifact_id = identifier
        trace.artifact_state = artifact.state.value
        trace.artifact_restorable = True
        trace.exposed_tool_tokens = count_tokens(artifact.content)
        trace.tool_output_tokens = count_tokens(artifact.content)
        return (
            f"RETRIEVED ARTIFACT {identifier} "
            f"({artifact.token_count} tokens):\n{artifact.content}"
        )