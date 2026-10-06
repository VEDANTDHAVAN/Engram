"""Provider-neutral composition root for the TaskFlow demonstration."""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from packages.context import (
    CacheContextArtifactStore,
    ContextManager,
    InMemoryContextStateStore,
)
from packages.decisions import DecisionProvider, RuleBasedDecisionProvider
from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.llm.base import BaseLLMProvider
from packages.llm.schemas import ChatResponse
from packages.providers.cache import CacheProvider
from packages.providers.memory import MemoryProvider, MemoryRecord
from packages.providers.sparse import BM25Provider
from packages.providers.vector import TurboVecProvider
from packages.retrieval.hybrid import HybridRetrievalService
from packages.retrieval.strategies import (
    SparseRetrievalStrategy,
    VectorRetrievalStrategy,
)

from .dataset import TASKFLOW_MEMORIES


class DemoMemoryProvider(MemoryProvider):
    """Small canonical-memory adapter used only by the reproducible demo."""

    def __init__(self) -> None:
        self._records: dict[UUID, MemoryRecord] = {}

    async def create(
        self,
        *,
        namespace: str,
        content: str,
        metadata: dict | None = None,
        memory_type: str,
    ) -> MemoryRecord:
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=uuid4(),
            namespace=namespace,
            content=content,
            memory_type=memory_type,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._records[record.id] = record
        return record

    async def get(
        self, memory_id: UUID, *, namespace: str,
    ) -> MemoryRecord | None:
        record = self._records.get(memory_id)
        return record if record is not None and record.namespace == namespace else None

    async def update(
        self,
        memory_id: UUID,
        *,
        namespace: str,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        record = await self.get(memory_id, namespace=namespace)
        if record is None:
            raise KeyError(f"Memory {memory_id} was not found.")
        updated = record.model_copy(
            update={
                "content": content if content is not None else record.content,
                "metadata": {**record.metadata, **(metadata or {})},
                "updated_at": datetime.now(timezone.utc),
            },
        )
        self._records[memory_id] = updated
        return updated

    async def delete(self, memory_id: UUID, *, namespace: str) -> None:
        record = await self.get(memory_id, namespace=namespace)
        if record is not None:
            del self._records[memory_id]

    async def list_ids(
        self, *, namespace: str, filters: dict | None = None,
    ) -> list[UUID]:
        records = [
            record for record in self._records.values()
            if record.namespace == namespace
        ]
        allowed_ids = (filters or {}).get("allowed_ids")
        if allowed_ids is not None:
            allowed = {str(value) for value in allowed_ids}
            records = [record for record in records if str(record.id) in allowed]
        return [record.id for record in records]


class DemoCacheProvider(CacheProvider):
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self._values.get(key)

    async def set(self, key: str, value: object, *, ttl: int | None = None) -> None:
        del ttl
        self._values[key] = value

    async def delete(self, key: str) -> None:
        self._values.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._values


class DemoHashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local embedding adapter so the demo needs no embedding API."""

    name = "demo-hash"
    dimensions = 64

    async def embed(self, text: str) -> EmbeddingResponse:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            values[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return EmbeddingResponse(
            embedding=[value / norm for value in values],
            dimensions=self.dimensions,
            model=self.name,
            provider=self.name,
        )


class DemoEchoLLMProvider(BaseLLMProvider):
    """Deterministic LLM stand-in that exposes exactly what the agent supplied.

    Returns the prompt as the response so the A/B evaluation is reproducible
    without an API key: the only difference between baseline and Engram mode
    is the context Engram placed in the prompt.
    """

    name = "demo-echo"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ChatResponse:
        del system_prompt, kwargs
        return ChatResponse(
            content=prompt,
            model="demo-echo",
            provider=self.name,
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ):
        del prompt, system_prompt, kwargs
        if False:
            yield ""


class DemoRuntime:
    """Wires existing Engram interfaces into a self-contained demo runtime."""

    def __init__(
        self,
        *,
        llm_provider: BaseLLMProvider,
        decision_provider: DecisionProvider | None = None,
    ) -> None:
        self.namespace = "taskflow-demo"
        self.llm_provider = llm_provider
        self.decision_provider = decision_provider or RuleBasedDecisionProvider()
        self.memory = DemoMemoryProvider()
        self.embedding = DemoHashEmbeddingProvider()
        self.sparse = BM25Provider()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="engram-demo-")
        self.vector = TurboVecProvider(
            dimension=self.embedding.dimensions,
            index_path=Path(self._temporary_directory.name) / "taskflow.tvim",
        )
        self.cache = DemoCacheProvider()
        self.artifact_manager = ContextManager(
            CacheContextArtifactStore(self.cache),
            InMemoryContextStateStore(),
        )
        self.retrieval = HybridRetrievalService(
            vector_strategy=VectorRetrievalStrategy(
                embedding_provider=self.embedding,
                vector_provider=self.vector,
                memory_provider=self.memory,
            ),
            sparse_strategy=SparseRetrievalStrategy(
                sparse_provider=self.sparse,
                memory_provider=self.memory,
            ),
        )

    async def initialize(self) -> None:
        await self.vector.initialize()
        await self.sparse.initialize()

    async def seed_taskflow(self) -> None:
        for demo_memory in TASKFLOW_MEMORIES:
            await self.store_memory(
                content=demo_memory.content,
                memory_type=demo_memory.memory_type,
                metadata=demo_memory.metadata,
            )

    async def store_memory(
        self,
        *,
        content: str,
        memory_type: str = "episodic",
        metadata: dict | None = None,
    ) -> MemoryRecord:
        memory = await self.memory.create(
            namespace=self.namespace,
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )
        embedding = await self.embedding.embed(memory.content)
        await self.vector.upsert(
            namespace=self.namespace,
            record_id=str(memory.id),
            embedding=embedding.embedding,
            content=memory.content,
            metadata=memory.metadata,
        )
        await self.sparse.upsert(
            namespace=self.namespace,
            record_id=str(memory.id),
            content=memory.content,
            metadata=memory.metadata,
        )
        return memory

    def close(self) -> None:
        self._temporary_directory.cleanup()


def create_demo_runtime(
    *,
    llm_provider: BaseLLMProvider | None = None,
    decision_provider: DecisionProvider | None = None,
) -> DemoRuntime:
    return DemoRuntime(
        llm_provider=(
            llm_provider
            if llm_provider is not None
            else DemoEchoLLMProvider()
        ),
        decision_provider=decision_provider,
    )
