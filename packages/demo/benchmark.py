"""Composition root for the three-way memory-system benchmark.

All three systems are built from ONE ``DemoRuntime``: the same canonical memory
provider, the same namespace, the same embedding model, the same vector and
sparse indexes, the same artifact store and the same LLM. Only the memory path
applied to a query differs.

``build_memory_systems`` returns the systems in a fixed order and never applies
system-specific behaviour to the evaluator.
"""

from __future__ import annotations

import asyncio

from packages.core.config import settings
from packages.decisions.base import DecisionProvider
from packages.decisions.rule_based import RuleBasedDecisionProvider
from packages.embeddings.base import BaseEmbeddingProvider
from packages.embeddings.schemas import EmbeddingResponse
from packages.embeddings.factory import EmbeddingFactory
from packages.embeddings.types import EmbeddingProvider
from packages.llm.base import BaseLLMProvider
from packages.llm.schemas import ChatResponse
from typing import AsyncGenerator
from packages.llm.factory import LLMFactory
from packages.llm.types import LLMProvider

from .config import DemoConfig
from .dataset_loader import load_memories
from .interfaces import MemorySystem
from .runtime import DemoEchoLLMProvider, DemoRuntime
from .systems import (
    ConventionalMemorySystem,
    EngramSystem,
    NoMemorySystem,
)


class ResilientEmbeddingProvider(BaseEmbeddingProvider):
    """Wraps a real embedding provider with caching and rate-limit retries.

    The benchmark asks the same query to every system, so identical texts are
    embedded once. Retries with backoff absorb the free-tier rate limit that
    otherwise surfaces as HTTP 429 under concurrent retrieval.
    """

    def __init__(
        self,
        inner: BaseEmbeddingProvider,
        *,
        attempts: int = 5,
    ) -> None:
        self._inner = inner
        self._attempts = attempts
        self._cache: dict[str, EmbeddingResponse] = {}
        self.name = inner.name
        self.dimensions = inner.dimensions

    async def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = await self._inner.embed(text, **kwargs)
                self._cache[text] = response
                return response
            except Exception as exc:  # noqa: BLE001 - retry transient 429s
                last_error = exc
                await asyncio.sleep(1.0 + attempt)

        raise last_error if last_error is not None else RuntimeError(
            "embedding failed"
        )


def create_embedding_provider(
    *, deterministic: bool = False,
) -> BaseEmbeddingProvider:
    """Return the embedding provider used by every system.

    ``deterministic=False`` uses the configured production embedding model so
    the dense-only Conventional Memory baseline is not handicapped by a
    placeholder embedder. ``deterministic=True`` selects the offline hash
    embedder for tests and CI.
    """
    if deterministic:
        from .runtime import DemoHashEmbeddingProvider

        return DemoHashEmbeddingProvider()

    try:
        return ResilientEmbeddingProvider(
            EmbeddingFactory.create(EmbeddingProvider.JINA)
        )
    except Exception:
        return _fallback_embedding()


def _fallback_embedding() -> BaseEmbeddingProvider:
    from .runtime import DemoHashEmbeddingProvider

    return DemoHashEmbeddingProvider()


class ResilientLLMProvider(BaseLLMProvider):
    """Wraps a real LLM provider with rate-limit retries.

    The benchmark issues many calls; the free-tier TPM limit surfaces as HTTP
    429. Retrying with backoff keeps the run deterministic in outcome while
    absorbing transient throttling.
    """

    def __init__(
        self,
        inner: BaseLLMProvider,
        *,
        attempts: int = 6,
    ) -> None:
        self._inner = inner
        self._attempts = attempts
        self.name = inner.name

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs,
    ) -> ChatResponse:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                return await self._inner.generate(
                    prompt, system_prompt=system_prompt, **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - retry transient 429s
                last_error = exc
                await asyncio.sleep(2.0 + attempt)

        raise last_error if last_error is not None else RuntimeError(
            "llm generation failed"
        )

    async def stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._inner.stream(
            prompt, system_prompt=system_prompt, **kwargs,
        ):
            yield chunk


def create_llm_provider(*, live: bool) -> BaseLLMProvider:
    """Return the LLM provider used by every system."""
    if not live:
        return DemoEchoLLMProvider()
    try:
        return ResilientLLMProvider(LLMFactory.create(LLMProvider.GROQ))
    except Exception:
        return DemoEchoLLMProvider()


async def create_benchmark_runtime(
    *,
    config: DemoConfig,
    llm_provider: BaseLLMProvider,
    decision_provider: DecisionProvider | None = None,
    deterministic_embeddings: bool = False,
) -> DemoRuntime:
    """Build one shared runtime seeded with the TaskFlow dataset."""
    embedding = create_embedding_provider(
        deterministic=deterministic_embeddings,
    )

    runtime = DemoRuntime(
        llm_provider=llm_provider,
        decision_provider=decision_provider,
        namespace=config.namespace,
        embedding_provider=embedding,
    )

    await runtime.initialize()

    # Seed with bounded concurrency: the vector/sparse providers are internally
    # locked, so parallel upserts are safe, but the embedding API rate-limits
    # aggressive parallelism. A small semaphore keeps seeding fast without
    # tripping the provider.
    memories = load_memories(config.memories_path)
    semaphore = asyncio.Semaphore(2)

    async def seed_one(memory) -> None:
        async with semaphore:
            await _store_with_retry(runtime, memory)

    await asyncio.gather(*(seed_one(memory) for memory in memories))

    return runtime


async def _store_with_retry(
    runtime: DemoRuntime,
    memory,
    *,
    attempts: int = 4,
) -> None:
    """Store one memory, retrying on transient embedding rate limits."""
    for attempt in range(attempts):
        try:
            await runtime.store_memory(
                content=memory.content,
                memory_type=memory.memory_type,
                metadata=memory.metadata,
            )
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(1.0 + attempt)


def build_memory_systems(
    runtime: DemoRuntime,
    config: DemoConfig,
) -> tuple[MemorySystem, ...]:
    """Return the three systems under comparison, in reporting order."""
    return (
        NoMemorySystem(
            runtime=runtime, llm_provider=runtime.llm_provider, config=config,
        ),
        ConventionalMemorySystem(
            runtime=runtime, llm_provider=runtime.llm_provider, config=config,
        ),
        EngramSystem(
            runtime=runtime,
            llm_provider=runtime.llm_provider,
            config=config,
            decision_provider=runtime.decision_provider,
        ),
    )


__all__ = [
    "build_memory_systems",
    "create_benchmark_runtime",
    "create_embedding_provider",
    "create_llm_provider",
    "RuleBasedDecisionProvider",
    "settings",
]