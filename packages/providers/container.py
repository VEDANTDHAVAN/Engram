from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.config import settings
from packages.database.session import AsyncSessionLocal
from packages.embeddings import EmbeddingFactory, EmbeddingProvider
from packages.embeddings.base import BaseEmbeddingProvider
from packages.providers.memory import (
    MemoryProvider,
    PostgresMemoryProvider,
)
from packages.providers.vector import (
    VectorFactory,
    VectorProvider,
)


class ProviderContainer:
    """
    Composition root for CAML infrastructure providers.

    The container resolves concrete implementations while the
    application services depend only on provider interfaces.

    Database sessions remain externally scoped and are never kept
    as a global/shared session by the container.
    """

    def __init__(
        self,
        *,
        vector_provider: VectorProvider | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        memory_provider: MemoryProvider | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        self._vector_provider = vector_provider
        self._embedding_provider = embedding_provider
        self._memory_provider = memory_provider

        self._session = session
        self._initialized = False

    @property
    def vector(self) -> VectorProvider:
        """
        Resolve the configured vector provider.
        """

        if self._vector_provider is None:
            self._vector_provider = VectorFactory.create()

        return self._vector_provider

    @property
    def embedding(self) -> BaseEmbeddingProvider:
        """
        Resolve the configured embedding provider.
        """
        if self._embedding_provider is None:
            provider = EmbeddingProvider(settings.EMBEDDING_PROVIDER)

            self._embedding_provider = (EmbeddingFactory.create(provider))

        return self._embedding_provider

    @property
    def memory(self) -> MemoryProvider:
        """
        Resolve the PostgreSQL memory provider.

        A database session must be explicitly supplied when constructing the container.
        This prevents the container from creating a long-lived AsyncSession.
        """
        if self._memory_provider is not None:
            return self._memory_provider

        if self._session is None:
            raise RuntimeError(
                "An AsyncSession is required to create PostgresMemoryProvider."
            )

        self._memory_provider = (
            PostgresMemoryProvider(self._session)
        )

        return self._memory_provider

    async def initialize(self) -> None:
        """
        Initialize infrastructure providers owned by CAML.
        """

        if self._initialized:
            return

        vector_provider = self.vector

        # TurboVecProvider currently exposes initialize().
        # Keep this lifecycle boundary here rather than leaking
        # initialization logic into business services.
        initialize = getattr(
            vector_provider,
            "initialize",
            None,
        )

        if initialize is None:
            raise RuntimeError(
                "Configured vector provider does not support "
                "initialization."
            )

        await initialize()

        self._initialized = True

    async def shutdown(self) -> None:
        """
        Shut down initialized providers.
        """
        if not self._initialized:
            return

        vector_provider = self._vector_provider

        if vector_provider is not None:
            shutdown = getattr(
                vector_provider,
                "shutdown",
                None,
            )

            if shutdown is not None:
                await shutdown()

        self._initialized = False