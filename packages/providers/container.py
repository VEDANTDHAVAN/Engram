from __future__ import annotations

from packages.providers.vector import (
    VectorFactory,
    VectorProvider,
)


class ProviderContainer:
    """
    Central dependency container for CAML infrastructure providers.

    The container owns provider instances and coordinates their
    application lifecycle.

    Factories construct providers.
    The container owns those instances.
    Business logic depends on provider interfaces.
    """

    def __init__(
        self,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> None:
        self._vector_provider = vector_provider
        self._initialized = False

    @property
    def vector(self) -> VectorProvider:
        """
        Return the configured vector provider.

        The provider is lazily constructed when one was not
        explicitly injected.
        """

        if self._vector_provider is None:
            self._vector_provider = VectorFactory.create()

        return self._vector_provider

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