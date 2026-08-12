from __future__ import annotations

from packages.core.config import settings

from .base import VectorProvider
from .turbovec import TurboVecProvider

class VectorFactory:
    """
    Creates the configured CAML vector provider.

    Application code depends on VectorProvider rather than knowing which vector backend is being used.
    """
    @staticmethod
    def create(
        provider: str | None = None,
    ) -> VectorProvider:
        provider_name = (
            provider or settings.VECTOR_PROVIDER
        ).strip().lower()

        if provider_name == "turbovec":
            return TurboVecProvider(
                dimension=settings.EMBEDDING_DIMENSIONS,
                index_path=settings.TURBOVEC_INDEX_PATH,
                bit_width=settings.TURBOVEC_BIT_WIDTH,
            )

        raise ValueError(
            f"Unsupported vector provider: "
            f"{provider_name!r}"
        )