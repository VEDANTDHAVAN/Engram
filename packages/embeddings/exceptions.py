class EmbeddingError(Exception):
    """Base embedding exception."""

class EmbeddingProviderError(EmbeddingError):
    """Embedding provider failure."""

class ProviderNotConfiguredError(EmbeddingError):
    """Raised when a provider is unavailable."""

class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""