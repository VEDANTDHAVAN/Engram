class EmbeddingError(Exception):
    """Base embedding exception."""

class ProviderNotConfiguredError(EmbeddingError):
    """Raised when a provider is unavailable."""

class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""