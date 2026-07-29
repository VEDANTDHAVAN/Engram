class LLMError(Exception):
    """Base LLM exception."""

class ProviderNotFoundError(LLMError):
    """Requested provider does not exist."""

class LLMGenerationError(LLMError):
    """Generation failed."""