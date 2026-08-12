class RetrievalError(Exception):
    """Base retrieval exception."""

class RetrievalStrategyError(RetrievalError):
    """A retrieval strategy failed."""

class RetrievalConfigurationError(RetrievalError):
    """A retrieval strategy is incorrectly configured."""