class ContextError(Exception):
    """Base exception for context-management errors."""

class InvalidContextStateTransition(ContextError):
    """Raised when an artifact lifecycle transition is invalid."""

class ContextArtifactNotFound(ContextError):
    """Raised when a requested context artifact does not exist."""

class ContextArtifactDeleted(ContextError):
    """Raised when an operation requires a deleted artifact."""