class MemoryServiceError(Exception):
    """Base exception for memory services."""

class MemoryNotFoundError(MemoryServiceError):
    """Raised when a memory does not exist."""