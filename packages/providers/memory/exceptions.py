class MemoryProviderError(Exception):
    """Base memory provider error."""


class MemoryNotFoundError(MemoryProviderError):
    """Requested memory does not exist."""