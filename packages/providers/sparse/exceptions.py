class SparseProviderError(Exception):
    """Base exception for sparse retrieval providers."""


class SparseInitializationError(SparseProviderError):
    """Sparse provider initialization failed."""


class SparseSearchError(SparseProviderError):
    """Sparse retrieval failed."""


class SparseRecordError(SparseProviderError):
    """A sparse record operation failed."""