class VectorProviderError(Exception):
    """Base vector provider error."""


class VectorSearchError(VectorProviderError):
    """Vector search failed."""


class VectorUpsertError(VectorProviderError):
    """Vector upsert failed."""