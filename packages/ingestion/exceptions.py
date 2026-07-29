class IngestionError(Exception):
    """Base ingestion exception."""

class NormalizationError(IngestionError):
    """Document normalization failed."""

class ChunkingError(IngestionError):
    """Document chunking failed."""