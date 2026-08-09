from __future__ import annotations

from .chunker import DocumentChunker
from .normalizer import DocumentNormalizer
from .schemas import IngestionResult, RawDocument

class IngestionPipeline:
    """Deterministic preprocessing pipeline.
       
       No LLM.
       No embeddings.
       No database.
    """
    def __init__(
        self, normalizer: DocumentNormalizer | None = None,
        chunker: DocumentChunker | None = None,
    ):
        self.normalizer = normalizer or DocumentNormalizer()
        self.chunker = chunker or DocumentChunker()

    def run(
        self, document: RawDocument,
    ) -> IngestionResult:
        normalized = self.normalizer.normalize(document)
        chunks = self.chunker.chunk(normalized)

        return IngestionResult(
            document=normalized, chunks=chunks,
            total_chunks=len(chunks),
        )