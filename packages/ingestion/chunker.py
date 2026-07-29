from __future__ import annotations

import tiktoken

from packages.core.config import settings

from .exceptions import ChunkingError
from .schemas import Chunk, NormalizedDocument

class DocumentChunker:
    """Token-aware chunker with overlap."""
    def __init__(self, chunk_size: int | None = None,
        overlap: int | None = None, encoding: str | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.encoder = tiktoken.get_encoding(
            encoding or settings.TOKENIZER_ENCODING
        )

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        if not document.content:
            raise ChunkingError("Document is empty.")

        tokens = self.encoder.encode(document.content)

        chunks: list[Chunk] = []

        start = 0
        index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            token_slice = tokens[start:end]

            chunks.append(
                Chunk(
                    index=index, text=self.encoder.decode(token_slice),
                    token_count=len(token_slice), token_start=start, token_end=end,
                )
            )

            if end == len(tokens):
                break

            start += self.chunk_size - self.overlap
            index += 1

        return chunks