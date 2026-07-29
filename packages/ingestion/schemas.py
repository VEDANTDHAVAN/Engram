from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

class RawDocument(BaseModel):
    """Input received by the ingestion pipeline."""
    id: UUID | None = None

    source: str
    content: str

    title: str | None = None
    url: str | None = None
    mime_type: str | None = None
    language: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="ignore")

class NormalizedDocument(BaseModel):
    """Normalized representation after preprocessing."""
    source: str
    content: str

    title: str | None = None
    url: str | None = None
    language: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

class Chunk(BaseModel):
    """Single chunk produced from a document."""
    index: int
    text: str
    token_count: int
    token_start: int
    token_end: int

    char_start: int | None = None
    char_end: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

class IngestionResult(BaseModel):
    """Returned by the ingestion pipeline."""
    document: NormalizedDocument
    chunks: list[Chunk]

    total_chunks: int 