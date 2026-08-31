from __future__ import annotations

import math
import re
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .base import SparseProvider
from .exceptions import SparseInitializationError, SparseRecordError, SparseSearchError
from .types import SparseRecord

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")

@dataclass(slots=True)
class _SparseDocument:
    namespace: str
    record_id: str
    content: str
    metadata: dict[str, Any]
    tokens: list[str]

class BM25Provider(SparseProvider):
    """
    In-process BM25 sparse retrieval provider.

    PostgreSQL remains the canonical source of memory.

    This provider owns only the searchable sparse index:
        namespace
        record ID
        searchable content
        metadata

    It is intentionally provider-neutral from the application's
    perspective and can later be replaced by a persistent search
    backend without changing CAML retrieval code.

    BM25 parameters:
        k1 controls term-frequency saturation.
        b controls document-length normalization.
    """
    def __init__(
        self, *, 
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("BM25 k1 must be >= 0.")

        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1.")

        self.k1 = k1
        self.b = b

        self._documents: dict[
            tuple[str, str], _SparseDocument,
        ] = {}
        self._initialized = False

        # BM25 index operations are synchronous and CPU-bound.
        # The lock protects the in-memory index from concurrent mutation while async application code accesses it.
        self._lock = RLock()

    async def initialize(self) -> None:
        """
        Initialize the sparse index.

        The current backend is in-memory, initialization only establishes the provider lifecycle state.
        """
        try:
            with self._lock:
                self._initialized = True
        except Exception as exc:
            raise SparseInitializationError(
                "Failed to initialize BM25 provider."
            ) from exc

    async def upsert(self, *, namespace: str, record_id: str, content: str, metadata: dict | None = None) -> None:
        """
        Insert or replace a searchable document.

        The `(namespace, record_id)` pair is the stable index key.
        """
        self._require_initialized()

        if not namespace.strip():
            raise ValueError("Sparse record namespace cannot be empty.")

        if not record_id.strip():
            raise ValueError("Sparse record ID cannot be empty.")

        if not isinstance(content, str):
            raise TypeError("Sparse record content must be a string.")

        try:
            tokens = self._tokenize(content)
            document = _SparseDocument(
                namespace=namespace, record_id=record_id,
                content=content, tokens=tokens,
                metadata=dict(metadata or {}),
            )

            with self._lock:
                self._documents[(namespace, record_id)] = document

        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise SparseRecordError(
                f"Failed to upsert sparse record {record_id!r}."
            ) from exc

    async def search(self, *, namespace: str, query: str, top_k: int = 10, filters: dict | None = None) -> list[SparseRecord]:
        """
        Search the namespace using BM25 scoring.

        Supported filter: allowed_ids

        `allowed_ids` is applied before scoring, which prevents 
        unauthorized/unwanted records from becoming condidates.
        """
        self._require_initialized()

        if not namespace.strip():
            raise ValueError("Sparse search namespace cannot be empty.")

        if top_k < 1:
            raise ValueError("Sparse search top_k must be >= 1.")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        try:
            with self._lock:
                documents = [
                    document for document in self._documents.values()
                    if document.namespace == namespace
                ]

            allowed_ids = self._build_allowlist(filters)

            if allowed_ids is not None:
                documents = [
                    document for document in documents
                    if document.record_id in allowed_ids
                ]

            if not documents:
                return []

            scores = self._score_documents(
                query_tokens=query_tokens, documents=documents,
            )

            ranked = sorted(
                scores, key=lambda item: (
                    -item[1], item[0].record_id,
                ),
            )

            results: list[SparseRecord] = []

            for document, score in ranked[:top_k]:
                # Zero-score documents are not useful lexical evidence.
                if score <= 0.0:
                    continue

                results.append(
                    SparseRecord(
                        id=document.record_id, score=score, content=document.content,
                        metadata=dict(document.metadata),
                    )
                )

            return results

        except ValueError:
            raise
        except Exception as exc:
            raise SparseSearchError(
                f"BM25 search failed for namespace {namespace!r}."
            ) from exc

    async def delete(
        self, *, namespace: str, record_id: str,
    ) -> None:
        """
        Delete a sparse index entry.

        Deleting a non-existent record is intentionally idempotent.
        """
        self._require_initialized()

        if not namespace.strip():
            raise ValueError("Sparse record namespace cannot be empty.")

        if not record_id.strip():
            raise ValueError("Sparse record ID cannot be empty.")

        try:
            with self._lock:
                self._documents.pop(
                    (namespace, record_id), None,
                )
        except Exception as exc:
            raise SparseRecordError(
                f"Failed to delete sparse record {record_id!r}."
            ) from exc

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("BM25Provider has not been initialized.")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize searchable text.

        Lower-casing gives deterministic lexical matching while preserving the original content in SparseRecord.
        """
        return [
            token.lower() for token in _TOKEN_PATTERN.findall(text)
        ]

    @staticmethod
    def _build_allowlist(
        filters: dict | None,
    ) -> set[str] | None:
        if not filters:
            return None

        allowed_ids = filters.get("allowed_ids")

        if allowed_ids is None:
            return None

        if not isinstance(
            allowed_ids, (list, tuple, set, frozenset),
        ):
            raise ValueError(
                "filters['allowed_ids'] must be a collection."
            )

        return {
            str(record_id) for record_id in allowed_ids
        }

    def _score_documents(
        self, *, query_tokens: list[str],
        documents: list[_SparseDocument],
    ) -> list[tuple[_SparseDocument, float]]:
        """
        Calculate Okapi BM25 scores.

        Formula:
            IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        """
        if not documents:
            return []

        document_count = len(documents)

        total_length = sum(len(document.tokens) for document in documents)

        average_document_length = (
            total_length/document_count if document_count else 0.0
        )

        if average_document_length <= 0:
            return []

        # Document frequency for every query-relevant token.
        document_frequency: dict[str, int] = {}

        for token in set(query_tokens):
            document_frequency[token] = sum(1 for document in documents if token in set(document.tokens))

        scores: list[
            tuple[_SparseDocument, float]
        ] = []

        for document in documents:
            term_frequencies = self._term_frequencies(document.tokens,)
            document_length = len(document.tokens)
            score = 0.0

            for token in query_tokens:
                tf = term_frequencies.get(token, 0)

                if tf == 0:
                    continue

                df = document_frequency.get(token, 0)

                if df == 0:
                    continue

                # Standard BM25 IDF with a +1 offset to keep the score non-negative.
                idf = math.log(
                    1.0 + (document_count - df + 0.5) / (df + 0.5)
                )

                denominator = (
                    tf + self.k1 * (1.0 - self.b + self.b * (
                        document_length/average_document_length
                    ))
                )

                score += idf * (
                    (tf * (self.k1 + 1.0)) / denominator
                )

            scores.append((document, score))

        return scores

    @staticmethod
    def _term_frequencies(
        tokens: list[str],
    ) -> dict[str, int]:
        frequencies: dict[str, int] = {}

        for token in tokens:
            frequencies[token] = (
                frequencies.get(token, 0) + 1
            )

        return frequencies