from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Sequence

import numpy as np
from turbovec import IdMapIndex

from .base import VectorProvider
from .exceptions import VectorProviderError
from .types import VectorRecord

class TurboVecProvider(VectorProvider):
    """
    CAML VectorProvider implementation backed by TurboVec.

    PostgreSQL remains the canonical memory store.
    TurboVec stores only the dense vector retrieval index.

    TurboVec's IdMapIndex uses uint64 IDs, while CAML uses provider-neutral string IDs.
    A deterministic mapping is therefore maintained between the two.
    """
    def __init__(
        self, *, dimension: int,
        index_path: str | Path,
        bit_width: int = 4,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        if bit_width not in (2, 4):
            raise ValueError("TurboVec bit_width must be either 2 or 4.")

        self.dimension = dimension
        self.index_path = Path(index_path)
        self.bit_width = bit_width

        self._index: IdMapIndex | None = None

        # CAML external ID -> TurboVec uint64 ID
        self._id_to_uint64: dict[str, int] = {}
        # Turbovec uint64 ID -> CAML external ID
        self._uint64_to_id: dict[int, str] = {}

        # TurboVec is a synchronous Python/Rust API.
        # Protect the index because CAML's surrounding architecture is asynchronous.
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create or load the TurboVec index."""
        async with self._lock:
            if self._index is not None:
                return

            self.index_path.parent.mkdir(
                parents=True, exist_ok=True,
            )

            if self.index_path.exists():
                self._index = await asyncio.to_thread(
                    IdMapIndex.load, str(self.index_path),
                )
            else:
                self._index = IdMapIndex(
                    dim=self.dimension, bit_width=self.bit_width,
                )

    def _require_index(self) -> IdMapIndex:
        """
        Return the initialized index.

        Initialization is explicit so application startup can control when disk I/O occurs.
        """
        if self._index is None:
            raise VectorProviderError(
                "TurboVecProvider has not been initalized."
                "Call initialize() before using the provider."
            )

        return self._index

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{self.dimension}, got {len(embedding)}"
            )

        if not all(
            np.isfinite(value) for value in embedding
        ):
            raise ValueError("Embedding contains NaN or infinite values.")

    @staticmethod
    def _external_id_to_uint64(record_id: str) -> int:
        """
        Deterministically map a CAML ID to uint64.

        UUID strings are converted from their hexadecimal representation when possible.
        Other identifiers use SHA-256 and take the first 64 bits.
        """
        import hashlib
        import uuid

        try:
            value = uuid.UUID(record_id).int
            return value & ((1 << 64) - 1)
        except ValueError:
            digest = hashlib.sha256(
                record_id.encode("utf-8")
            ).digest()

            return int.from_bytes(
                digest[:8], signed=False,
                byteorder="big",
            )

    def _register_id(
    self, record_id: str,
    ) -> tuple[int, bool]:
        """
        Register a CAML external ID and return:

        (TurboVec numeric ID, already_registered)
        
        The boolean allows callers to distinguish between a new
        record and an existing record before performing an upsert.
        """

        existing_numeric_id = self._id_to_uint64.get(record_id)

        if existing_numeric_id is not None:
            return existing_numeric_id, True

        numeric_id = self._external_id_to_uint64(record_id)

        existing_external_id = self._uint64_to_id.get(numeric_id)

        if (
            existing_external_id is not None
            and existing_external_id != record_id
        ):
            raise VectorProviderError(
                "TurboVec ID collision detected between "
                f"{record_id!r} and {existing_external_id!r}"
            )

        self._id_to_uint64[record_id] = numeric_id
        self._uint64_to_id[numeric_id] = record_id

        return numeric_id, False

    async def upsert(
        self, *, namespace: str, record_id: str,
        embedding: list[float], content: str, 
        metadata: dict | None = None,
    ) -> None:
        """
        Insert or replace a vector.

        PostgreSQL remains the canonical source for content and metadata.
        TurboVec stores the dense vector index.
        """
        del namespace
        del content
        del metadata

        self._validate_embedding(embedding)
        index = self._require_index()
        numeric_id, already_registered = self._register_id(record_id)

        vector = np.asarray(
            [embedding], dtype=np.float32,
        )
        ids = np.asarray(
            [numeric_id], dtype=np.uint64,
        )

        async with self._lock:
            # TurboVec exposes remove() + add_with_ids()
            # rather than an explicit upsert operation.
            if already_registered:
                await asyncio.to_thread(
                    index.remove, numeric_id,
                )

            await asyncio.to_thread(
                index.add_with_ids, vector, ids,
            )

    async def batch_add(
        self, *, namespace: str,
        records: list[tuple[str, list[float]]],
    ) -> None:
        """
        Add multiple vectors in one TurboVec operation.
        
        Existing IDs are removed first so batch insertion behaves consistently with upsert semantics.
        """
        del namespace

        if not records:
            return

        index = self._require_index()

        vectors: list[list[float]] = []
        numeric_ids: list[int] = []

        existing_numeric_ids: list[int] = []

        for record_id, embedding in records:
            self._validate_embedding(embedding)

            numeric_id, already_registered = self._register_id(record_id)

            vectors.append(embedding)
            numeric_ids.append(numeric_id)

            if already_registered:
                existing_numeric_ids.append(numeric_id)

        vector_array = np.asarray(
            vectors, dtype=np.float32,
        )
        id_array = np.asarray(
            numeric_ids, dtype=np.uint64,
        )

        async with self._lock:
            for numeric_id in existing_numeric_ids:
                await asyncio.to_thread(
                    index.remove, numeric_id,
                )

            await asyncio.to_thread(
                index.add_with_ids, vector_array, id_array,
            )
            
    async def search(
        self, *, namespace: str,
        embedding: list[float], top_k: int = 10,
        filters: dict | None = None,
    ) -> list[VectorRecord]:
        """
        Dense vector search.

        The actual canonical memory lookup remains outside this provider.
        TurboVec returns IDs and similarity scores.
        """
        del namespace

        if top_k <= 0:
            return []

        self._validate_embedding(embedding)
        index = self._require_index()

        query = np.asarray(
            [embedding], dtype=np.float32,
        )
        allowlist = self._build_allow_list(filters)

        async with self._lock:
            if allowlist is not None:
                scores, ids = await asyncio.to_thread(
                    index.search, query, top_k, allowlist=allowlist,
                )
            else:
                scores, ids = await asyncio.to_thread(
                    index.search, query, top_k,
                )

        results: list[VectorRecord] = []

        for score, numeric_id in zip(scores[0], ids[0],):
            numeric_id = int(numeric_id)
            record_id = self._uint64_to_id.get(numeric_id)

            if record_id is None:
                continue

            results.append(
                VectorRecord(
                    id=record_id, score=float(score),
                    content=None, metadata={},
                )
            )

        return results

    def _build_allow_list(
        self, filters: dict | None,
    ) -> np.ndarray | None:
        """
        Convert a CAML filter containing explicit allowed IDs into TurboVec's native allowlist representation.

        Arbitrary metadata predicates are intentionally NOT implemented here because TurboVec's documented filtering primitive is an ID allowlist.
        """
        if not filters:
            return None

        allowed_ids = filters.get("allowed_ids")

        if allowed_ids is None:
            return None

        numeric_ids: list[int] = []

        for record_id in allowed_ids:
            numeric_id = self._id_to_uint64.get(str(record_id))

            if numeric_id is not None:
                numeric_ids.append(numeric_id)

        return np.asarray(
            numeric_ids, dtype=np.uint64,
        )

    async def delete(
        self, *, namespace: str, record_id: str,
    ) -> None:
        """
        Remove a vector from TurboVec.
        """
        del namespace

        index = self._require_index()
        numeric_id = self._id_to_uint64.get(record_id)

        if numeric_id is None:
            return

        async with self._lock:
            await asyncio.to_thread(
                index.remove, numeric_id,
            )

    async def update(
        self, *, namespace: str, 
        record_id: str, embedding: list[float],
    ) -> None:
        """
        Update an existing vector.

        TurboVec does not expose a separate update primitive,
        so CAML semantics are implemented as remove + add.
        """
        await self.upsert(
            namespace=namespace, record_id=record_id,
            embedding=embedding, content="", metadata=None,
        )

    async def persist(self) -> None:
        """Persist the current index to disk."""
        index = self._require_index()

        self.index_path.parent.mkdir(
            parents=True, exist_ok=True,
        )

        async with self._lock:
            await asyncio.to_thread(
                index.write, str(self.index_path),
            )