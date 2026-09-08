from __future__ import annotations
from collections import defaultdict

from ..models import RetrievalEvidence, RetrievalResult
from .base import CandidateFusion
from .models import FusedCandidate

class ReciprocalRankFusion(CandidateFusion):
    """
    Provider-independent Reciprocal Rank Fusion.

    RRF combines ranked results without requiring the underlying retrieval strategies to expose comparable score scales.

    RRF score:

        score(d) = sum(1 / (k + rank(d)))

    where rank starts at 1.
    """
    def __init__(self, *, k: int = 60) -> None:
        if k < 1:
            raise ValueError("RRF k must be >= 1.")

        self.k = k

    def fuse(
        self, results: list[RetrievalResult],
    ) -> list[FusedCandidate]:
        candidates: dict[str, dict] = defaultdict(
            lambda: {
                "score": 0.0, "content": "",
                "sources": set(), "metadata": {},
            }
        )

        for result in results:
            seen_ids: set[str] = set()

            for rank, evidence in enumerate(
                result.evidence, start=1,
            ):
                if evidence.id in seen_ids:
                    continue

                seen_ids.add(evidence.id)
                
                candidate = candidates[evidence.id]

                candidate["score"] += (
                    1.0 / (self.k + rank)
                )

                candidate["sources"].add(
                    evidence.source
                )

                if not candidate["content"]:
                    candidate["content"] = evidence.content

                for key, value in evidence.metadata.items():
                    candidate["metadata"].setdefault(key, value,)

        fused: list[FusedCandidate] = []

        for candidate_id, candidate in candidates.items():
            fused.append(
                FusedCandidate(
                    id=candidate_id, content=candidate["content"],
                    score=candidate["score"], sources=sorted(candidate["sources"]),
                    metadata=candidate["metadata"],
                )
            )

        fused.sort(
            key=lambda candidate: (
                -candidate.score, candidate.id,
            )
        )

        return fused