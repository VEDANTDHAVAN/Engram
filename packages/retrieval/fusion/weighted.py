from __future__ import annotations
from collections import defaultdict

from ..models import RetrievalResult, RetrievalEvidence
from .base import CandidateFusion
from .models import FusedCandidate

class WeightedCandidateFusion(CandidateFusion):
    """
    Combine candidates from multiple retrieval strategies using strategy-specific weights.

    Scores are normalized independently per strategy before applying the configured weight.

    This prevents a raw BM25 score from being compared directly against a vector similarity score.
    """
    def __init__(
        self, *, 
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = dict(weights or {})

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
            if not result.evidence:
                continue

            weight = self.weights.get(result.strategy, 1.0)

            if weight <= 0.0:
                continue

            normalized = self._normalize(result.evidence)

            for evidence, normalized_score in normalized:
                candidate = candidates[evidence.id]

                candidate["score"] += (normalized_score*weight)
                candidate["sources"].add(evidence.source)

                if not candidate["content"]:
                    candidate["content"] = (evidence.content)

                for key, value in evidence.metadata.items():
                    candidate["metadata"].setdefault(key, value)

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
            key=lambda candidate: (-candidate.score, candidate.id)
        )

        return fused

    @staticmethod
    def _normalize(
        evidence: list[RetrievalEvidence],
    ) -> list[tuple[RetrievalEvidence, float]]:
        """
        Min-max normalize scores within one retrieval result.

        A single result receives 1.0 because it is the strongest
        candidate available from that strategy.
        """
        if not evidence:
            return []

        scores = [
            item.score for item in evidence
        ]

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [
                (item, 1.0) for item in evidence
            ]

        return [
            (
                item, (item.score - minimum) / (maximum - minimum),
            ) for item in evidence
        ]