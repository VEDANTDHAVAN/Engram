from __future__ import annotations
from abc import ABC, abstractmethod

from ..models import RetrievalResult
from .models import FusedCandidate

class CandidateFusion(ABC):
    """
    Provider-independent candidate fusion interface.

    Fusion operated on retrieval results, not concrete storage or search providers.
    """
    @abstractmethod
    def fuse(
        self, results: list[RetrievalResult],
    ) -> list[FusedCandidate]:
        raise NotImplementedError