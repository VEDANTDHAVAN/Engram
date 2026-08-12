from __future__ import annotations
from abc import abstractmethod

from ..base import RetrievalStrategy
from ..models import RetrievalQuery, RetrievalResult

class TemporalRetrievalStrategy(RetrievalStrategy):
    """
    Base strategy for time-aware memory retrieval.

    Concrete implementation will eventually use the MemoryProvider / PostgreSQL temporal state.
    """
    name = "temporal"

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError