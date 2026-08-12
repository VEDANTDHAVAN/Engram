from __future__ import annotations
from abc import ABC, abstractmethod
from .models import RetrievalQuery, RetrievalResult

class RetrievalStrategy(ABC):
    """Port for all CAML retrieval strategies."""
    name: str

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError