from __future__ import annotations
from abc import abstractmethod

from ..base import RetrievalStrategy
from ..models import RetrievalQuery, RetrievalResult

class StructuralRetrievalStrategy(RetrievalStrategy):
    """
    Retrieval based on document structure rather than embedding similarity.

    PageIndex can become an implementation behind this interface later.
    """
    name = "structural"
    @abstractmethod
    async def retrieve(
        self, query: RetrievalQuery
    ) -> RetrievalResult:
        raise NotImplementedError