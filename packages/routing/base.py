from abc import ABC, abstractmethod

from .models import RoutingDecision

class MemoryRouter(ABC):
    """
    Interface for determining which memory types are relevant to a query.
    """
    @abstractmethod
    async def route(self, query: str) -> RoutingDecision:
        raise NotImplementedError