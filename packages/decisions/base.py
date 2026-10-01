from __future__ import annotations
from abc import ABC, abstractmethod

from .models import DecisionRequest, DecisionResponse

class DecisionProvider(ABC):
    """
    Provider-neutral interface for structured Engram decisions.

    Decision providers decide what Engram should do.
    They do not execute retrieval, memory access, or tool calls.    
    """
    name: str

    @abstractmethod
    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        raise NotImplementedError