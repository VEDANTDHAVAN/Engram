from __future__ import annotations
from abc import ABC, abstractmethod
from .types import GraphEntity, GraphRelationship

class GraphProvider(ABC):
    """
    Abstract knowledge-graph interface.

    This interface intentionally contains no HelixDB specific concepts or types.
    """
    @abstractmethod
    async def create_entity(self, entity: GraphEntity) -> GraphEntity:
        raise NotImplementedError

    @abstractmethod
    async def create_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        raise NotImplementedError

    @abstractmethod
    async def expand(
        self, entity_id: str, *, depth: int = 1, limit: int = 50,
        relationship_types: list[str] | None = None,
    ) -> list[GraphEntity]:
        raise NotImplementedError