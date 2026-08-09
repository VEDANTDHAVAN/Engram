from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class GraphEntity(BaseModel):
    id: str
    entity_type: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)