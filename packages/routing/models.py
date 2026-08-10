from pydantic import BaseModel, Field

from packages.state_engine.types import MemoryType

class RoutingDecision(BaseModel):
    """Result of Memory Routing."""
    memory_types: list[MemoryType] = Field(default_factory=list)
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
    )
    reasons: list[str] = Field(default_factory=list)