from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

class CreateMemoryRequest(BaseModel):
    namespace: str
    content: str
    embedding: list[float]
    confidence: float = 1.0
    importance: float = 0.5

class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    namespace: str
    content: str
    confidence: float
    importance: float
    version: int
    state: str
    created_at: datetime