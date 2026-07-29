from pydantic import BaseModel

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    provider: str