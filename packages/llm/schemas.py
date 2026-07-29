from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str