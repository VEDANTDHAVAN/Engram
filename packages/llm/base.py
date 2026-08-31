from abc import ABC, abstractmethod
from typing import AsyncGenerator

from packages.llm.schemas import ChatResponse

class BaseLLMProvider(ABC):
    """Base interface implemented by every LLM provider."""
    name: str

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: str | None = None,
        **kwargs,) -> ChatResponse:
        """Generate a text response."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self, prompt: str, 
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a response."""
        raise NotImplementedError