from typing import AsyncGenerator

from packages.core.config import settings
from packages.llm.base import BaseLLMProvider
from packages.llm.schemas import ChatResponse
from packages.llm.client import get_groq_client

class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(self) -> None:
        self.client = get_groq_client()

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ChatResponse:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL, messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        )

        return ChatResponse(
            content=response.choices[0].message.content or "",
            model=response.model, provider=self.name
        )

    async def stream(self, prompt: str, system_prompt: str | None = None, **kwargs) -> AsyncGenerator[str, None]:
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        stream = await self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta