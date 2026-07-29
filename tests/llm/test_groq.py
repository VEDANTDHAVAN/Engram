import pytest

from packages.llm import LLMFactory
from packages.llm.types import LLMProvider

@pytest.mark.asyncio
async def test_groq_generation():
    llm = LLMFactory.create(LLMProvider.GROQ)

    response = await llm.generate(
        prompt="Reply with exactly: pong"
    )

    assert "pong" in response.content.lower()