import pytest

from packages.llm.factory import LLMFactory
from packages.llm.types import LLMProvider

@pytest.mark.asyncio
async def test_stream():
    llm = LLMFactory.create(LLMProvider.GROQ)

    chunks = []

    async for token in llm.stream(
        "Reply only with Hello."
    ):
        chunks.append(token)

    text = "".join(chunks)

    assert "hello" in text.lower()