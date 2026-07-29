import asyncio

from packages.llm.factory import LLMFactory
from packages.llm.types import LLMProvider

async def main():
    llm = LLMFactory.create(
        LLMProvider.GROQ,
    )

    response = await llm.generate(
        prompt="Say hello from Engram."
    )

    print(response.content)


asyncio.run(main())