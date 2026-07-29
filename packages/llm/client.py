from functools import lru_cache

from openai import AsyncOpenAI

from packages.core.config import settings

@lru_cache(maxsize=1)
def get_groq_client() -> AsyncOpenAI:
    """Singleton Groq client."""
    return AsyncOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )