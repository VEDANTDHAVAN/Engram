from functools import lru_cache

import httpx

from packages.core.config import settings

@lru_cache(maxsize=1)
def get_jina_client() -> httpx.AsyncClient:
    if not settings.JINA_API_KEY:
        raise RuntimeError(
            "JINA_API_KEY is not configured."
        )

    return httpx.AsyncClient(
        base_url="https://api.jina.ai/v1",
        headers={
            "Authorization": f"Bearer {settings.JINA_API_KEY}"
        }, timeout=30,
    )