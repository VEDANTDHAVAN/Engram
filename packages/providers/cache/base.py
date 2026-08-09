from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class CacheProvider(ABC):
    """
    Fast ephemeral storage abstraction.

    Primarily intended for working memory and session state.
    """
    @abstractmethod
    async def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def set(
        self, key: str, value: Any, *, ttl: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self, key: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(
        self, key: str, 
    ) -> bool:
        raise NotImplementedError