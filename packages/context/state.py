from __future__ import annotations

from abc import ABC, abstractmethod

class ContextStateStore(ABC):
    """
    Provider-agnostic storage for the current context membership.

    This store contains artifact IDs, not complete artifact content.
    """
    @abstractmethod
    async def get_ids(
        self, *, namespace: str,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def add(
        self, *, namespace: str, artifact_id: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def remove(
        self, *, namespace: str, artifact_id: str, 
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(
        self, *, namespace: str, artifact_id: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear(
        self, *, namespace: str,
    ) -> None:
        raise NotImplementedError