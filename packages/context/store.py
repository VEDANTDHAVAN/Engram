from __future__ import annotations

from abc import ABC, abstractmethod

from packages.context.models import ContextArtifact

class ContextArtifactStore(ABC):
    """
    Provider-neutral persistence contract for context atrifacts.

    The store retains the complete artifact and manages its lifecycle. 
    It does not decide whether an artifact should remain in the active LLM context.
    """
    @abstractmethod
    async def store(
        self, *, ttl: int | None = None,
        artifact: ContextArtifact,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(
        self, *, namespace: str,
        artifact_id: str,
    ) -> ContextArtifact | None:
        raise NotImplementedError

    @abstractmethod
    async def hide(
        self, *, namespace: str, 
        artifact_id: str,
    ) -> ContextArtifact:
        raise NotImplementedError

    @abstractmethod
    async def restore(
        self, *, namespace: str,
        artifact_id: str,
    ) -> ContextArtifact:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self, *, namespace: str,
        artifact_id: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(
        self, *, namespace: str,
        artifact_id: str,
    ) -> bool:
        raise NotImplementedError