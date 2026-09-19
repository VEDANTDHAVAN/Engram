from __future__ import annotations

from packages.context.state import ContextStateStore

class InMemoryContextStateStore(ContextStateStore):
    """
    In-memory ContextStateStore implementation.

    Intended for deterministic tests and local development.
    """
    def __init__(self) -> None:
        self._contexts: dict[str, list[str]] = {}

    async def get_ids(self, *, namespace: str) -> list[str]:
        return list(
            self._contexts.get(namespace, [])
        )

    async def add(self, *, namespace: str, artifact_id: str) -> None:
        entries = self._contexts.setdefault(namespace,[])

        if artifact_id not in entries:
            entries.append(artifact_id)

    async def remove(self, *, namespace: str, artifact_id: str) -> None:
        entries = self._contexts.get(namespace)

        if entries is None:
            return

        self._contexts[namespace] = [
            current_id for current_id in entries
            if current_id != artifact_id
        ]

        if not self._contexts[namespace]:
            del self._contexts[namespace]

    async def exists(self, *, namespace: str, artifact_id: str) -> bool:
        return artifact_id in self._contexts.get(namespace, [])

    async def clear(self, *, namespace: str) -> None:
        self._contexts.pop(namespace, None)