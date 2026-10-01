from __future__ import annotations

import json

from packages.context.state import ContextStateStore
from packages.providers.cache import CacheProvider

class RedisContextStateStore(ContextStateStore):
    """
    Redis-backed ContextStateStore.

    Stores only ordered artifact IDs. Complete artifact content remains in ContextArtifactStore.
    """
    def __init__(
        self, cache: CacheProvider,
        *, key_prefix: str = "engram:context",
    ) -> None:
        self._cache = cache
        self._key_prefix = key_prefix.rstrip(":")

    def _key(self, namespace: str) -> str:
        return f"{self._key_prefix}:{namespace}:active"

    async def get_ids(
        self, *, namespace: str,
    ) -> list[str]:
        value = await self._cache.get(
            self._key(namespace)
        )

        if value is None:
            return []

        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value

        if not isinstance(parsed, list):
            raise ValueError("Invalid context state stored in Redis.")

        return [
            artifact_id for artifact_id in parsed
            if isinstance(artifact_id, str)
        ]

    async def add(
        self, *, namespace: str, artifact_id: str,
    ) -> None:
        ids = await self.get_ids(
            namespace=namespace,
        )

        if artifact_id in ids:
            return

        ids.append(artifact_id)

        await self._cache.set(
            self._key(namespace),
            json.dumps(ids),
        )

    async def remove(
        self, *, namespace: str, artifact_id: str,
    ) -> None:
        ids = await self.get_ids(
            namespace=namespace,
        )
        if artifact_id not in ids:
            return

        ids.remove(artifact_id)

        if not ids:
            await self._cache.delete(
                self._key(namespace)
            )
            return

        await self._cache.set(
            self._key(namespace),
            json.dumps(ids),
        )

    async def exists(
        self, *, namespace: str, artifact_id: str,
    ) -> bool:
        ids = await self.get_ids(namespace=namespace)

        return artifact_id in ids

    async def clear(self, *, namespace: str) -> None:
        await self._cache.delete(self._key(namespace))