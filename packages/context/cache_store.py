from __future__ import annotations

from typing import Any

from packages.context.exceptions import (
    ContextArtifactDeleted,
    ContextArtifactNotFound,
    InvalidContextStateTransition,
)
from packages.context.models import (
    ContextArtifact,
    ContextArtifactState,
    can_transition,
)
from packages.context.store import ContextArtifactStore
from packages.providers.cache import CacheProvider


class CacheContextArtifactStore(ContextArtifactStore):
    """
    ContextArtifactStore backed by the existing CacheProvider.

    The cache stores the complete ContextArtifact. Hiding an artifact
    changes only its lifecycle state and never removes its original
    content.
    """

    KEY_PREFIX = "context-artifact"

    def __init__(self, cache: CacheProvider) -> None:
        self.cache = cache

    @classmethod
    def _key(
        cls,
        *,
        namespace: str,
        artifact_id: str,
    ) -> str:
        return f"{cls.KEY_PREFIX}:{namespace}:{artifact_id}"

    async def store(
        self, *,
        artifact: ContextArtifact,
        ttl: int | None = None,
    ) -> None:
        await self._save(
            artifact=artifact,
            ttl=ttl,
        )

    async def _save(
        self,
        artifact: ContextArtifact,
        *,
        ttl: int | None = None,
    ) -> None:
        key = self._key(
            namespace=artifact.namespace,
            artifact_id=artifact.id,
        )

        await self.cache.set(
            key,
            artifact.model_dump(mode="json"),
            ttl=ttl,
        )

    async def get(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> ContextArtifact | None:
        key = self._key(
            namespace=namespace,
            artifact_id=artifact_id,
        )

        value = await self.cache.get(key)

        if value is None:
            return None

        artifact = ContextArtifact.model_validate(value)

        if artifact.state == ContextArtifactState.DELETED:
            return None

        return artifact

    async def hide(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> ContextArtifact:
        return await self._transition(
            artifact_id=artifact_id,
            namespace=namespace,
            target=ContextArtifactState.HIDDEN,
        )

    async def restore(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> ContextArtifact:
        return await self._transition(
            artifact_id=artifact_id,
            namespace=namespace,
            target=ContextArtifactState.ACTIVE,
        )

    async def delete(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> None:
        artifact = await self._get_required(
            artifact_id=artifact_id,
            namespace=namespace,
        )

        if not can_transition(
            artifact.state,
            ContextArtifactState.DELETED,
        ):
            raise InvalidContextStateTransition(
                f"Cannot transition artifact "
                f"{artifact.id!r} from "
                f"{artifact.state.value!r} to "
                f"{ContextArtifactState.DELETED.value!r}."
            )

        key = self._key(
            namespace=namespace,
            artifact_id=artifact_id,
        )

        await self.cache.delete(key)

    async def exists(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> bool:
        key = self._key(
            namespace=namespace,
            artifact_id=artifact_id,
        )

        return await self.cache.exists(key)

    async def _transition(
        self,
        *,
        artifact_id: str,
        namespace: str,
        target: ContextArtifactState,
    ) -> ContextArtifact:
        artifact = await self._get_required(
            artifact_id=artifact_id,
            namespace=namespace,
        )

        if not can_transition(
            artifact.state,
            target,
        ):
            raise InvalidContextStateTransition(
                f"Cannot transition artifact "
                f"{artifact.id!r} from "
                f"{artifact.state.value!r} to "
                f"{target.value!r}."
            )

        updated = artifact.model_copy(
            update={"state": target},
        )

        await self._save(updated)

        return updated

    async def _get_required(
        self,
        *,
        artifact_id: str,
        namespace: str,
    ) -> ContextArtifact:
        key = self._key(
            namespace=namespace,
            artifact_id=artifact_id,
        )

        value: Any | None = await self.cache.get(key)

        if value is None:
            raise ContextArtifactNotFound(
                f"Context artifact {artifact_id!r} "
                f"was not found in namespace {namespace!r}."
            )

        artifact = ContextArtifact.model_validate(value)

        if artifact.state == ContextArtifactState.DELETED:
            raise ContextArtifactDeleted(
                f"Context artifact {artifact_id!r} has been deleted."
            )

        return artifact