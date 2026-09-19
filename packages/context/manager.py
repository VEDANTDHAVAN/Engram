from __future__ import annotations

from packages.context.exceptions import ContextArtifactNotFound
from packages.context.models import ActiveContextEntry, ContextArtifact, ContextArtifactState
from packages.context.store import ContextArtifactStore
from packages.context.in_memory_state import ContextStateStore

class ContextManager:
    """
    Manages which context artifacts are currently exposed to the model context.

    ContextManager does not own the canonical artifact content.
    Complete artifacts are persisted through ContextArtifactStore.
    """
    def __init__(
            self, store: ContextArtifactStore,
            state_store: ContextStateStore,
        ) -> None:
        self._store = store
        # Namespace -> ordered artifact IDs.
        # Ordering is intentionally deterministic and represents insertion order  into the current context.
        self._state_store = state_store

    async def add_artifact(
        self, *, ttl: int | None = None,
        artifact: ContextArtifact,
    ) -> ActiveContextEntry:
        """Persist an artifact and add it to the current context."""
        await self._store.store(
            artifact=artifact, ttl=ttl,
        )

        await self._state_store.add(
            namespace=artifact.namespace, artifact_id=artifact.id,
        )

        return self._to_entry(artifact)

    async def get_active_context(
        self, *,
        namespace: str,
    ) -> list[ActiveContextEntry]:
        artifact_ids = await self._state_store.get_ids(
            namespace=namespace,
        )

        entries: list[ActiveContextEntry] = []

        for artifact_id in artifact_ids:
            artifact = await self._store.get(
                artifact_id=artifact_id,
                namespace=namespace,
            )

            if artifact is None:
                continue

            entries.append(self._to_entry(artifact))

        return entries

    async def hide(
        self, *, namespace: str,
        artifact_id: str,
    ) -> ActiveContextEntry:
        """
        Hide an artifact from active model context while preserving
        the complete original artifact.
        """
        artifact = await self._store.hide(
            artifact_id=artifact_id, namespace=namespace,
        )

        return self._to_entry(artifact)

    async def restore(
        self, *, artifact_id: str, 
        namespace: str,
    ) -> ActiveContextEntry:
        """
        Restore a hidden artifact to active context.

        The original artifact is retrieved from the store; 
        no re-generation or re-fetching is performed.
        """
        artifact = await self._store.restore(
            artifact_id=artifact_id, namespace=namespace,
        )

        await self._state_store.add(
            namespace=namespace, artifact_id=artifact.id,
        )

        return self._to_entry(artifact)

    async def remove(
        self, *, namespace: str, 
        artifact_id: str,
    ) -> None:
        """
        Permanently delete an artifact from storage and remove it from the current context.
        """
        await self._store.delete(
            artifact_id=artifact_id, namespace=namespace,
        )

        await self._state_store.remove(
            namespace=namespace, artifact_id=artifact_id,
        )

    async def get_artifact(
        self, *, namespace: str, 
        artifact_id: str,
    ) -> ContextArtifact:
        """
        Retrieve the complete canonical artifact.

        This is intentionally separate from get_active_context().
        """
        artifact = await self._store.get(
            artifact_id=artifact_id, namespace=namespace,
        )

        if artifact is None:
            raise ContextArtifactNotFound(
                f"Context artifact {artifact_id!r}"
                f"was not found in namespace {namespace!r}."
            )

        return artifact

    def _to_entry(
        self, artifact: ContextArtifact,
    ) -> ActiveContextEntry:
        return ActiveContextEntry(
            artifact_id=artifact.id, namespace=artifact.namespace,
            token_count=artifact.token_count, state=artifact.state,
            source=artifact.source, metadata=artifact.metadata,
        )