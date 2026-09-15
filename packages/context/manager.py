from __future__ import annotations

from packages.context.exceptions import ContextArtifactNotFound
from packages.context.models import ActiveContextEntry, ContextArtifact, ContextArtifactState
from packages.context.store import ContextArtifactStore

class ContextManager:
    """
    Manages which context artifacts are currently exposed to the model context.

    ContextManager does not own the canonical artifact content.
    Complete artifacts are persisted through ContextArtifactStore.
    """
    def __init__(self, store: ContextArtifactStore) -> None:
        self._store = store
        # Namespace -> ordered artifact IDs.
        # Ordering is intentionally deterministic and represents insertion order  into the current context.
        self._active_context: dict[str, list[str]] = {}

    async def add_artifact(
        self, *, ttl: int | None = None,
        artifact: ContextArtifact,
    ) -> ActiveContextEntry:
        """Persist an artifact and add it to the current context."""
        await self._store.store(
            artifact=artifact, ttl=ttl,
        )
        namespace_entries = self._active_context.setdefault(artifact.namespace, [])

        if artifact.id not in namespace_entries:
            namespace_entries.append(artifact.id)

        return self._to_entry(artifact)

    async def get_active_context(
        self, *, namespace: str,
    ) -> list[ActiveContextEntry]:
        """
        Return the current context entries for a namespace.

        Hidden artifacts remain represented as lightweight references.
        They are not returned with their full content.
        """
        artifact_ids = self._active_context.get(namespace, [])
        entries: list[ActiveContextEntry] = []

        for artifact_id in artifact_ids:
            artifact = await self._store.get(
                artifact_id=artifact_id, namespace=namespace,
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
        namespace_entries = self._active_context.setdefault(namespace, [])

        if artifact.id not in namespace_entries:
            namespace_entries.append(artifact.id)

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
        namespace_entries = self._active_context.get(namespace)

        if namespace_entries is None:
            return

        self._active_context[namespace] = [
            current_id for current_id in namespace_entries
            if current_id != artifact_id
        ]

        if not self._active_context[namespace]:
            del self._active_context[namespace]

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