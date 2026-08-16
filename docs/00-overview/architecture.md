## Dense Semantic Retrieval

CAML's vector retrieval strategy is implemented through the existing
RetrievalStrategy abstraction.

VectorRetrievalStrategy depends on provider interfaces:

    BaseEmbeddingProvider
    VectorProvider
    MemoryProvider

The retrieval flow is:

    RetrievalQuery
        |
        v
    VectorRetrievalStrategy
        |
        v
    EmbeddingProvider
        |
        v
    Query Embedding
        |
        v
    VectorProvider
        |
        v
    Dense Candidates
        |
        +-- memory_id
        +-- similarity score
        |
        v
    MemoryProvider
        |
        v
    Canonical MemoryRecord
        |
        v
    RetrievalEvidence

The vector backend is not authoritative for memory content.

For TurboVec:

    TurboVec = candidate generation
    PostgreSQL = canonical memory hydration

This separation allows TurboVec to be replaced by another
VectorProvider without changing the retrieval strategy.

Stale vector-index entries that cannot be hydrated from the
MemoryProvider are discarded rather than returned to the model.

## Namespace Isolation in Vector Retrieval

TurboVec does not own CAML namespace authorization.

PostgreSQL remains the canonical source of namespace membership.

Before dense vector retrieval, CAML resolves the IDs authorized for
the requested namespace through MemoryProvider.

Those IDs are passed to VectorProvider as an explicit allowlist.

    RetrievalQuery
        |
        v
    MemoryProvider
        |
        | namespace
        v
    Authorized Memory IDs
        |
        v
    VectorProvider
        |
        | allowed_ids
        v
    TurboVec
        |
        | native allowlist
        v
    Dense Candidates

This prevents unrestricted cross-tenant vector retrieval.

TurboVec therefore remains a specialized dense vector index rather
than an authorization or canonical memory store.