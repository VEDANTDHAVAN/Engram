from .base import RetrievalStrategy
from .models import RetrievalEvidence, RetrievalQuery, RetrievalResult
from .strategies import StructuralRetrievalStrategy, TemporalRetrievalStrategy, VectorRetrievalStrategy
from .strategies.sparse import SparseRetrievalStrategy
from .hybrid import HybridRetrievalService


def __getattr__(name: str) -> object:
    if name == "RetrievalOrchestrator":
        from .orchestrator import RetrievalOrchestrator

        return RetrievalOrchestrator

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

__all__ = [
    "RetrievalStrategy",
    "RetrievalQuery",
    "RetrievalEvidence",
    "RetrievalResult",
    "VectorRetrievalStrategy",
    "TemporalRetrievalStrategy",
    "StructuralRetrievalStrategy",
    "SparseRetrievalStrategy",
    "HybridRetrievalService",
    "RetrievalOrchestrator",
]