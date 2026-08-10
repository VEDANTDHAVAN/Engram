from packages.state_engine.types import MemoryType

from .base import MemoryRouter
from .models import RoutingDecision

class RuleBasedMemoryRouter(MemoryRouter):
    """
    Lightweight deterministic router.

    This is intentionally simple for the MVP.
    """
    HISTORY_KEYWORDS = {
        "before", "previous", "history", "last time", 
        "earlier", "previously", "what happened",
    }

    PREFERENCE_KEYWORDS = {
        "prefer", "preference", "likes",
        "favorite", "usually", "always",
    }

    FACT_KEYWORDS = {
        "what is", "what are", "does", "support",
        "specification", "specs", "fact",
    }

    PROCEDURE_KEYWORDS = {
        "how do", "how to", "steps", "process", "procedure",
        "workflow", "configure", "setup", "set up",
    }

    def _contains(
        self, query: str, keywords: set[str],
    ) -> bool:
        return any(
            keyword in query for keyword in keywords
        )

    async def route(
        self, query: str,
    ) -> RoutingDecision:
        normalized = query.lower().strip()

        memory_types: list[MemoryType] = []
        reasons: list[str] = []

        if self._contains(
            normalized, self.HISTORY_KEYWORDS,
        ):
            memory_types.append(
                MemoryType.EPISODIC
            )
            reasons.append(
                "Query appears to request historical events."
            )

        if self._contains(
            normalized, self.PREFERENCE_KEYWORDS,
        ):
            memory_types.append(
                MemoryType.LONG_TERM
            )

            reasons.append(
                "Query appears to concern persistent preferences."
            )

        if self._contains(
            normalized, self.FACT_KEYWORDS,
        ):
            memory_types.append(
                MemoryType.SEMANTIC
            )

            reasons.append(
                "Query appears to request factual knowledge."
            )

        if self._contains(
            normalized, self.PROCEDURE_KEYWORDS,
        ):
            memory_types.append(
                MemoryType.PROCEDURAL
            )

            reasons.append(
                "Query appears to request a procedure or workflow."
            )

        # If nothing specific was detected, semantic memory
        # is the safest default for a knowledge query.
        if not memory_types:
            memory_types.append(
                MemoryType.SEMANTIC
            )

            reasons.append(
                "No specialized memory intent detected."
            )

        return RoutingDecision(
            memory_types=memory_types, reasons=reasons,
            confidence=0.7 if len(memory_types) == 1 else 0.85,
        )