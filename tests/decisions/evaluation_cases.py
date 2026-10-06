from packages.retrieval.policy import RetrievalStrategyName
from packages.decisions.evaluation import RoutingEvaluationCase


VECTOR = frozenset({
    RetrievalStrategyName.VECTOR,
})

SPARSE = frozenset({
    RetrievalStrategyName.SPARSE,
})

HYBRID = frozenset({
    RetrievalStrategyName.VECTOR,
    RetrievalStrategyName.SPARSE,
})


CASES = [
    RoutingEvaluationCase(
        query="Why does vector search work well for semantic similarity?",
        expected=VECTOR,
    ),
    RoutingEvaluationCase(
        query="Explain the meaning of distributed consensus.",
        expected=VECTOR,
    ),
    RoutingEvaluationCase(
        query="How are embeddings related to retrieval?",
        expected=VECTOR,
    ),
    RoutingEvaluationCase(
        query="Find ValueError in auth.py.",
        expected=SPARSE,
    ),
    RoutingEvaluationCase(
        query="Find function calculate_score.",
        expected=SPARSE,
    ),
    RoutingEvaluationCase(
        query="Find filename settings.py.",
        expected=SPARSE,
    ),
    RoutingEvaluationCase(
        query="Find identifier USER_123.",
        expected=SPARSE,
    ),
    RoutingEvaluationCase(
        query="Why did auth.py fail with error ValueError?",
        expected=HYBRID,
    ),
    RoutingEvaluationCase(
        query="Explain the authentication error and find its source.",
        expected=HYBRID,
    ),
    RoutingEvaluationCase(
        query="Understand why function calculate_score raises this exception.",
        expected=HYBRID,
    ),
]