from packages.retrieval.policy import (
    FusionStrategyName,
    RetrievalPolicy,
    RetrievalStrategyName,
)


def test_default_policy_uses_vector_and_sparse():
    policy = RetrievalPolicy()

    assert policy.strategies == [
        RetrievalStrategyName.VECTOR,
        RetrievalStrategyName.SPARSE,
    ]

    assert policy.fusion == FusionStrategyName.RRF
    assert policy.top_k == 10


def test_policy_accepts_string_values():
    policy = RetrievalPolicy(
        strategies=[RetrievalStrategyName.VECTOR],
        fusion=FusionStrategyName.WEIGHTED,
        top_k=5,
    )

    assert policy.strategies == [
        RetrievalStrategyName.VECTOR,
    ]

    assert policy.fusion == FusionStrategyName.WEIGHTED
    assert policy.top_k == 5


def test_duplicate_strategies_are_removed():
    policy = RetrievalPolicy(
        strategies=[
            RetrievalStrategyName.VECTOR,
            RetrievalStrategyName.SPARSE,
            RetrievalStrategyName.VECTOR,
        ]
    )

    assert policy.strategies == [
        RetrievalStrategyName.VECTOR,
        RetrievalStrategyName.SPARSE,
    ]


def test_vector_only_policy():
    policy = RetrievalPolicy(
        strategies=[RetrievalStrategyName.VECTOR],
    )

    assert policy.strategies == [
        RetrievalStrategyName.VECTOR,
    ]


def test_sparse_only_policy():
    policy = RetrievalPolicy(
        strategies=[RetrievalStrategyName.SPARSE],
    )

    assert policy.strategies == [
        RetrievalStrategyName.SPARSE,
    ]