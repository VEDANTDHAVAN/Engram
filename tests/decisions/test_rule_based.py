import pytest

from packages.decisions import (
    DecisionRequest,
    RuleBasedDecisionProvider,
)
from packages.retrieval.policy import RetrievalStrategyName


@pytest.mark.asyncio
async def test_rule_provider_selects_sparse_for_exact_error_query() -> None:
    provider = RuleBasedDecisionProvider()

    response = await provider.decide(
        DecisionRequest(
            query="find the exact error in auth.py",
            namespace="user_001",
        )
    )

    assert response.provider == "rules"
    assert RetrievalStrategyName.SPARSE in response.decision.strategies
    assert response.confidence >= 0.0
    assert response.confidence <= 1.0


@pytest.mark.asyncio
async def test_rule_provider_selects_vector_for_semantic_query() -> None:
    provider = RuleBasedDecisionProvider()

    response = await provider.decide(
        DecisionRequest(
            query="explain the concept behind authentication",
            namespace="user_001",
        )
    )

    assert response.provider == "rules"
    assert RetrievalStrategyName.VECTOR in response.decision.strategies


@pytest.mark.asyncio
async def test_rule_provider_defaults_to_hybrid() -> None:
    provider = RuleBasedDecisionProvider()

    response = await provider.decide(
        DecisionRequest(
            query="authentication",
            namespace="user_001",
        )
    )

    assert response.decision.strategies == [
        RetrievalStrategyName.VECTOR,
        RetrievalStrategyName.SPARSE,
    ]

@pytest.mark.asyncio
async def test_rule_provider_implements_decision_provider() -> None:
    from packages.decisions import DecisionProvider

    provider = RuleBasedDecisionProvider()

    assert isinstance(provider, DecisionProvider)