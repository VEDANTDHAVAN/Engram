import pytest

from packages.routing import RuleBasedMemoryRouter
from packages.state_engine import MemoryType

@pytest.mark.asyncio
async def test_history_query():
    router = RuleBasedMemoryRouter()

    result = await router.route(
        "What happened in our previous meeting?"
    )

    assert MemoryType.EPISODIC in result.memory_types

@pytest.mark.asyncio
async def test_preference_query():
    router = RuleBasedMemoryRouter()

    result = await router.route(
        "What database dose the customer prefer?"
    )

    assert MemoryType.LONG_TERM in result.memory_types

@pytest.mark.asyncio
async def test_procedure_query():
    router = RuleBasedMemoryRouter()

    result = await router.route(
        "How do I configure the deployment workflow?"
    )

    assert MemoryType.PROCEDURAL in result.memory_types

@pytest.mark.asyncio
async def test_default_semantic():
    router = RuleBasedMemoryRouter()

    result = await router.route(
        "What does our product support?"
    )

    assert MemoryType.SEMANTIC in result.memory_types