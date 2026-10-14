"""Tests for the three-way memory-system comparison.

Covers the three ``MemorySystem`` implementations, DTOC externalization, the
metric helpers, and the system-agnostic evaluator. All tests run offline using
the deterministic hash embedder and echo LLM.
"""

from __future__ import annotations

import pytest

from packages.demo import (
    ConventionalMemorySystem,
    DemoConfig,
    EngramSystem,
    MemorySystemEvaluator,
    NoMemorySystem,
    SystemName,
    SystemReport,
    build_memory_systems,
    create_demo_runtime,
)
from packages.demo.metrics import (
    contains_all_terms,
    first_forbidden_hit,
    looks_like_abstention,
    normalize,
    reciprocal_rank_of,
)


@pytest.fixture
async def runtime():
    config = DemoConfig()
    rt = create_demo_runtime(namespace=config.namespace)
    await rt.initialize()
    await rt.seed_taskflow()
    try:
        yield rt
    finally:
        rt.close()


@pytest.fixture
def config() -> DemoConfig:
    return DemoConfig()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_normalize_collapses_punctuation():
    assert normalize("Don't, stop!") == "don t stop"


def test_contains_all_terms():
    assert contains_all_terms("Cursor pagination is required", ["cursor", "pagination"])
    assert not contains_all_terms("Cursor pagination", ["cursor", "offset"])


def test_first_forbidden_hit():
    assert first_forbidden_hit("Use offset pagination", ["offset"]) == "offset"
    assert first_forbidden_hit("Use cursor pagination", ["offset"]) is None


def test_reciprocal_rank_of():
    assert reciprocal_rank_of(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank_of(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank_of(["a"], []) == 0.0


def test_looks_like_abstention():
    assert looks_like_abstention("I do not have that information.")
    assert looks_like_abstention("I don't have any data on that.")
    assert not looks_like_abstention("Cursor pagination is required.")


def test_memory_grounded_and_unsupported_metrics():
    from packages.demo.metrics import CaseOutcome

    def make_outcome(
        case_id: str,
        *,
        correct: bool,
        abstention_correct: bool,
        retrieved: list[str],
    ) -> CaseOutcome:
        return CaseOutcome(
            case_id=case_id,
            category="no_memory_knowledge",
            system=SystemName.NO_MEMORY,
            question="q",
            answer="a",
            expected_answer="a",
            correct=correct,
            requires_memory=True,
            abstention_correct=abstention_correct,
            forbidden_hit=None,
            latency_ms=1.0,
            retrieved_memory_ids=retrieved,
            recall_hit=False,
            reciprocal_rank=0.0,
            context_tokens=10,
            response=None,
        )

    outcomes = [
        make_outcome("c1", correct=True, abstention_correct=False, retrieved=[]),
        make_outcome("c2", correct=True, abstention_correct=False, retrieved=["m1"]),
        make_outcome("c3", correct=False, abstention_correct=True, retrieved=[]),
    ]
    report = SystemReport(system=SystemName.NO_MEMORY, outcomes=outcomes)
    # 1 of 3 correct answers is grounded in retrieved evidence.
    assert report.memory_grounded_accuracy() == 1 / 3
    # 1 of 3 answered without evidence and without abstaining.
    assert report.unsupported_answer_rate() == 1 / 3


# ---------------------------------------------------------------------------
# NoMemorySystem
# ---------------------------------------------------------------------------


async def test_no_memory_does_not_retrieve(runtime, config):
    system = NoMemorySystem(
        runtime=runtime, llm_provider=runtime.llm_provider, config=config,
    )
    response = await system.answer(
        "What pagination strategy did TaskFlow adopt?",
        namespace=config.namespace,
    )
    assert response.system is SystemName.NO_MEMORY
    assert response.trace.retrieved_memory_ids == []
    assert response.trace.retrieved_memory_count == 0
    assert system.uses_memory is False
    assert system.uses_externalization is False


# ---------------------------------------------------------------------------
# ConventionalMemorySystem
# ---------------------------------------------------------------------------


async def test_conventional_retrieves_dense_only(runtime, config):
    system = ConventionalMemorySystem(
        runtime=runtime, llm_provider=runtime.llm_provider, config=config,
    )
    response = await system.answer(
        "What pagination strategy did TaskFlow adopt?",
        namespace=config.namespace,
    )
    assert response.system is SystemName.CONVENTIONAL_MEMORY
    assert response.trace.strategies == ["vector"]
    assert response.trace.fusion is None
    assert response.trace.retrieved_memory_count > 0
    assert system.uses_memory is True
    assert system.uses_externalization is False


# ---------------------------------------------------------------------------
# EngramSystem
# ---------------------------------------------------------------------------


async def test_engram_uses_hybrid_retrieval(runtime, config):
    system = EngramSystem(
        runtime=runtime,
        llm_provider=runtime.llm_provider,
        config=config,
        decision_provider=runtime.decision_provider,
    )
    response = await system.answer(
        "What pagination strategy did TaskFlow adopt?",
        namespace=config.namespace,
    )
    assert response.system is SystemName.ENGRAM
    assert len(response.trace.strategies) >= 1
    assert response.trace.fusion is not None
    assert response.trace.retrieved_memory_count > 0
    assert system.uses_memory is True
    assert system.uses_externalization is True


async def test_engram_externalizes_large_tool_output(runtime, config):
    from packages.demo.dataset import large_codebase_output

    system = EngramSystem(
        runtime=runtime,
        llm_provider=runtime.llm_provider,
        config=config,
        decision_provider=runtime.decision_provider,
    )
    tool_output = large_codebase_output()
    response = await system.answer(
        "Summarize the search results.",
        namespace=config.namespace,
        tool_output=tool_output,
        artifact_id=config.artifact_id,
    )
    trace = response.trace
    assert trace.tool_output_tokens > 0
    assert trace.exposed_tool_tokens < trace.tool_output_tokens
    assert trace.artifact_state == "hidden"
    assert trace.artifact_restorable is True
    assert trace.tokens_saved is not None and trace.tokens_saved > 0


async def test_engram_restores_artifact(runtime, config):
    from packages.demo.dataset import large_codebase_output

    system = EngramSystem(
        runtime=runtime,
        llm_provider=runtime.llm_provider,
        config=config,
        decision_provider=runtime.decision_provider,
    )
    tool_output = large_codebase_output()
    await system.answer(
        "Summarize the search results.",
        namespace=config.namespace,
        tool_output=tool_output,
        artifact_id=config.artifact_id,
    )
    restored = await system.answer(
        "Retrieve the original output.",
        namespace=config.namespace,
        restore_artifact_id=config.artifact_id,
    )
    assert restored.trace.exposed_tool_tokens > 0
    assert tool_output in restored.prompt


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


async def test_evaluator_runs_all_systems(runtime, config):
    systems = build_memory_systems(runtime, config)
    cases = [
        {
            "id": "case_01_knowledge",
            "category": "no_memory_knowledge",
            "question": "What pagination strategy did TaskFlow adopt?",
            "expected_terms": ["cursor"],
            "expected_answer": "Cursor pagination",
            "supporting_memory_ids": [],
            "forbidden_terms": [],
            "requires_memory": True,
            "tool": None,
        }
    ]
    evaluator = MemorySystemEvaluator(config=config)
    result = await evaluator.run(systems, cases)
    assert len(result.reports) == 3
    assert {r.system for r in result.reports} == {
        SystemName.NO_MEMORY,
        SystemName.CONVENTIONAL_MEMORY,
        SystemName.ENGRAM,
    }
    for report in result.reports:
        assert len(report.outcomes) == 1
