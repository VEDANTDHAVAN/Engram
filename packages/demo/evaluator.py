"""System-agnostic evaluation driver.

The evaluator knows only ``MemorySystem``. It contains no per-system branch, so
a fourth architecture could be added without touching this module.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import DemoConfig
from .dataset_loader import load_tool_output
from .interfaces import MemorySystem, SystemName, SystemResponse
from .metrics import (
    CaseOutcome,
    SystemReport,
    contains_all_terms,
    first_forbidden_hit,
    looks_like_abstention,
    reciprocal_rank_of,
)


ARTIFACT_CATEGORY = "artifact_restoration"


@dataclass(slots=True)
class BenchmarkResult:
    """Full benchmark output: one report per system, in reporting order."""

    reports: list[SystemReport] = field(default_factory=list)
    restoration_verified: dict[SystemName, bool] = field(
        default_factory=dict
    )

    def report_for(self, system: SystemName) -> SystemReport | None:
        for report in self.reports:
            if report.system is system:
                return report
        return None

    @property
    def systems(self) -> list[SystemName]:
        return [report.system for report in self.reports]


class MemorySystemEvaluator:
    """Runs every evaluation case against every system under comparison."""

    def __init__(
        self,
        *,
        config: DemoConfig,
        tool_output: str | None = None,
        concurrency: int = 4,
    ) -> None:
        self._config = config
        self._tool_output = tool_output
        self._concurrency = max(1, concurrency)

    async def run(
        self,
        systems: Sequence[MemorySystem],
        cases: Sequence[dict[str, Any]],
    ) -> BenchmarkResult:
        result = BenchmarkResult()
        tool_output = self._resolve_tool_output(cases)

        # Systems run one after another: the artifact store is shared, so a
        # tool case from one system must not race a tool case from another.
        for system in systems:
            outcomes = await self._run_system(system, cases, tool_output)
            verified = await self._verify_system_restoration(
                system, cases, outcomes
            )
            result.reports.append(
                SystemReport(system=system.name, outcomes=outcomes)
            )
            result.restoration_verified[system.name] = verified

        return result

    async def _run_system(
        self,
        system: MemorySystem,
        cases: Sequence[dict[str, Any]],
        tool_output: str | None,
    ) -> list[CaseOutcome]:
        # Restore cases read an artifact that a tool case creates, so they run
        # after the rest. Everything else runs concurrently, bounded.
        restore_cases = [c for c in cases if self._restore_target(c)]
        other_cases = [c for c in cases if not self._restore_target(c)]

        semaphore = asyncio.Semaphore(self._concurrency)

        async def run_one(case: dict[str, Any]) -> CaseOutcome:
            async with semaphore:
                case_tool_output, artifact_id = self._case_tool(
                    case, tool_output
                )
                response = await system.answer(
                    case["question"],
                    namespace=self._config.namespace,
                    tool_output=case_tool_output,
                    artifact_id=artifact_id,
                    restore_artifact_id=None,
                )
            return self._score(case, response)

        outcomes = list(
            await asyncio.gather(*(run_one(c) for c in other_cases))
        )
        for case in restore_cases:
            response = await system.answer(
                case["question"],
                namespace=self._config.namespace,
                tool_output=None,
                artifact_id=None,
                restore_artifact_id=self._restore_target(case),
            )
            outcomes.append(self._score(case, response))

        # Preserve the original case order in the report.
        order = {case["id"]: index for index, case in enumerate(cases)}
        outcomes.sort(key=lambda item: order[item.case_id])
        return outcomes

    async def _verify_system_restoration(
        self,
        system: MemorySystem,
        cases: Sequence[dict[str, Any]],
        outcomes: list[CaseOutcome],
    ) -> bool:
        verified = True
        by_id = {item.case_id: item for item in outcomes}
        for case in cases:
            target = self._restore_target(case)
            if not target:
                continue
            outcome = by_id.get(case["id"])
            if outcome is None:
                continue
            verified = verified and await self._verify_restoration(
                system, outcome.response
            )
        return verified

    # -- case plumbing -------------------------------------------------

    def _resolve_tool_output(
        self, cases: Sequence[dict[str, Any]],
    ) -> str | None:
        if self._tool_output is not None:
            return self._tool_output
        if any(case.get("tool") for case in cases):
            return load_tool_output()
        return None

    def _case_tool(
        self, case: dict[str, Any], tool_output: str | None,
    ) -> tuple[str | None, str | None]:
        if not case.get("tool"):
            return None, None
        return tool_output, case.get("artifact_id", self._config.artifact_id)

    def _restore_target(self, case: dict[str, Any]) -> str | None:
        return case.get("requires_artifact")

    async def _verify_restoration(
        self, system: MemorySystem, response: SystemResponse,
    ) -> bool:
        """Verify the exposed artifact body matches the stored original.

        Hidden must mean recoverable, not lost. The comparison is exact: the
        original tool output must appear verbatim in what the LLM received.
        """
        original = self._resolve_tool_output(())
        if original is None:
            return True

        exposed = response.trace.exposed_tool_tokens
        return exposed > 0 and original in response.prompt

    # -- scoring -------------------------------------------------------

    def _score(
        self, case: dict[str, Any], response: SystemResponse,
    ) -> CaseOutcome:
        expected_terms = list(case.get("expected_terms") or [])
        forbidden_terms = list(case.get("forbidden_terms") or [])
        supporting = list(case.get("supporting_memory_ids") or [])

        requires_memory = case.get("requires_memory", True)
        answer_ok = (
            contains_all_terms(response.answer, expected_terms)
            if requires_memory
            else True
        )
        forbidden_hit = first_forbidden_hit(response.answer, forbidden_terms)

        # A restoration case is only correct if the system actually restored
        # the artifact (its exposed tool tokens are non-zero).
        restore_target = self._restore_target(case)
        restored_ok = (
            response.trace.exposed_tool_tokens > 0
            if restore_target
            else True
        )

        correct = bool(answer_ok) and forbidden_hit is None and restored_ok

        retrieved = list(response.trace.retrieved_memory_ids)
        recall_hit = bool(
            [item for item in retrieved if item in set(supporting)]
        ) if supporting else None

        return CaseOutcome(
            case_id=case["id"],
            category=case["category"],
            system=response.system,
            question=case["question"],
            answer=response.answer,
            expected_answer=case.get("expected_answer", ""),
            correct=correct,
            requires_memory=bool(requires_memory),
            abstention_correct=(
                looks_like_abstention(response.answer)
                if case.get("requires_memory", True)
                else False
            ),
            forbidden_hit=forbidden_hit,
            latency_ms=response.latency_ms,
            retrieved_memory_ids=retrieved,
            recall_hit=bool(recall_hit),
            reciprocal_rank=reciprocal_rank_of(retrieved, supporting),
            context_tokens=response.trace.final_context_tokens,
            response=response,
        )