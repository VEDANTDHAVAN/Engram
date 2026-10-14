"""Shared console rendering for the three-way benchmark.

Both ``scripts/evaluate_memory_systems.py`` and ``scripts/demo_comparison.py``
render through these helpers so the aggregate and per-case output stay
consistent. Nothing here computes a metric; it only formats ``SystemReport``
values that the evaluator already produced.
"""

from __future__ import annotations

from .config import CostConfig, DemoConfig
from .evaluator import BenchmarkResult
from .interfaces import SystemName
from .metrics import SystemReport, format_optional

_LABELS = {
    SystemName.NO_MEMORY: "No Memory",
    SystemName.CONVENTIONAL_MEMORY: "Conventional",
    SystemName.ENGRAM: "Engram",
}

_WIDTH = 14


def _label(system: SystemName) -> str:
    return _LABELS.get(system, system.value)


def _header(title: str) -> str:
    line = "=" * 60
    return f"\n{line}\n{title.center(60)}\n{line}"


def _row(name: str, values: list[str]) -> str:
    cells = [name.ljust(28)]
    cells.extend(value.rjust(_WIDTH) for value in values)
    return " ".join(cells)


def _column_header(systems: list[SystemName]) -> str:
    cells = [" " * 28]
    cells.extend(_label(s).rjust(_WIDTH) for s in systems)
    return " ".join(cells)


def render_systems(reports: list[SystemReport]) -> str:
    lines = [_header("SYSTEMS"), "-" * 60]
    for index, report in enumerate(reports, start=1):
        lines.append(f"{index}. {_label(report.system)}")
    return "\n".join(lines)


def render_accuracy(reports: list[SystemReport]) -> str:
    systems = [report.system for report in reports]
    lines = [_header("ACCURACY"), _column_header(systems), "-" * 60]

    def values(getter) -> list[str]:
        return [
            format_optional(getter(report), percent=True)
            for report in reports
        ]

    lines.append(_row("Recall@K", values(lambda r: r.recall_at_k())))
    lines.append(_row("MRR", values(lambda r: r.mrr())))
    lines.append(_row("Answer Accuracy", values(lambda r: r.accuracy())))
    lines.append(_row("Memory-Grounded", values(lambda r: r.memory_grounded_accuracy())))
    lines.append(_row("Unsupported", values(lambda r: r.unsupported_answer_rate())))
    lines.append(_row("Temporal Accuracy", values(lambda r: r.temporal_accuracy())))
    lines.append(_row("Conflict Accuracy", values(lambda r: r.conflict_accuracy())))
    lines.append(_row("Abstention Rate", values(lambda r: r.abstention_rate())))
    return "\n".join(lines)


def render_context(reports: list[SystemReport]) -> str:
    systems = [report.system for report in reports]
    lines = [_header("CONTEXT"), _column_header(systems), "-" * 60]

    def values(getter) -> list[str]:
        return [
            f"{getter(report):,.0f}" for report in reports
        ]

    lines.append(_row("Memory Tokens", values(lambda r: r.average_memory_tokens)))
    lines.append(_row("Tool Tokens", values(lambda r: r.average_tool_tokens)))
    lines.append(_row("Exposed Tool Tokens", values(lambda r: r.average_exposed_tool_tokens)))
    lines.append(_row("Final Context Tokens", values(lambda r: r.average_final_context_tokens)))
    return "\n".join(lines)


def render_latency(reports: list[SystemReport]) -> str:
    systems = [report.system for report in reports]
    lines = [_header("LATENCY"), _column_header(systems), "-" * 60]

    def values(getter) -> list[str]:
        return [f"{getter(report):,.0f}" for report in reports]

    lines.append(_row("Avg Latency (ms)", values(lambda r: r.average_latency_ms)))
    lines.append(_row("P95 Latency (ms)", values(lambda r: r.p95_latency_ms)))
    return "\n".join(lines)


def render_dtoc(reports: list[SystemReport]) -> str:
    engram = next(
        (report for report in reports if report.system is SystemName.ENGRAM),
        None,
    )
    if engram is None:
        return ""

    lines = [_header("DTOC (Engram)"), "-" * 60]
    lines.append(
        "Artifact Externalization Rate: "
        + format_optional(engram.artifact_externalization_rate(), percent=True)
    )
    lines.append(
        "Restoration Success Rate: "
        + format_optional(engram.restoration_success_rate(), percent=True)
    )
    lines.append(
        "Tokens Saved: "
        + format_optional(engram.tokens_saved())
    )
    return "\n".join(lines)


def render_cost(reports: list[SystemReport], cost: CostConfig) -> str:
    systems = [report.system for report in reports]
    lines = [_header("COST"), _column_header(systems), "-" * 60]
    if not cost.is_configured:
        lines.append("token_usage_unavailable (no pricing configured)")
        return "\n".join(lines)

    def values(getter) -> list[str]:
        return [
            format_optional(getter(report)) if getter(report) is not None
            else "token_usage_unavailable"
            for report in reports
        ]

    lines.append(_row("Total Cost ($)", values(lambda r: r.total_cost(cost))))
    lines.append(_row("Cost/Success ($)", values(lambda r: r.cost_per_successful_answer(cost))))
    return "\n".join(lines)


def render_case(
    index: int,
    case: dict,
    reports: list[SystemReport],
) -> str:
    """Render one case across all systems, in reporting order."""
    by_system = {report.system: report for report in reports}
    lines = [f"CASE {index:02d}", "-" * 60]
    lines.append("Question:")
    lines.append(f"  {case['question']}")
    lines.append("")

    for report in reports:
        outcome = next(
            (item for item in report.outcomes if item.case_id == case["id"]),
            None,
        )
        label = _label(report.system)
        if outcome is None:
            lines.append(f"{label}: (no outcome)")
            continue
        lines.append(f"{label}:")
        lines.append(f"  {outcome.answer.strip()}")
        lines.append(
            f"  {outcome.status}  "
            f"latency={outcome.latency_ms:,.0f}ms  "
            f"tokens={outcome.context_tokens}  "
            f"retrieved={len(outcome.retrieved_memory_ids)}"
        )
        if outcome.forbidden_hit:
            lines.append(f"  forbidden term hit: {outcome.forbidden_hit}")
        lines.append("")

    lines.append(f"Expected: {case.get('expected_answer', '')}")
    return "\n".join(lines)


def render_restoration(result: BenchmarkResult) -> str:
    lines = [_header("ARTIFACT RESTORATION"), "-" * 60]
    for system, verified in result.restoration_verified.items():
        lines.append(f"{_label(system)}: {'verified' if verified else 'not verified'}")
    return "\n".join(lines)


def render_all(
    result: BenchmarkResult,
    *,
    config: DemoConfig,
    cases: list[dict],
    show_cases: bool = True,
) -> str:
    reports = result.reports
    parts = [
        _header("ENGRAM MEMORY SYSTEM EVALUATION"),
        f"Dataset: TaskFlow",
        f"Cases: {len(cases)}",
        "",
        render_systems(reports),
        render_accuracy(reports),
        render_context(reports),
        render_latency(reports),
        render_dtoc(reports),
        render_cost(reports, config.cost),
        render_restoration(result),
    ]
    if show_cases:
        parts.append(_header("PER-CASE RESULTS"))
        for index, case in enumerate(cases, start=1):
            parts.append(render_case(index, case, reports))
    return "\n".join(parts)