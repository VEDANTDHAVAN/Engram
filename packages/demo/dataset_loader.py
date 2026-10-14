"""Loader for the TaskFlow demo dataset on disk.

The original inline ``TASKFLOW_MEMORIES`` tuple stays available for the legacy
two-way demo. This module reads the richer ``data/demo`` dataset used by the
three-way comparison and reuses the existing ``DemoMemory`` dataclass, so there
is exactly one in-memory memory representation in the project.
"""

from __future__ import annotations

import json
from typing import Any

from .config import DEFAULT_DATA_DIRECTORY
from .dataset import DemoMemory, large_codebase_output


def load_memories(
    path: Any = None,
) -> tuple[DemoMemory, ...]:
    """Load demo memories from ``data/demo/memories.json``."""
    source = (
        path
        if path is not None
        else DEFAULT_DATA_DIRECTORY / "memories.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    return tuple(
        DemoMemory(
            content=item["content"],
            memory_type=item["memory_type"],
            # `dataset_id` is the stable benchmark identifier. The canonical
            # memory provider generates a UUID, so the original id travels in
            # metadata to keep recall/MRR measurable without a lookup table.
            metadata={
                **dict(item.get("metadata") or {}),
                "dataset_id": item["id"],
            },
        )
        for item in payload
    )


def load_evaluation_cases(
    path: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Load evaluation cases from ``data/demo/evaluation_cases.json``."""
    source = (
        path
        if path is not None
        else DEFAULT_DATA_DIRECTORY / "evaluation_cases.json"
    )
    return tuple(json.loads(source.read_text(encoding="utf-8")))


def load_tool_output() -> str:
    """Return the deterministic large tool output.

    Reuses the existing ``large_codebase_output`` so the three-way benchmark and
    the legacy demo measure exactly the same tool payload.
    """
    return large_codebase_output()