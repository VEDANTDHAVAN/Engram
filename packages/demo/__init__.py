"""Engram's reproducible engineering-agent demonstration.

Two demos share this package:

* The **legacy two-way demo** (``AgentMode`` BASELINE vs ENGRAM) — preserved
  unchanged per the three-way spec.
* The **primary three-way comparison** (No Memory vs Conventional Memory vs
  Engram) built on the ``MemorySystem`` interface.
"""

from .agent import (
    AgentMode,
    AgentResponse,
    EngramEngineeringAgent,
    EngramTrace,
)
from .benchmark import (
    build_memory_systems,
    create_benchmark_runtime,
    create_embedding_provider,
    create_llm_provider,
)
from .config import CostConfig, DemoConfig
from .dataset_loader import (
    load_evaluation_cases,
    load_memories,
    load_tool_output,
)
from .evaluator import BenchmarkResult, MemorySystemEvaluator
from .interfaces import (
    MemorySystem,
    MemoryTrace,
    SystemName,
    SystemResponse,
)
from .metrics import CaseOutcome, SystemReport
from .report import render_all
from .runtime import (
    DemoEchoLLMProvider,
    DemoRuntime,
    create_demo_runtime,
)
from .systems import (
    ConventionalMemorySystem,
    EngramSystem,
    NoMemorySystem,
)

__all__ = [
    # Legacy two-way demo
    "AgentMode",
    "AgentResponse",
    "DemoEchoLLMProvider",
    "DemoRuntime",
    "EngramEngineeringAgent",
    "EngramTrace",
    "create_demo_runtime",
    # Three-way comparison
    "BenchmarkResult",
    "CaseOutcome",
    "ConventionalMemorySystem",
    "CostConfig",
    "DemoConfig",
    "EngramSystem",
    "MemorySystem",
    "MemorySystemEvaluator",
    "MemoryTrace",
    "NoMemorySystem",
    "SystemName",
    "SystemReport",
    "SystemResponse",
    "build_memory_systems",
    "create_benchmark_runtime",
    "create_embedding_provider",
    "create_llm_provider",
    "load_evaluation_cases",
    "load_memories",
    "load_tool_output",
    "render_all",
]