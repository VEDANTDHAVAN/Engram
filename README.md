# Engram

Temporal Memory Infrastructure for AI Applications.

Engram gives AI agents a memory layer: it stores memories with temporal
metadata (state, topics, decision history), retrieves them through a
decision-driven hybrid pipeline (vector + sparse + RRF fusion), manages
context budgets, and can park large tool outputs as restorable artifacts
instead of flooding the LLM context window.

## Repository layout

```
packages/
  core/        Configuration and shared primitives
  database/    SQLAlchemy models, session, repositories
  providers/   Memory, vector (TurboVec), sparse (BM25), graph, cache adapters
  retrieval/   Hybrid retrieval service, strategies, decision-driven policies
  decisions/   Decision providers (rule-based, Laya) and orchestrator
  context/     Context budget and usage tracking
  llm/         LLM factory and provider base
  demo/        Reproducible engineering-agent demo (see docs/DEMO.md)
scripts/       Evaluation and benchmark scripts
tests/         Test suite (pytest)
```

## Demo implementation steps

The demo (`packages/demo/`) is a self-contained, deterministic vertical
slice: an engineering agent that answers questions about a fictional
project ("TaskFlow") using Engram's memory, retrieval, and context
management. It runs fully offline — no API keys, no external services.

### 1. Deterministic demo dataset

`packages/demo/dataset.py` defines 15 realistic TaskFlow memories with
metadata: `state` (`active`/`superseded`), `topic`, `decision`
(`current`/`historical`), and memory type (`semantic`, `episodic`,
`procedural`, `long_term`). The dataset is the only place demo facts
live — the agent never hardcodes answers.

### 2. Demo runtime (composition root)

`packages/demo/runtime.py` wires the Engram components together:

- `DemoMemoryProvider` — in-process memory store seeded from the dataset.
- `DemoHashEmbeddingProvider` — deterministic local embedding stand-in
  (no embedding API needed).
- `DemoCacheProvider` — in-process cache.
- `DemoEchoLLMProvider` — deterministic LLM stand-in that echoes the
  assembled prompt, so the demo is reproducible without an LLM API.
- `DemoRuntime` — composes the above into the real Engram services:
  `HybridRetrievalService`, `DecisionOrchestrator` (with
  `RuleBasedDecisionProvider` as the deterministic fallback), and the
  context budget manager.
- `create_demo_runtime()` — factory used by the CLI and tests.

The agent depends only on Engram interfaces (retrieval service, decision
provider, context manager) — never on TurboVec, BM25, Laya, Groq, or
PostgreSQL directly.

### 3. Engineering agent

`packages/demo/agent.py` implements `EngramEngineeringAgent` with two
modes:

- **baseline** — no memory: the agent answers from the raw prompt only.
- **engram** — full pipeline: decision → retrieval policy → hybrid
  retrieval → context assembly → LLM.

The agent also demonstrates artifact management: large tool outputs
(e.g. `search_codebase`) are stored as artifacts and only a token-budgeted
summary is exposed to the LLM; the full output is restorable on demand.

### 4. Three-way memory architecture comparison

The primary demo compares three architectures under identical conditions
(same LLM, model, temperature, system prompt, dataset, namespace, embedding
model, and tool output):

- **No Memory** — query goes straight to the LLM; no retrieval.
- **Conventional Memory** — dense similarity search → fixed top-K → direct
  prompt injection.
- **Engram** — decision → retrieval policy → dense + sparse → RRF fusion →
  canonical memory → context management → DTOC → lightweight context package.

```bash
python -m scripts.evaluate_memory_systems   # full benchmark (aggregate + per-case)
python -m scripts.demo_comparison           # live side-by-side comparison
python scripts/demo_chat.py                 # interactive three-way CLI
```

Both benchmark scripts default to the deterministic echo provider (`--llm echo`)
and run offline; pass `--llm groq` for a live run. The evaluator
(`MemorySystemEvaluator`) is system-agnostic — it contains no per-system
branching, so a fourth architecture can be added without touching it.

Metrics are observed only: recall@K, MRR, answer accuracy, **memory-grounded
accuracy**, **unsupported answer rate**, temporal/conflict accuracy, abstention
rate, context tokens, latency, DTOC externalization/restoration, and tokens
saved. Grounded accuracy and unsupported rate exist because a stateless agent
can guess correctly from parametric knowledge — a correct guess is not evidence
that memory retrieval worked. Cost reports `token_usage_unavailable` because
`ChatResponse` carries no usage field.

### 5. Interactive CLI

```bash
python -m demo            # top-level entry point
python -m packages.demo   # package entry point
```

Prompts: `baseline` or `engram` mode, then ask questions about TaskFlow
(`/new` starts a fresh session, `/quit` exits). Each answer prints an
Engram trace: decision provider, retrieval strategies, retrieved memory
count, context tokens, and artifact status.

### 6. Evaluation (legacy two-way)

```bash
python -m scripts.demo_evaluation
```

Runs the same 7 cases against both modes and reports:

| Metric | Baseline | Engram |
| --- | --- | --- |
| Memory questions correct | 1/7 | 7/7 |
| Architecture consistency | 0/2 | 2/2 |
| Temporal correctness | 0/2 | 2/2 |
| Security consistency | 1/2 | 2/2 |
| Average context tokens | 534 | 116 |
| Tool output retained | No | Yes |
| Artifact restoration | No | Yes |

### 7. Tests

```bash
python -m pytest tests/demo -q   # demo tests
python -m pytest -q              # full suite (234 passed)
```

## Development

```bash
python -m pytest -q              # run the test suite
```

See `docs/DEMO.md` for the full demo walkthrough and design notes.