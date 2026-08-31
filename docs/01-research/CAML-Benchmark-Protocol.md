# CAML Benchmark Protocol

## 1. Purpose

The purpose of this benchmark is to determine whether adding the **Context-Aware Memory Layer (CAML)** to an existing Agentic AI system produces measurable improvements in:

* Task success
* Retrieval quality
* Context relevance
* Faithfulness
* Memory recall
* Token efficiency
* API cost
* Latency
* Tool efficiency
* Hallucination rate

The benchmark must evaluate whether CAML's additional architectural complexity produces a **net positive benefit** over the existing architecture.

The central research question is:

> **Does CAML enable an existing Agentic AI system to complete tasks more accurately and reliably while using less useful context and maintaining acceptable latency and cost?**

---

# 2. Experimental Principle

The comparison must be a controlled experiment.

We will maintain the same:

* LLM
* LLM version
* Agent
* Agent instructions
* Tools
* Dataset
* Embedding model
* Hardware
* Model parameters
* Task definitions

The primary variable will be the **context/memory architecture**.

This prevents improvements from being incorrectly attributed to a different model, prompt, or agent implementation.

---

# 3. Baseline Architecture

The baseline represents a conventional production Agentic AI system.

```text
User
  │
  ▼
Agent
  │
  ▼
Existing Retrieval / Memory
  │
  ▼
Retrieved Context
  │
  ▼
LLM
  │
  ▼
Tool Execution
  │
  ▼
Final Response
```

Depending on the baseline system, retrieval may consist of:

* Vector search
* Conversation history
* Top-K retrieval
* Basic memory
* Framework-provided memory

The baseline must be documented before experimentation.

---

# 4. CAML Architecture

The experimental architecture introduces CAML between the agent and memory infrastructure.

```text
User
 │
 ▼
Agent
 │
 ▼
CAML
 │
 ├── Intent Analysis
 ├── Retrieval Planning
 ├── Memory Routing
 ├── Dense Retrieval
 ├── Sparse Retrieval
 ├── Graph Retrieval
 ├── Evidence Fusion
 ├── Reranking
 ├── Deduplication
 ├── Context Compression
 └── Context Packaging
 │
 ▼
LLM
 │
 ▼
Tools
```

CAML may use:

```text
Redis       → Working Memory
PostgreSQL  → Episodic / Long-Term / Procedural Memory
TurboVec    → Dense Semantic Retrieval
BM25        → Sparse Retrieval
HelixDB     → Knowledge Graph Retrieval
```

The use of any specific backend must remain configurable.

---

# 5. Benchmark Dataset

The benchmark should contain approximately:

```text
500–1,000 tasks
```

distributed across different context requirements.

### Task Categories

| Category         | Purpose                                 |
| ---------------- | --------------------------------------- |
| No-memory        | Determine CAML overhead on simple tasks |
| Working-memory   | Test current-session context            |
| Episodic         | Test historical experience retrieval    |
| Semantic         | Test factual knowledge retrieval        |
| Procedural       | Test workflow retrieval                 |
| Long-term        | Test cross-session memory               |
| Multi-hop        | Test graph-based reasoning              |
| Conflicting      | Test stale/conflicting memory handling  |
| Long-context     | Test context-window management          |
| Distractor-heavy | Test retrieval precision                |
| Multi-agent      | Test shared/isolation memory            |
| Tool-dependent   | Test context-driven tool selection      |

This is important because CAML should not be expected to improve every query.

---

# 6. Ground Truth

Each benchmark task should have a structured ground-truth record.

Example:

```json
{
  "task_id": "task_047",
  "query": "What database did we decide to use for semantic memory?",
  "expected_answer": "TurboVec",
  "required_memory_ids": [
    "memory_182"
  ],
  "memory_type": "semantic",
  "distractor_memory_ids": [
    "memory_091",
    "memory_234"
  ],
  "expected_evidence": [
    "architecture_decision_18"
  ]
}
```

Ground truth should identify not only the expected answer but also the evidence required to produce it.

---

# 7. Experimental Configurations

We should not compare only:

```text
Baseline vs CAML
```

because this would make it difficult to determine which CAML component produced the improvement.

Instead, conduct an ablation study.

### Configuration A — Baseline

```text
Existing Agent
+
Existing Retrieval
```

### Configuration B — Improved Vector Retrieval

```text
Baseline
+
TurboVec
```

### Configuration C — Hybrid Retrieval

```text
Baseline
+
Vector
+
BM25
```

### Configuration D — Graph Retrieval

```text
Baseline
+
Vector
+
HelixDB
```

### Configuration E — Hybrid + Graph

```text
Baseline
+
Vector
+
BM25
+
HelixDB
```

### Configuration F — Full CAML

```text
Baseline
+
Intent Analysis
+
Retrieval Planning
+
Memory Routing
+
Vector
+
BM25
+
Graph
+
Reranking
+
Compression
+
Context Packaging
```

This allows us to determine the marginal contribution of each architectural component.

---

# 8. Primary Metrics

## 8.1 Task Success Rate

The percentage of tasks successfully completed.

```text
Task Success Rate =
Successful Tasks
────────────────── × 100
Total Tasks
```

This should be the primary system-level metric.

---

# 9. Retrieval Metrics

For tasks with known relevant memories:

### Precision@K

```text
Relevant Retrieved Items
─────────────────────────
Total Retrieved Items
```

### Recall@K

```text
Relevant Retrieved Items
────────────────────────
Total Relevant Items
```

### MRR

Measures how highly the first relevant result appears.

### nDCG

Measures ranking quality while accounting for relevance grades.

These metrics determine whether CAML is actually retrieving better evidence.

---

# 10. Context Metrics

Measure:

### Context Relevance

Percentage of injected context that contributes to the task.

### Context Redundancy

Amount of duplicate or semantically overlapping information.

### Context Compression Ratio

```text
Original Context Tokens
───────────────────────
Final Context Tokens
```

### Context Utilization

Percentage of supplied information that is actually useful to the task.

---

# 11. Token Efficiency

Measure:

```text
Baseline Input Tokens
vs
CAML Input Tokens
```

Calculate:

```text
Token Reduction %

=
(Baseline Tokens - CAML Tokens)
───────────────────────────── × 100
       Baseline Tokens
```

However, **total system tokens** must also be measured.

CAML may introduce additional LLM calls for:

* Intent analysis
* Retrieval planning
* Compression

Therefore:

```text
Total Tokens
=
Planning Tokens
+
Retrieval Tokens
+
Final Prompt Tokens
+
Output Tokens
```

We must not claim token efficiency based only on the final prompt.

---

# 12. Cost

Calculate the complete cost of each experiment.

```text
Total Cost
=
LLM Cost
+
Embedding Cost
+
CAML Compute
+
Vector DB Cost
+
Graph DB Cost
+
Other API Costs
```

Primary metric:

```text
Cost per Successful Task
```

This is more meaningful than simply measuring cost per request.

---

# 13. Latency

Measure end-to-end latency.

```text
Request
   │
   ├── Agent
   ├── CAML
   ├── Retrieval
   ├── LLM
   └── Tools
   │
   ▼
Response
```

Record:

* p50
* p95
* p99

Also record individual CAML components:

```text
Intent Analysis
Retrieval Planning
TurboVec
BM25
HelixDB
Reranking
Compression
Packaging
```

---

# 14. Agent Performance Metrics

Measure:

### Tool Selection Accuracy

Did the agent select the correct tool?

### Tool Execution Success

Did the tool execute successfully?

### Tool Call Count

Average number of tool calls per task.

### Task Completion Rate

Did the agent actually accomplish the requested objective?

### Recovery Rate

Can the agent recover from tool/API failures?

---

# 15. Memory Metrics

CAML must additionally be evaluated on memory-specific tasks.

### Cross-Session Recall

Can the agent retrieve information from previous sessions?

### Preference Recall

Can it correctly remember persistent preferences?

### Memory Freshness

Does it prioritize newer information?

### Conflict Resolution

Given:

```text
Old Decision
      +
New Decision
```

does the system select the correct current state?

### Memory Consolidation Lag

Measure the time between:

```text
Experience
    ↓
Consolidation
    ↓
Available Memory
```

The target is:

```text
< 1 hour
```

for production configurations where near-current memory is required.

---

# 16. Generation Quality

Measure:

* Answer accuracy
* Faithfulness
* Groundedness
* Hallucination rate
* Contradiction rate

For grounded tasks:

```text
Answer
  │
  ▼
Supporting Evidence
```

Every factual claim should be traceable to retrieved evidence where required.

---

# 17. Reliability Testing

The benchmark must include failure scenarios.

Test:

* Vector database unavailable
* Graph database unavailable
* PostgreSQL unavailable
* Redis unavailable
* LLM timeout
* Embedding API timeout
* Tool failure
* Duplicate memories
* Conflicting memories
* Stale memories
* Malformed memories
* Empty retrieval
* Extremely large context
* Unauthorized namespace access

Measure:

```text
Recovery Rate
Graceful Degradation
Error Rate
Data Integrity
```

---

# 18. Security Evaluation

Test whether CAML prevents context leakage.

Example:

```text
Tenant A
   │
   └── Memory A

Tenant B
   │
   └── Memory B
```

Queries from Tenant A must never retrieve Memory B.

Test:

* Tenant isolation
* Namespace isolation
* RBAC
* Memory deletion
* Audit logging
* Access-control enforcement

---

# 19. Statistical Methodology

LLM systems are stochastic.

Therefore, each configuration should be executed multiple times.

Recommended:

```text
500 tasks
×
3–5 runs
```

For each metric calculate:

* Mean
* Median
* Standard deviation
* Confidence interval

For paired experiments:

```text
Baseline Task 001
        ↕
CAML Task 001

Baseline Task 002
        ↕
CAML Task 002
```

This produces a fair comparison because both systems solve identical tasks.

---

# 20. Benchmark Execution Pipeline

```text
                    Benchmark Dataset
                           │
                           ▼
                    Task Normalizer
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
         Baseline Agent             CAML Agent
              │                         │
              ▼                         ▼
         Run Experiment            Run Experiment
              │                         │
              └────────────┬────────────┘
                           ▼
                    Metrics Collector
                           │
                           ▼
                    Evaluation Engine
                           │
                           ▼
                 Statistical Analysis
                           │
                           ▼
                    Comparison Report
```

---

# 21. Observability

Every benchmark execution should generate a trace.

Example:

```json
{
  "task_id": "047",
  "configuration": "caml_full",
  "latency_ms": 842,
  "input_tokens": 2140,
  "output_tokens": 580,
  "retrieval_candidates": 20,
  "final_context_items": 6,
  "tool_calls": 2,
  "llm_calls": 2,
  "task_success": true,
  "faithfulness": 0.97,
  "cost": 0.009
}
```

This allows us to investigate **why** CAML improved or degraded a task rather than only looking at aggregate numbers.

---

# 22. Expected Results

We must **not assume that CAML will outperform the baseline**.

The experiment should test the hypothesis:

> CAML improves context-heavy agent performance while maintaining or reducing overall token consumption and operational cost.

Possible outcomes:

### Outcome A

```text
Quality ↑
Cost ↓
Latency ↓
```

Strong evidence for CAML.

### Outcome B

```text
Quality ↑
Cost ↑
Latency ↑
```

CAML provides quality benefits but requires optimization.

### Outcome C

```text
Quality ≈
Cost ↑
Latency ↑
```

CAML may not justify its complexity.

### Outcome D

```text
Quality ↓
```

The architecture or retrieval strategy requires redesign.

All outcomes are valid research results.

---

# 23. Acceptance Criteria

Before declaring CAML successful, establish thresholds **before running the final benchmark**.

For example:

```text
Task Success       ≥ +5 percentage points
Retrieval Precision ≥ +10%
Faithfulness        ≥ +5%
Token Consumption   ≤ -20%
Cost/Success        ≤ -10%
p95 Latency         ≤ +10%
```

These are **proposed experimental thresholds, not established results**. They should be finalized before the final experiment.

---

# 24. Final Comparison

The final report should contain a table such as:

| Metric        | Baseline | Full CAML |   Δ |
| ------------- | -------: | --------: | --: |
| Task Success  |      TBD |       TBD | TBD |
| Precision@K   |      TBD |       TBD | TBD |
| Recall@K      |      TBD |       TBD | TBD |
| Faithfulness  |      TBD |       TBD | TBD |
| Hallucination |      TBD |       TBD | TBD |
| Input Tokens  |      TBD |       TBD | TBD |
| Total Cost    |      TBD |       TBD | TBD |
| p50 Latency   |      TBD |       TBD | TBD |
| p95 Latency   |      TBD |       TBD | TBD |
| Tool Calls    |      TBD |       TBD | TBD |
| Memory Recall |      TBD |       TBD | TBD |

**No performance numbers should be inserted until the experiments are actually executed.**

---

# 25. Research Conclusion

The benchmark is designed to answer three questions:

### Question 1

**Does CAML improve agent quality?**

Measured through:

* Task success
* Accuracy
* Faithfulness
* Hallucination
* Memory recall

### Question 2

**Does CAML improve context efficiency?**

Measured through:

* Retrieval precision
* Context relevance
* Token consumption
* Compression
* Redundancy

### Question 3

**Is CAML worth its architectural overhead?**

Measured through:

* Total cost
* End-to-end latency
* Throughput
* Infrastructure overhead
* Reliability

The final claim should therefore not simply be:

> **"CAML is better than RAG."**

Instead, the scientifically defensible conclusion should be:

> **"Under the defined benchmark conditions, CAML [does/does not] provide a statistically significant improvement in agent task performance and context efficiency relative to the baseline architecture, with [acceptable/unacceptable] latency and cost overhead."**

That gives us a **proper experimental protocol**, rather than a benchmark designed to prove CAML is good. This distinction will be important when you eventually present the project as a research/engineering contribution.
