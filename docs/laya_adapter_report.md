# Laya Adapter Implementation — Final Report

Date: 2026-10-06
Scope: LayaDecisionProvider adapter milestone (Decision → Retrieval integration is a separate phase, not implemented).

## Files added

- `packages/decisions/exceptions.py` — `DecisionProviderError` (base) + `LayaDecisionError`
- `packages/decisions/laya.py` — `LayaDecisionProvider` adapter
- `tests/decisions/test_laya.py` — 15 focused tests with a fake router
- `scripts/smoke_laya.py` — real-Router smoke test (not part of CI)
- `scripts/evaluate_laya.py` — Laya benchmark on the same 10 evaluation cases

## Files modified

- `packages/decisions/__init__.py` — added `LayaDecisionError` / `DecisionProviderError` exports; all prior exports preserved
- `tests/decisions/test_orchestrator.py` — added `test_laya_failure_falls_back_to_rules`

## Architecture

```
Agent → Engram → DecisionProvider
  ├── RuleBasedDecisionProvider
  └── LayaDecisionProvider → RetrievalPolicy → HybridRetrievalService (next phase)
```

Laya decides; Engram executes. The adapter only produces a `RetrievalPolicy` — no retrieval, memory, DB, DTOC, or tool access. Laya stays optional: the adapter never imports the `laya` package; the `Router` is injected, so importing `packages.decisions` works without Laya installed.

## Laya API used (verified against installed 0.3.22)

- `Router.decide(state, schema=..., return_details=True)` → `DecisionResult` dataclass
- `state` is a string (query + namespace + top_k); schema is a JSON-schema dict with an `enum`, which Laya converts into a typed `choice` question
- `DecisionResult.values` → `{"strategy": ...}`
- `answer_confidence` (max(p) per field — Laya's documented decision quantity) preferred; `confidence` (normalized entropy) as fallback; `routing["model"]` → checkpoint name
- `decide` is synchronous → wrapped in `asyncio.to_thread`

## Decision mapping

- `vector` → `[RetrievalStrategyName.VECTOR]`
- `sparse` → `[RetrievalStrategyName.SPARSE]`
- `hybrid` → `[RetrievalStrategyName.VECTOR, RetrievalStrategyName.SPARSE]`
- fusion always `FusionStrategyName.RRF`; `top_k` from request
- unknown strategy → `LayaDecisionError` (no silent conversion)

## Tests

- Focused tests: 15 passed (vector, sparse, hybrid, top_k propagation, namespace metadata, provider name, latency, model from routing, model fallback, confidence ×3, invalid strategy, provider failure, deterministic reasons)
- Pyright: 0 errors, 0 warnings on `packages/decisions tests/decisions`
- Full suite: 221 passed (baseline 209 → +15 laya, +1 fallback, −4 replaced draft tests)

## Baseline (control, unchanged)

- Rules accuracy: 100%
- Rules latency: 0.031 ms avg / 0.039 ms P95
- Rules confidence: 80.50% avg
- False high-confidence: 0

## Laya (real model, same 10 cases)

- Accuracy: 70% (7/10)
- Latency: 27.2 s avg (first-call model download/load dominates; P95 590 ms)
- Confidence: 45.26% avg
- False high-confidence: 3

Laya is not better than Rules on this dataset; measurements are reported as-is per the experimental rule (no tuning of Rules, no dataset changes, no prompt changes after seeing results).

## Fallback

`DecisionOrchestrator` catches `LayaDecisionError` and falls back to `RuleBasedDecisionProvider`; the adapter itself never falls back. Test proves `provider == "rules"` and `metadata["fallback"] is True`.

## Next step

Decision → Retrieval integration evaluation (separate phase).