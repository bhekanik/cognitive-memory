# Changelog

## [0.5.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-v0.4.0...cognitive-memory-v0.5.0) (2026-05-08)

### Empirical default tuning

Three default values changed to match Python SDK v0.5.0. All three
grounded in the cognitive-memory-benchmarks Phase 0g→5 tuning
campaign (LTI-Bench OFAT + Optuna + LoCoMo head-to-head):

* **`associativeBoost`: 0.03 → 0.05**. Phase 1 OFAT (n=15) found
  0.03 was the WORST value tested across [0.01, 0.10]. Phase 5
  LoCoMo full benchmark confirmed +1.87pp F1 / +2.73pp LLM
  accuracy at the new defaults vs paper-faithful (1540 questions,
  gpt-4o-mini answer + gpt-4o-2024-08-06 judge, full mem0 prompt
  stack).
* **`decayRates.semantic`: 120 → 240** (days). Phase 1 OFAT swept
  [30, 60, 120, 180, 240]; 240 was the maximum (+1.4pp F1).
  Phase 2 Optuna confirmed any value in [200, 370] is
  statistically equivalent.
* **`coreSessionThreshold`: 3 → 2**. Phase 2 Optuna joint search
  (50 trials) showed cst=2 lands in the high-fitness cluster 91%
  of trials vs cst=3's 67% (n=23 vs n=12). Phase 1 OFAT had all
  values flat at default; the cst=3 underperformance only
  surfaces in joint search with the other tuned dims.

Other Tier 1+2 parameters unchanged.

See `cognitive-memory-benchmarks/docs/milestones/campaign-summary-2026-05.md`
for the full per-phase write-up.

### Tests

`__tests__/CognitiveMemory.test.ts` consolidation tests
recalibrated: under the longer v0.5 β_semantic, memories need
≥300d aging (was ~150d at paper β=120) to fall below the 0.20
consolidation threshold. 88/88 TS tests pass.

### Migration

Users who explicitly set these three params are unaffected. Users
on default config get the new behavior on upgrade — no API
changes, no breaking changes. To restore paper-faithful defaults:

```typescript
new CognitiveMemory({
  // ...
  config: {
    associativeBoost: 0.03,
    coreSessionThreshold: 3,
    decayRates: { episodic: 45, semantic: 120, procedural: Infinity, core: 120 },
  },
});
```

## [0.4.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-v0.3.0...cognitive-memory-v0.4.0) (2026-03-12)


### Features

* add BM25 lexical search to adapters and export v6 types ([3e0c6f4](https://github.com/planetaryescape/cognitive-memory/commit/3e0c6f49bb68d2d2923089376cef3b4f14140c6b))
* add extraction modes, comprehensive config docs, TypeScript SDK parity ([05a0897](https://github.com/planetaryescape/cognitive-memory/commit/05a0897914aef41089fba0875e9e19c03bdaa4b5))
* add v6 data model — semantic types, validity metadata, instrumentation types ([89cf2f9](https://github.com/planetaryescape/cognitive-memory/commit/89cf2f96e06efebe950f9c7c5cb1d5c594069ad7))
* add v6 retrieval pipeline — power-law decay, hybrid search, validity filtering, graph expansion, rerank, instrumentation ([34e1bc6](https://github.com/planetaryescape/cognitive-memory/commit/34e1bc6648b0a54d363a94381a4629d5de8e903a))
* monorepo with Python + TypeScript SDKs and docs ([736f112](https://github.com/planetaryescape/cognitive-memory/commit/736f112a3f0191f0f227110a0ad70b1a1928c6d2))
* update extraction prompts for semantic types and add LLM reranking ([13a8dcf](https://github.com/planetaryescape/cognitive-memory/commit/13a8dcf3be87420425f3ff336f84c04b14e1c29b))
* wire v6 features through CognitiveMemory public API ([7b7c15b](https://github.com/planetaryescape/cognitive-memory/commit/7b7c15b4fb4d4b6cc2e0cff0c85ea5900163c354))


### Bug Fixes

* address code review feedback ([f875100](https://github.com/planetaryescape/cognitive-memory/commit/f8751009b32c9fc2d769b6e8ec941e3a8c6cc8e6))
* persist memory mutations for non-InMemory adapters ([a698f45](https://github.com/planetaryescape/cognitive-memory/commit/a698f456ddb5205c3d004e07de318418ac8ca691))

## [0.3.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-v0.2.0...cognitive-memory-v0.3.0) (2026-03-12)


### Features

* add BM25 lexical search to adapters and export v6 types ([3e0c6f4](https://github.com/planetaryescape/cognitive-memory/commit/3e0c6f49bb68d2d2923089376cef3b4f14140c6b))
* add extraction modes, comprehensive config docs, TypeScript SDK parity ([05a0897](https://github.com/planetaryescape/cognitive-memory/commit/05a0897914aef41089fba0875e9e19c03bdaa4b5))
* add v6 data model — semantic types, validity metadata, instrumentation types ([89cf2f9](https://github.com/planetaryescape/cognitive-memory/commit/89cf2f96e06efebe950f9c7c5cb1d5c594069ad7))
* add v6 retrieval pipeline — power-law decay, hybrid search, validity filtering, graph expansion, rerank, instrumentation ([34e1bc6](https://github.com/planetaryescape/cognitive-memory/commit/34e1bc6648b0a54d363a94381a4629d5de8e903a))
* monorepo with Python + TypeScript SDKs and docs ([736f112](https://github.com/planetaryescape/cognitive-memory/commit/736f112a3f0191f0f227110a0ad70b1a1928c6d2))
* update extraction prompts for semantic types and add LLM reranking ([13a8dcf](https://github.com/planetaryescape/cognitive-memory/commit/13a8dcf3be87420425f3ff336f84c04b14e1c29b))
* wire v6 features through CognitiveMemory public API ([7b7c15b](https://github.com/planetaryescape/cognitive-memory/commit/7b7c15b4fb4d4b6cc2e0cff0c85ea5900163c354))


### Bug Fixes

* address code review feedback ([f875100](https://github.com/planetaryescape/cognitive-memory/commit/f8751009b32c9fc2d769b6e8ec941e3a8c6cc8e6))
* persist memory mutations for non-InMemory adapters ([a698f45](https://github.com/planetaryescape/cognitive-memory/commit/a698f456ddb5205c3d004e07de318418ac8ca691))
