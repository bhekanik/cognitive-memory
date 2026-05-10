# Changelog

## [0.5.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-python-v0.4.0...cognitive-memory-python-v0.5.0) (2026-05-08)

### Features

* **config**: `base_decay_rates` is now a `CognitiveMemoryConfig` field (not a module constant). Lets benchmarks/tuning override per-category β_c without monkey-patching. JSON-loaded configs work via string-key coercion in `__post_init__`.

### Empirical default tuning

Three default values changed based on a systematic tuning campaign in
[`cognitive-memory-benchmarks`](https://github.com/planetaryescape/cognitive-memory-benchmarks)
(Phase 0g→5, ~$245 spend, ~28h compute):

* **`associative_boost`: 0.03 → 0.05**. Phase 1 OFAT (n=15) found 0.03
  was the WORST value tested across [0.01, 0.10]. Phase 5 LoCoMo
  full benchmark confirmed +1.87pp F1 / +2.73pp LLM accuracy at the
  new defaults vs paper-faithful (1540 questions, gpt-4o-mini answer
  + gpt-4o-2024-08-06 judge, full mem0 prompt stack).
* **`base_decay_rates.semantic`: 120 → 240** (days). Phase 1 OFAT
  swept [30, 60, 120, 180, 240]; 240 was the maximum (+1.4pp f1).
  Phase 2 Optuna confirmed any value in [200, 370] is statistically
  equivalent; picked 240 as the closest improvement to paper's 120.
* **`core_session_threshold`: 3 → 2**. Phase 2 Optuna joint search
  (50 trials) showed cst=2 lands in the high-fitness cluster 91% of
  trials vs cst=3's 67% (n=23 vs n=12). Phase 1 OFAT had all values
  flat at default; the cst=3 underperformance only surfaces in joint
  search with the other tuned dims.

Other Tier 1+2 parameters unchanged — Phase 1+2 didn't surface
evidence to move them. 6 of 10 swept parameters had no measurable
signal.

### Validation chain

| phase | what | finding |
|---|---|---|
| 1 | OFAT sensitivity (n=15) | assoc=0.03 worst, β_sem=240 max |
| 2 | Optuna joint search (n=50) | cst=3 trails; top-5 within noise |
| 2.5 | per-question variance | 3 of 42 LTI-Bench Q cause bimodality |
| 2.5b | top-K confirm @n=5 | trial closest to v0.5 defaults won |
| 3 | LoCoMo conv0 cross-check | rank stability holds, no overfitting |
| 4 | LoCoMo conv0 head-to-head | v0.5 +2.92pp F1 |
| **5** | **full LoCoMo (n=1540)** | **v0.5 +1.87pp F1, +2.73pp LLM acc** |
| 7 | LongMemEval-S | attempted; OpenAI billing-cap blocked at 30%; partial inconclusive |

See `cognitive-memory-benchmarks/docs/milestones/` for full per-phase
write-ups.

### Tests

3 new value-lock tests in `tests/test_config.py`
(`test_associative_boost_default_is_v0_5_tuned`,
`test_core_session_threshold_default_is_v0_5_tuned`,
`test_base_decay_rates_semantic_default_is_v0_5_tuned`) so
accidental reverts to paper Table 2 defaults fail loudly. 8/8 SDK
config tests pass.

### Migration

Users who explicitly set these three params are unaffected. Users
on default config get the new behavior on upgrade — no API changes,
no breaking changes. To restore paper-faithful defaults, set
explicitly in `CognitiveMemoryConfig(...)`:

```python
CognitiveMemoryConfig(
    associative_boost=0.03,
    core_session_threshold=3,
    base_decay_rates={"semantic": 120.0},  # other categories default
)
```

## [0.4.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-python-v0.3.0...cognitive-memory-python-v0.4.0) (2026-03-12)


### Features

* add BM25 lexical search to adapters and export v6 types ([3e0c6f4](https://github.com/planetaryescape/cognitive-memory/commit/3e0c6f49bb68d2d2923089376cef3b4f14140c6b))
* add extraction modes, comprehensive config docs, TypeScript SDK parity ([05a0897](https://github.com/planetaryescape/cognitive-memory/commit/05a0897914aef41089fba0875e9e19c03bdaa4b5))
* add v6 data model — semantic types, validity metadata, instrumentation types ([89cf2f9](https://github.com/planetaryescape/cognitive-memory/commit/89cf2f96e06efebe950f9c7c5cb1d5c594069ad7))
* add v6 retrieval pipeline — power-law decay, hybrid search, validity filtering, graph expansion, rerank, instrumentation ([34e1bc6](https://github.com/planetaryescape/cognitive-memory/commit/34e1bc6648b0a54d363a94381a4629d5de8e903a))
* monorepo with Python + TypeScript SDKs and docs ([736f112](https://github.com/planetaryescape/cognitive-memory/commit/736f112a3f0191f0f227110a0ad70b1a1928c6d2))
* update extraction prompts for semantic types and add LLM reranking ([13a8dcf](https://github.com/planetaryescape/cognitive-memory/commit/13a8dcf3be87420425f3ff336f84c04b14e1c29b))
* wire v6 features through CognitiveMemory public API ([7b7c15b](https://github.com/planetaryescape/cognitive-memory/commit/7b7c15b4fb4d4b6cc2e0cff0c85ea5900163c354))


### Bug Fixes

* persist memory mutations for non-InMemory adapters ([a698f45](https://github.com/planetaryescape/cognitive-memory/commit/a698f456ddb5205c3d004e07de318418ac8ca691))


### Documentation

* fix quickstart examples and SDK READMEs for onboarding ([68d7f0e](https://github.com/planetaryescape/cognitive-memory/commit/68d7f0e0997ef8594208323b03a7dd58a302ec96))

## [0.3.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-python-v0.2.0...cognitive-memory-python-v0.3.0) (2026-03-12)


### Features

* add BM25 lexical search to adapters and export v6 types ([3e0c6f4](https://github.com/planetaryescape/cognitive-memory/commit/3e0c6f49bb68d2d2923089376cef3b4f14140c6b))
* add extraction modes, comprehensive config docs, TypeScript SDK parity ([05a0897](https://github.com/planetaryescape/cognitive-memory/commit/05a0897914aef41089fba0875e9e19c03bdaa4b5))
* add v6 data model — semantic types, validity metadata, instrumentation types ([89cf2f9](https://github.com/planetaryescape/cognitive-memory/commit/89cf2f96e06efebe950f9c7c5cb1d5c594069ad7))
* add v6 retrieval pipeline — power-law decay, hybrid search, validity filtering, graph expansion, rerank, instrumentation ([34e1bc6](https://github.com/planetaryescape/cognitive-memory/commit/34e1bc6648b0a54d363a94381a4629d5de8e903a))
* monorepo with Python + TypeScript SDKs and docs ([736f112](https://github.com/planetaryescape/cognitive-memory/commit/736f112a3f0191f0f227110a0ad70b1a1928c6d2))
* update extraction prompts for semantic types and add LLM reranking ([13a8dcf](https://github.com/planetaryescape/cognitive-memory/commit/13a8dcf3be87420425f3ff336f84c04b14e1c29b))
* wire v6 features through CognitiveMemory public API ([7b7c15b](https://github.com/planetaryescape/cognitive-memory/commit/7b7c15b4fb4d4b6cc2e0cff0c85ea5900163c354))


### Bug Fixes

* persist memory mutations for non-InMemory adapters ([a698f45](https://github.com/planetaryescape/cognitive-memory/commit/a698f456ddb5205c3d004e07de318418ac8ca691))


### Documentation

* fix quickstart examples and SDK READMEs for onboarding ([68d7f0e](https://github.com/planetaryescape/cognitive-memory/commit/68d7f0e0997ef8594208323b03a7dd58a302ec96))
