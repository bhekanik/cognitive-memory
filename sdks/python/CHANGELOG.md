# Changelog

## [0.5.0](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-python-v0.4.0...cognitive-memory-python-v0.5.0) (2026-05-27)


### ⚠ BREAKING CHANGES

* **config:** Phase 6 — tune defaults from cognitive-memory-benchmarks

### Features

* **adapters:** RemoteAdapter for daemon-backed deployment ([e0bba5c](https://github.com/planetaryescape/cognitive-memory/commit/e0bba5c2fb8e06aa0b505ac33496a8bea94ad519))
* **config:** Phase 0a-sdk — base_decay_rates as a CognitiveMemoryConfig field ([abb32d4](https://github.com/planetaryescape/cognitive-memory/commit/abb32d40becfeb3f6f4c3c892c7ee4caaaad022b))
* **config:** Phase 6 — tune defaults from cognitive-memory-benchmarks ([707758d](https://github.com/planetaryescape/cognitive-memory/commit/707758dc55a02394c31cb273dd8f562c75276be1))
* **config:** v0.5.1 — decay_floors as a config field for the Phase 8 ablation ([7da1467](https://github.com/planetaryescape/cognitive-memory/commit/7da14676268ec98f522eabc82da3831d18f676b3))
* **core:** bring CognitiveMemory class to CLI parity ([8774e71](https://github.com/planetaryescape/cognitive-memory/commit/8774e71891c258c6d39f8bd50ccaed6861e415a0))
* **python:** temporal reconstruction behind default-off flags ([2322833](https://github.com/planetaryescape/cognitive-memory/commit/23228338dbb1d3d7a5b2d1b941d3480b70556173))
* **sdk:** v0.4.0 SDK behavioural parity ([91d4c7b](https://github.com/planetaryescape/cognitive-memory/commit/91d4c7becef109d08294d544daccdcd916437fbd))
* **ts:** align hot/cold/stub filtering and add rerankFactor knob ([7f85372](https://github.com/planetaryescape/cognitive-memory/commit/7f8537233ac94df7da11b2465195ae663c1d8bdf))


### Bug Fixes

* **extraction:** require valid_time.status and raw_time_expressions in extraction prompt ([ecfcd2e](https://github.com/planetaryescape/cognitive-memory/commit/ecfcd2e0061c78497f011a4de35c36c12b665bb3))
* **temporal:** tighten _is_temporal_query precision ([ee3015a](https://github.com/planetaryescape/cognitive-memory/commit/ee3015ad2ed9ff74be5cc9d3f0712c92c997d948))


### Documentation

* align spec to shipped code; add v0.4.0 migration guides ([449d127](https://github.com/planetaryescape/cognitive-memory/commit/449d127ed1e6040033305ebc5f328e3dd20e1e2d))
* **changelog:** v0.5.0 entry — empirical default tuning campaign ([bb7fcc5](https://github.com/planetaryescape/cognitive-memory/commit/bb7fcc55681bfcc581d9f51a09a38155e3914adf))
* README refresh — daemon mode, related repos, v0.4.0 highlights ([2b0bc7a](https://github.com/planetaryescape/cognitive-memory/commit/2b0bc7aa03b17b235919e1b93967bce6bf87590d))
* refresh benchmark numbers and docs URL in READMEs ([27c831c](https://github.com/planetaryescape/cognitive-memory/commit/27c831c6742b6293ab202c859ee1570d19db6f10))
* refresh public README and Astro site with v6 benchmark numbers ([1ecdf05](https://github.com/planetaryescape/cognitive-memory/commit/1ecdf0582cc9b9978a8a90aa6c728187f41ce8b1))

## [0.5.1](https://github.com/planetaryescape/cognitive-memory/compare/cognitive-memory-python-v0.5.0...cognitive-memory-python-v0.5.1) (2026-05-11)

### Features

* **config**: `decay_floors` is now a `CognitiveMemoryConfig` field (not just a module constant). Mirrors the v0.5 `base_decay_rates` change. Lets the cognitive-memory-benchmarks decay-floor ablation (Phase 8) override the floor mechanism without monkey-patching. Default factory copies `DECAY_FLOORS` so per-config mutations don't leak. `__post_init__` merges partial overrides over defaults so `{"core": 0.0}` keeps the regular floor at 0.02.

### Empirical finding (Phase 8)

The decay-floor ablation on LTI-Bench produced a NEGATIVE result:
setting both floors to 0 left `critical_fact_retention` unchanged at
100% and `decay_trivial` unchanged at 0.614 (n=3 sub-runs each arm).
The simplest version of the architectural claim ("floors keep
critical facts retrievable") is **falsified on LTI-Bench's 30-day
window**. The actual mechanism keeping critical facts retrievable on
this distribution is stability accumulation through repeated direct
retrieval combined with the relevance-driven scoring at α=0.3.

The architectural claim survives in weaker form: floors are designed
to matter at horizons where stability decays past the clamping
point. LTI-Bench's 30-day window doesn't reach there often enough.
A 90d/180d ablation is the cleanest test of the strong version of
the claim and is now Phase 8 future work.

See `cognitive-memory-benchmarks/docs/milestones/phase-8-decay-floor-ablation.md`
for full per-arm numbers + caveats. Paper §6.10 in
`cognitive-memory-benchmarks/paper/paper.tex` has the same finding
as a "Decay-floor ablation (negative result)" paragraph.

### Tests

3 new value-lock tests in `tests/test_config.py`:
- `test_decay_floors_default_matches_paper_table_2`
- `test_decay_floors_override_replaces_one_key_only`
- `test_compute_retention_reads_decay_floor_from_config`

11/11 SDK config tests pass (8 prior + 3 new).

### Migration

Additive only. Users who don't touch `decay_floors` see identical
behaviour. Users who want to ablate or experiment can now pass
`CognitiveMemoryConfig(decay_floors={"core": 0.0, "regular": 0.0})`
or any partial override.

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
