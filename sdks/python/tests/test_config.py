"""Tests for CognitiveMemoryConfig — base_decay_rates plumbing (Phase 0).

Each test is one vertical slice. RED → 5-question gate → GREEN. See
`~/.claude/plans/now-create-a-plan-validated-yao.md` Stage 0a-sdk.
"""

from cognitive_memory import CognitiveMemoryConfig
from cognitive_memory.types import BASE_DECAY_RATES, MemoryCategory


def test_config_base_decay_rates_default_matches_module_constant():
    """Config's base_decay_rates field defaults to a copy of BASE_DECAY_RATES.

    Spec: paper §3.2 Table 2 — 45d episodic, 120d semantic, ∞ procedural,
    120d core. Value lives in `types.py:22`.
    """
    cfg = CognitiveMemoryConfig()
    assert cfg.base_decay_rates == BASE_DECAY_RATES
    # Must be a copy, not the same object — otherwise mutating one
    # config silently mutates the module constant + every other config.
    assert cfg.base_decay_rates is not BASE_DECAY_RATES


def test_config_base_decay_rates_construction_time_override_wins():
    """Passing base_decay_rates= at construction overrides the default.

    Tuning/benchmark trials need to flip per-category rates without
    touching the module constant. Other categories keep their default.
    """
    override = {MemoryCategory.SEMANTIC: 60.0}
    cfg = CognitiveMemoryConfig(base_decay_rates=override)
    assert cfg.base_decay_rates[MemoryCategory.SEMANTIC] == 60.0
    # Sibling categories untouched at default (45d episodic, ∞ procedural,
    # 120d core).
    assert cfg.base_decay_rates[MemoryCategory.EPISODIC] == 45.0
    assert cfg.base_decay_rates[MemoryCategory.CORE] == 120.0
    assert cfg.base_decay_rates[MemoryCategory.PROCEDURAL] == float("inf")


def test_config_base_decay_rates_string_keys_are_coerced():
    """JSON-loaded configs use string keys; coerce to MemoryCategory.

    `lti_bench.py --config X.json` reads JSON; JSON has no enum support.
    `{"semantic": 60.0}` must yield the same config as
    `{MemoryCategory.SEMANTIC: 60.0}`.
    """
    cfg = CognitiveMemoryConfig(base_decay_rates={"semantic": 60.0})
    assert cfg.base_decay_rates[MemoryCategory.SEMANTIC] == 60.0
    # All keys must be `MemoryCategory` enum instances, not raw strings.
    # (`MemoryCategory` is a str-subclass enum, so `"semantic" in
    # {MemoryCategory.SEMANTIC: ...}` returns True regardless — that's
    # not a useful assertion. Instead check the actual key types.)
    for key in cfg.base_decay_rates:
        assert isinstance(key, MemoryCategory), f"got {type(key)} key {key!r}"


def _engine(config):
    """Helper: build a CognitiveEngine with an in-memory adapter."""
    from cognitive_memory.adapters.memory import InMemoryAdapter
    from cognitive_memory.engine import CognitiveEngine

    return CognitiveEngine(adapter=InMemoryAdapter(), config=config)


def _aged_memory(category, age_days):
    """Helper: build a Memory `age_days` old with stability=0.5,
    importance=0.0 so retention is purely β/age driven."""
    from datetime import datetime, timedelta, timezone

    from cognitive_memory.types import Memory

    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    last = now - timedelta(days=age_days)
    mem = Memory(
        user_id="alice",
        content="x",
        category=category,
        importance=0.0,
        stability=0.5,
        last_accessed_at=last,
        created_at=last,
    )
    return mem, now


def test_compute_retention_uses_config_base_decay_rates():
    """The engine's compute_retention reads β from config, not the
    module constant. With β halved, retention at the same age drops
    faster (strict inequality)."""
    mem_default, now = _aged_memory(MemoryCategory.SEMANTIC, age_days=60)
    mem_fast, _ = _aged_memory(MemoryCategory.SEMANTIC, age_days=60)

    cfg_default = CognitiveMemoryConfig()  # semantic = 120d
    cfg_fast = CognitiveMemoryConfig(base_decay_rates={"semantic": 60.0})

    r_default = _engine(cfg_default).compute_retention(mem_default, now=now)
    r_fast = _engine(cfg_fast).compute_retention(mem_fast, now=now)

    # Halving β doubles the effective decay rate ⇒ retention at the
    # same age drops further. Strict inequality is the contract.
    assert r_fast < r_default, (
        f"override didn't take effect: r_default={r_default}, r_fast={r_fast}"
    )
    # Both should still be above the floor (regular = 0.02).
    assert r_fast > 0.02


def test_compute_retention_procedural_infinity_short_circuits_under_override():
    """Procedural memories must remain non-decaying even when the user
    overrides one category. β=∞ short-circuits to retention=1.0
    regardless of override path."""
    mem, now = _aged_memory(MemoryCategory.PROCEDURAL, age_days=365 * 5)

    # Override episodic only (irrelevant) — procedural stays ∞ default.
    cfg = CognitiveMemoryConfig(base_decay_rates={"episodic": 30.0})
    assert _engine(cfg).compute_retention(mem, now=now) == 1.0


# ---------------------------------------------------------------------------
# v0.5 tuned defaults — value lock so accidental regressions to paper-Table-2
# values fail a test instead of silently shipping. Empirical justification:
# `cognitive-memory-benchmarks` Phase 1 (sensitivity sweep, n=15) and Phase 2
# (Optuna 50 trials). See
# `cognitive-memory-benchmarks/docs/milestones/phase-2-optuna-tuning.md`.
# ---------------------------------------------------------------------------


def test_associative_boost_default_is_v0_5_tuned():
    """v0.5: 0.05 (was 0.03 in paper-faithful default). Phase 1 OFAT
    found 0.03 was the WORST value tested across [0.01, 0.10]; 0.05 hit
    sweep best at +2pp f1. Strongest empirical signal in the campaign.

    If this test fails, someone reverted associative_boost to 0.03 (or
    paper-Table-2-default) without updating tuning evidence — confirm
    the benchmarks Phase 1/2 results changed before changing this back.
    """
    cfg = CognitiveMemoryConfig()
    assert cfg.associative_boost == 0.05, (
        f"expected v0.5 tuned default 0.05, got {cfg.associative_boost}"
    )


def test_core_session_threshold_default_is_v0_5_tuned():
    """v0.5: 2 (was 3). Phase 2 Optuna joint search: cst=1 (93%) and
    cst=2 (91%) tied at landing in the high-fitness cluster; cst=3
    trailed at 67% (n=12). Phase 1 OFAT had all three flat — Phase 2
    surfaced the cst=3 underperformance in joint search."""
    cfg = CognitiveMemoryConfig()
    assert cfg.core_session_threshold == 2, (
        f"expected v0.5 tuned default 2, got {cfg.core_session_threshold}"
    )


def test_base_decay_rates_semantic_default_is_v0_5_tuned():
    """v0.5: semantic = 240d (was 120d in paper Table 2). Phase 1 OFAT
    swept [30, 60, 120, 180, 240]; 240 was the maximum (f1=0.703 vs
    default's 0.689 = +1.4pp). Phase 2 confirmed any value in
    [200, 370] is statistically equivalent. Other categories
    unchanged from paper Table 2."""
    cfg = CognitiveMemoryConfig()
    assert cfg.base_decay_rates[MemoryCategory.SEMANTIC] == 240.0, (
        f"expected v0.5 tuned semantic β = 240.0, got "
        f"{cfg.base_decay_rates[MemoryCategory.SEMANTIC]}"
    )
    # Other categories must remain paper-Table-2 — Phase 1/2 didn't
    # surface evidence to change them.
    assert cfg.base_decay_rates[MemoryCategory.EPISODIC] == 45.0
    assert cfg.base_decay_rates[MemoryCategory.CORE] == 120.0
    assert cfg.base_decay_rates[MemoryCategory.PROCEDURAL] == float("inf")
