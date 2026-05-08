"""
Types, configuration, and data structures for cognitive-memory.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MemoryCategory(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CORE = "core"


# Base decay rates (days) by category.
#
# Paper §3.2 Table 2 (original): episodic=45, semantic=120,
# procedural=inf, core=120.
#
# v0.5 update — semantic raised from 120 → 240 based on
# `cognitive-memory-benchmarks` Phase 1 (sensitivity sweep at n=15)
# and Phase 2 (Optuna, 50 trials). Phase 1 OFAT found
# semantic=240 hits f1=0.703 vs default's 0.689 (+1.4pp); Phase 2
# confirmed any value in [200, 370] is statistically equivalent.
# Episodic stays at paper 45 (Phase 1 found shorter ≤30d also fine,
# but 45d default is squarely in the working range; 90d+ hurts).
# Procedural stays at ∞ (no decay, paper-faithful). Core stays at
# 120 (no Phase 1/2 evidence to change).
BASE_DECAY_RATES: dict[MemoryCategory, float] = {
    MemoryCategory.EPISODIC: 45.0,
    MemoryCategory.SEMANTIC: 240.0,  # v0.5: tuned from paper's 120
    MemoryCategory.PROCEDURAL: float("inf"),
    MemoryCategory.CORE: 120.0,
}

# Decay floors - Equation 2 in the paper
DECAY_FLOORS: dict[str, float] = {
    "core": 0.60,
    "regular": 0.02,
}


@dataclass
class CognitiveMemoryConfig:
    """All tunable parameters from the paper, centralised."""

    # Decay
    faint_threshold: float = 0.15  # memories below this are "faint"

    # Retrieval boosting (Section 3.5)
    direct_boost: float = 0.1
    # v0.5: raised from 0.03 to 0.05. Phase 1 OFAT (n=15) found 0.03
    # was the WORST value tested; values 0.05/0.07/0.10 all hit
    # f1=0.685 vs default's 0.664 (+2pp). Phase 2 Optuna sweep
    # converged on assoc ∈ [0.05, 0.08] in the high-cluster trials.
    # Strongest empirical signal in the whole tuning campaign —
    # /docs/milestones/phase-1-sensitivity-analysis.md in benchmarks.
    associative_boost: float = 0.05
    max_spaced_rep_multiplier: float = 2.0
    spaced_rep_interval_days: float = 7.0

    # Core promotion thresholds (Section 3.4)
    core_access_threshold: int = 10
    core_stability_threshold: float = 0.85
    # v0.5: lowered from 3 to 2. Phase 2 Optuna joint search showed
    # cst=1 (93%) and cst=2 (91%) tied at landing in the high
    # fitness cluster; cst=3 trailed at 67% (n=12). Phase 1 OFAT had
    # all three flat at default, but joint search with the other
    # tuned dims surfaced the cst=3 underperformance. Picking cst=2
    # over cst=1 because the benchmarks adapter has been pinning
    # cst=2 already (matches existing benchmark behaviour) and
    # because cst=2 has more samples (23 vs 15) backing the rate.
    core_session_threshold: int = 2

    # Associations (Section 3.6)
    association_strengthen_amount: float = 0.1
    association_retrieval_threshold: float = 0.3
    association_decay_constant_days: float = 90.0

    # Consolidation (Section 3.7)
    consolidation_retention_threshold: float = 0.20
    consolidation_group_size: int = 5
    consolidation_similarity_threshold: float = 0.70

    # Tiered storage (Section 3.8)
    cold_migration_days: int = 7  # consecutive days at floor before migration
    cold_storage_ttl_days: int = 180  # days in cold before permanent deletion

    # Deep recall (Section 3.8)
    deep_recall_penalty: float = 0.5

    # Hybrid retrieval (v6)
    hybrid_search: bool = False
    k_sparse: int = 30  # top-k for BM25 lexical search

    # Validity filtering (v6)
    filter_expired_transients: bool = True
    include_expired_in_deep_recall: bool = True

    # Graph expansion / bridge discovery (v6)
    graph_expansion_hops: int = 1  # 0=disabled, 1 or 2
    bridge_discovery: bool = False
    max_bridge_paths: int = 3
    min_bridge_edge_weight: float = 0.3

    # LLM rerank (v6)
    rerank_enabled: bool = False
    k_rerank: int = 10  # top candidates to send to LLM for reranking
    rerank_factor: int = 1  # multiplier for candidate pool before reranking
    rerank_model: Optional[str] = None  # defaults to extraction_model if None

    # Decay model
    decay_model: str = "exponential"  # "exponential" | "power"
    power_decay_gamma: float = 1.4427  # 1/ln(2), calibrated match point
    # Per-category base decay rate (β_c, days). Default mirrors the
    # module constant `BASE_DECAY_RATES` (paper §3.2 Table 2). Made a
    # config field in v0.5 so benchmarks/tuning can override per-trial
    # without monkey-patching. Default factory copies the constant so
    # mutating one config doesn't affect siblings.
    base_decay_rates: dict = field(default_factory=lambda: dict(BASE_DECAY_RATES))

    # Retrieval scoring
    retrieval_score_exponent: float = 0.3  # alpha in score = sim * R^alpha

    # Ingestion behavior
    run_maintenance_during_ingestion: bool = True  # set False for batch benchmarks

    # Extraction
    extraction_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    custom_extraction_instructions: Optional[str] = None
    extraction_mode: str = "semantic"  # "raw" | "semantic" | "hybrid"

    def __post_init__(self):
        # Merge user-specified base_decay_rates over defaults so a
        # caller can override one category without losing the others.
        # Also coerce string keys ("semantic") to MemoryCategory enum
        # so JSON-loaded configs work.
        merged = dict(BASE_DECAY_RATES)
        for k, v in self.base_decay_rates.items():
            if isinstance(k, str):
                k = MemoryCategory(k)
            merged[k] = v
        self.base_decay_rates = merged


@dataclass
class Association:
    """Bidirectional link between two memories."""
    target_id: str
    weight: float = 0.3
    last_co_retrieval: Optional[datetime] = None
    created_at: Optional[datetime] = None


@dataclass
class Memory:
    """Core memory object - Table 1 in the paper."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"  # owner; enables multi-tenant scoping
    content: str = ""
    category: MemoryCategory = MemoryCategory.EPISODIC
    importance: float = 0.5  # [0, 1] LLM-assessed at extraction
    stability: float = 0.1  # [0, 1] resistance to decay, increases with use
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    # Embedding
    embedding: Optional[list[float]] = None

    # Associations
    associations: dict[str, Association] = field(default_factory=dict)

    # Session tracking (for core promotion)
    session_ids: set[str] = field(default_factory=set)

    # Tiered storage
    is_cold: bool = False
    cold_since: Optional[datetime] = None
    days_at_floor: int = 0

    # Consolidation
    is_superseded: bool = False
    superseded_by: Optional[str] = None

    # Conflict
    contradicted_by: Optional[str] = None

    # Summary stub (for TTL-deleted memories)
    is_stub: bool = False

    # v6: Semantic type classification (orthogonal to category)
    memory_type: str = "other"  # "fact" | "preference" | "plan" | "transient_state" | "other"
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    source_turn_ids: list[str] = field(default_factory=list)

    @property
    def floor(self) -> float:
        if self.category == MemoryCategory.CORE:
            return DECAY_FLOORS["core"]
        return DECAY_FLOORS["regular"]

    @property
    def base_decay_rate(self) -> float:
        return BASE_DECAY_RATES[self.category]

    @property
    def is_faint(self) -> bool:
        return not self.is_core_memory and self.stability < 0.3

    @property
    def is_core_memory(self) -> bool:
        return self.category == MemoryCategory.CORE


@dataclass
class SearchResult:
    """A scored memory from retrieval."""
    memory: Memory
    relevance_score: float  # cosine similarity
    retention_score: float  # R(m) from decay
    combined_score: float   # relevance * retention^alpha
    is_associative: bool = False  # came via association, not direct match
    via_deep_recall: bool = False
    evidence_chains: list[list[str]] = field(default_factory=list)  # v6: bridge paths (memory ID chains)


@dataclass
class StageTrace:
    """Timing and stats for a single pipeline stage."""
    name: str = ""
    wall_ms: float = 0.0
    candidate_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchTrace:
    """Per-query instrumentation trace."""
    total_wall_ms: float = 0.0
    total_tokens: int = 0
    stages: dict[str, StageTrace] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Full search response with results, evidence chains, and optional trace."""
    results: list[SearchResult] = field(default_factory=list)
    evidence_chains: list[list[str]] = field(default_factory=list)
    trace: Optional[SearchTrace] = None

    def __len__(self) -> int:
        """Backward-compatible list length for callers that treated search() as results."""
        return len(self.results)

    def __iter__(self):
        """Backward-compatible iteration over search results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Backward-compatible indexing into search results."""
        return self.results[index]
