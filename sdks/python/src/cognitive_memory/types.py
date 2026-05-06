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


# Base decay rates (days) by category - Table 2 in the paper
BASE_DECAY_RATES: dict[MemoryCategory, float] = {
    MemoryCategory.EPISODIC: 45.0,
    MemoryCategory.SEMANTIC: 120.0,
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
    associative_boost: float = 0.03
    max_spaced_rep_multiplier: float = 2.0
    spaced_rep_interval_days: float = 7.0

    # Core promotion thresholds (Section 3.4)
    core_access_threshold: int = 10
    core_stability_threshold: float = 0.85
    core_session_threshold: int = 3

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
