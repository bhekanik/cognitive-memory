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

    # Retrieval scoring
    retrieval_score_exponent: float = 0.3  # alpha in score = sim * R^alpha

    # Ingestion behavior
    run_maintenance_during_ingestion: bool = True  # set False for batch benchmarks

    # Extraction
    extraction_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    custom_extraction_instructions: Optional[str] = None


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
    combined_score: float   # relevance * retention
    is_associative: bool = False  # came via association, not direct match
    via_deep_recall: bool = False
