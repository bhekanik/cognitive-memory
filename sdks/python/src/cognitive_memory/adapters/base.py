"""
Abstract base class for memory storage adapters.

All adapters are async-first. Use SyncCognitiveMemory for sync wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, TypeVar

from ..types import Memory

T = TypeVar("T")


class MemoryAdapter(ABC):
    """
    Abstract adapter interface for cognitive-memory storage backends.

    Implementations must provide async methods for CRUD, vector search,
    tiered storage migration, associative links, and consolidation.
    """

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    async def create(self, memory: Memory) -> None:
        """Store a new memory."""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID from any tier."""
        ...

    @abstractmethod
    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by ID."""
        ...

    @abstractmethod
    async def update(self, memory: Memory) -> None:
        """Update an existing memory in place."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Hard delete from any tier."""
        ...

    @abstractmethod
    async def delete_batch(self, memory_ids: list[str]) -> None:
        """Hard delete multiple memories."""
        ...

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_superseded: bool = False,
    ) -> list[tuple[Memory, float]]:
        """
        Search by cosine similarity.

        Returns list of (memory, similarity_score) sorted descending.
        Superseded memories are excluded by default.
        """
        ...

    # ------------------------------------------------------------------
    # Tiered storage
    # ------------------------------------------------------------------

    @abstractmethod
    async def migrate_to_cold(self, memory_id: str, cold_since: "datetime") -> None:
        """Move a memory from hot to cold store."""
        ...

    @abstractmethod
    async def migrate_to_hot(self, memory_id: str) -> None:
        """Reactivate a cold memory back to hot store."""
        ...

    @abstractmethod
    async def convert_to_stub(self, memory_id: str, stub_content: str) -> None:
        """Replace a memory with a lightweight stub (TTL expiry)."""
        ...

    # ------------------------------------------------------------------
    # Associative links
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_or_strengthen_link(
        self, source_id: str, target_id: str, weight: float,
    ) -> None:
        """Create or strengthen a bidirectional association."""
        ...

    @abstractmethod
    async def get_linked_memories(
        self, memory_id: str, min_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        """Get memories linked to a given memory above threshold."""
        ...

    @abstractmethod
    async def delete_link(self, source_id: str, target_id: str) -> None:
        """Delete an association between two memories."""
        ...

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_fading(
        self, threshold: float, exclude_core: bool = True,
    ) -> list[Memory]:
        """Find non-superseded memories with retention below threshold."""
        ...

    @abstractmethod
    async def find_stable(
        self, min_stability: float, min_access_count: int,
    ) -> list[Memory]:
        """Find highly stable, frequently accessed memories."""
        ...

    @abstractmethod
    async def mark_superseded(
        self, memory_ids: list[str], summary_id: str,
    ) -> None:
        """Mark memories as superseded by a consolidation summary."""
        ...

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    @abstractmethod
    async def all_active(self) -> list[Memory]:
        """All memories in hot + cold (not stubs)."""
        ...

    @abstractmethod
    async def all_hot(self) -> list[Memory]:
        """All memories in the hot store."""
        ...

    @abstractmethod
    async def all_cold(self) -> list[Memory]:
        """All memories in the cold store."""
        ...

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    @abstractmethod
    async def hot_count(self) -> int:
        ...

    @abstractmethod
    async def cold_count(self) -> int:
        ...

    @abstractmethod
    async def stub_count(self) -> int:
        ...

    @abstractmethod
    async def total_count(self) -> int:
        ...

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def batch_update(self, memories: list[Memory]) -> None:
        """Update multiple memories at once."""
        ...

    @abstractmethod
    async def update_retention_scores(self, updates: dict[str, float]) -> None:
        """Bulk update retention scores: {memory_id: new_retention}."""
        ...

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @abstractmethod
    async def transaction(self, callback: Callable[["MemoryAdapter"], T]) -> T:
        """Execute callback within a transaction (adapter-specific)."""
        ...

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data."""
        ...


# Avoid circular import — just for type hints
from datetime import datetime  # noqa: E402
