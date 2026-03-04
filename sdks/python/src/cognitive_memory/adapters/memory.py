"""
In-memory adapter for cognitive-memory.

Tiered storage with hot/cold/stub dicts and brute-force cosine similarity.
Zero dependencies, great for testing and single-process use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Callable, TypeVar

from ..types import Memory, MemoryCategory
from ..embeddings import cosine_similarity
from .base import MemoryAdapter

T = TypeVar("T")


class InMemoryAdapter(MemoryAdapter):
    """
    Dict-based tiered in-memory store.

    Hot: vector-indexed, supports similarity search.
    Cold: key-value only, accessed by ID.
    Stub: lightweight summary records, no embedding.
    """

    def __init__(self):
        self.hot: dict[str, Memory] = {}
        self.cold: dict[str, Memory] = {}
        self.stubs: dict[str, Memory] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, memory: Memory) -> None:
        self.hot[memory.id] = memory

    async def get(self, memory_id: str) -> Optional[Memory]:
        if memory_id in self.hot:
            return self.hot[memory_id]
        if memory_id in self.cold:
            return self.cold[memory_id]
        if memory_id in self.stubs:
            return self.stubs[memory_id]
        return None

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        results = []
        for mid in memory_ids:
            m = await self.get(mid)
            if m is not None:
                results.append(m)
        return results

    async def update(self, memory: Memory) -> None:
        if memory.id in self.hot:
            self.hot[memory.id] = memory
        elif memory.id in self.cold:
            self.cold[memory.id] = memory
        elif memory.id in self.stubs:
            self.stubs[memory.id] = memory

    async def delete(self, memory_id: str) -> None:
        self.hot.pop(memory_id, None)
        self.cold.pop(memory_id, None)
        self.stubs.pop(memory_id, None)

    async def delete_batch(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            await self.delete(mid)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_superseded: bool = False,
    ) -> list[tuple[Memory, float]]:
        results = []
        for mem in self.hot.values():
            if mem.embedding is None:
                continue
            if mem.is_superseded and not include_superseded:
                continue
            if mem.is_stub:
                continue
            sim = cosine_similarity(query_embedding, mem.embedding)
            results.append((mem, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Tiered storage
    # ------------------------------------------------------------------

    async def migrate_to_cold(self, memory_id: str, cold_since: datetime) -> None:
        if memory_id not in self.hot:
            return
        mem = self.hot.pop(memory_id)
        mem.is_cold = True
        mem.cold_since = cold_since
        self.cold[memory_id] = mem

    async def migrate_to_hot(self, memory_id: str) -> None:
        if memory_id not in self.cold:
            return
        mem = self.cold.pop(memory_id)
        mem.is_cold = False
        mem.cold_since = None
        mem.days_at_floor = 0
        self.hot[memory_id] = mem

    async def convert_to_stub(self, memory_id: str, stub_content: str) -> None:
        if memory_id in self.cold:
            old = self.cold.pop(memory_id)
        elif memory_id in self.hot:
            old = self.hot.pop(memory_id)
        else:
            return

        stub = Memory(
            id=old.id,
            content=stub_content,
            category=old.category,
            importance=old.importance,
            stability=0.0,
            created_at=old.created_at,
            is_stub=True,
            is_cold=True,
            embedding=None,
        )
        self.stubs[memory_id] = stub

    # ------------------------------------------------------------------
    # Associative links
    # ------------------------------------------------------------------

    async def create_or_strengthen_link(
        self, source_id: str, target_id: str, weight: float,
    ) -> None:
        # Links are stored directly on Memory objects in the in-memory adapter
        # This is a no-op — the engine manages links on Memory.associations directly
        pass

    async def get_linked_memories(
        self, memory_id: str, min_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        mem = await self.get(memory_id)
        if mem is None:
            return []
        results = []
        for assoc in mem.associations.values():
            if assoc.weight < min_weight:
                continue
            target = await self.get(assoc.target_id)
            if target is not None and not target.is_stub:
                results.append((target, assoc.weight))
        return results

    async def delete_link(self, source_id: str, target_id: str) -> None:
        source = await self.get(source_id)
        if source and target_id in source.associations:
            del source.associations[target_id]
        target = await self.get(target_id)
        if target and source_id in target.associations:
            del target.associations[source_id]

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    async def find_fading(
        self, threshold: float, exclude_core: bool = True,
    ) -> list[Memory]:
        # Note: caller must compute retention separately — this is a storage-level
        # filter. For in-memory adapter, we return all hot non-superseded memories
        # and let the engine filter by computed retention.
        results = []
        for mem in self.hot.values():
            if mem.is_superseded:
                continue
            if exclude_core and mem.category == MemoryCategory.CORE:
                continue
            results.append(mem)
        return results

    async def find_stable(
        self, min_stability: float, min_access_count: int,
    ) -> list[Memory]:
        results = []
        for mem in list(self.hot.values()) + list(self.cold.values()):
            if mem.stability >= min_stability and mem.access_count >= min_access_count:
                results.append(mem)
        return results

    async def mark_superseded(
        self, memory_ids: list[str], summary_id: str,
    ) -> None:
        for mid in memory_ids:
            mem = await self.get(mid)
            if mem is not None:
                mem.is_superseded = True
                mem.superseded_by = summary_id

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    async def all_active(self) -> list[Memory]:
        return list(self.hot.values()) + list(self.cold.values())

    async def all_hot(self) -> list[Memory]:
        return list(self.hot.values())

    async def all_cold(self) -> list[Memory]:
        return list(self.cold.values())

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    async def hot_count(self) -> int:
        return len(self.hot)

    async def cold_count(self) -> int:
        return len(self.cold)

    async def stub_count(self) -> int:
        return len(self.stubs)

    async def total_count(self) -> int:
        return len(self.hot) + len(self.cold) + len(self.stubs)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    async def batch_update(self, memories: list[Memory]) -> None:
        for mem in memories:
            await self.update(mem)

    async def update_retention_scores(self, updates: dict[str, float]) -> None:
        # Retention is computed on-the-fly in the in-memory adapter,
        # so this is a no-op. Remote adapters cache retention.
        pass

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def transaction(self, callback):
        # No-op for in-memory — just call directly
        return await callback(self)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        self.hot.clear()
        self.cold.clear()
        self.stubs.clear()
