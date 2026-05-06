"""
In-memory adapter for cognitive-memory.

Tiered storage with hot/cold/stub dicts and brute-force cosine similarity.
Zero dependencies, great for testing and single-process use.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Optional, Callable, TypeVar

from ..types import Memory, MemoryCategory
from ..embeddings import cosine_similarity
from .base import MemoryAdapter
from .errors import DuplicateMemoryError, MemoryNotFoundError

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
        # Adapter-level link table per spec/adapter-interface.md:147-177.
        # Keyed by canonically-sorted (a, b) tuple to make lookups symmetric.
        # Distinct from `Memory.associations`, which is the engine-side cache.
        self._links: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _link_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, memory: Memory) -> None:
        if memory.id in self.hot or memory.id in self.cold or memory.id in self.stubs:
            raise DuplicateMemoryError(memory.id)
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
        else:
            raise MemoryNotFoundError(memory.id)

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
        include_cold: bool = False,
        include_stubs: bool = False,
        user_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        results = []
        items = list(self.hot.values())
        if include_cold:
            items.extend(self.cold.values())
        if include_stubs:
            items.extend(self.stubs.values())
        for mem in items:
            if mem.embedding is None:
                continue
            if mem.is_superseded and not include_superseded:
                continue
            if mem.is_stub and not include_stubs:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            sim = cosine_similarity(query_embedding, mem.embedding)
            results.append((mem, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Lexical search (BM25)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    async def search_lexical(
        self,
        query: str,
        top_k: int = 10,
        include_superseded: bool = False,
        include_cold: bool = False,
        include_stubs: bool = False,
        user_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """BM25 lexical search over hot store."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build corpus stats
        docs: list[tuple[Memory, list[str]]] = []
        items = list(self.hot.values())
        if include_cold:
            items.extend(self.cold.values())
        if include_stubs:
            items.extend(self.stubs.values())
        for mem in items:
            if mem.is_stub and not include_stubs:
                continue
            if mem.is_superseded and not include_superseded:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            docs.append((mem, self._tokenize(mem.content)))

        if not docs:
            return []

        N = len(docs)
        avgdl = sum(len(tokens) for _, tokens in docs) / N if N else 1.0
        k1, b = 1.2, 0.75

        # IDF per query token
        df: dict[str, int] = Counter()
        for _, tokens in docs:
            unique = set(tokens)
            for t in query_tokens:
                if t in unique:
                    df[t] += 1

        idf: dict[str, float] = {}
        for t in query_tokens:
            n = df.get(t, 0)
            idf[t] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)

        # Score each doc
        results: list[tuple[Memory, float]] = []
        for mem, tokens in docs:
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in query_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                num = f * (k1 + 1)
                denom = f + k1 * (1 - b + b * dl / avgdl)
                score += idf.get(t, 0.0) * num / denom
            if score > 0:
                results.append((mem, score))

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
            is_cold=False,
            embedding=None,
        )
        self.stubs[memory_id] = stub

    # ------------------------------------------------------------------
    # Associative links
    # ------------------------------------------------------------------

    async def create_or_strengthen_link(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        link_type: str = "association",
    ) -> None:
        """Persist a link in the adapter-level table.

        Strengthen semantics match the engine's expectation: existing weight is
        added, capped at 1.0. The spec's prior 'max(existing, new)' wording is
        being corrected as part of this work — see spec/adapter-interface.md.
        """
        key = self._link_key(source_id, target_id)
        now = datetime.now()
        existing = self._links.get(key)
        existing_weight = existing["weight"] if existing else 0.0
        self._links[key] = {
            "weight": min(1.0, existing_weight + weight),
            "link_type": existing["link_type"] if existing else link_type,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }

    async def get_linked_memories(
        self, memory_id: str, min_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        """Return memories linked to ``memory_id`` from the adapter-level
        store, merged with any per-memory ``associations`` cache. Per-target
        weight = max across the two stores."""
        # Adapter-level store
        weights: dict[str, float] = {}
        for (a, b), row in self._links.items():
            if a == memory_id:
                other = b
            elif b == memory_id:
                other = a
            else:
                continue
            if row["weight"] >= min_weight:
                weights[other] = max(weights.get(other, 0.0), row["weight"])

        # Per-memory cache (legacy path; engine still uses this for decay)
        mem = await self.get(memory_id)
        if mem is not None:
            for assoc in mem.associations.values():
                if assoc.weight >= min_weight:
                    weights[assoc.target_id] = max(
                        weights.get(assoc.target_id, 0.0), assoc.weight
                    )

        results: list[tuple[Memory, float]] = []
        for target_id, weight in weights.items():
            target = await self.get(target_id)
            if target is not None and not target.is_stub:
                results.append((target, weight))
        return results

    async def delete_link(self, source_id: str, target_id: str) -> None:
        """Idempotent — removing a non-existent link is a no-op."""
        self._links.pop(self._link_key(source_id, target_id), None)

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

    async def all_active(self, user_id: Optional[str] = None) -> list[Memory]:
        items = list(self.hot.values()) + list(self.cold.values())
        return [m for m in items if user_id is None or m.user_id == user_id]

    async def all_hot(self, user_id: Optional[str] = None) -> list[Memory]:
        return [
            m for m in self.hot.values()
            if user_id is None or m.user_id == user_id
        ]

    async def all_cold(self, user_id: Optional[str] = None) -> list[Memory]:
        return [
            m for m in self.cold.values()
            if user_id is None or m.user_id == user_id
        ]

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    async def hot_count(self, user_id: Optional[str] = None) -> int:
        return sum(
            1 for m in self.hot.values()
            if user_id is None or m.user_id == user_id
        )

    async def cold_count(self, user_id: Optional[str] = None) -> int:
        return sum(
            1 for m in self.cold.values()
            if user_id is None or m.user_id == user_id
        )

    async def stub_count(self, user_id: Optional[str] = None) -> int:
        return sum(
            1 for m in self.stubs.values()
            if user_id is None or m.user_id == user_id
        )

    async def total_count(self, user_id: Optional[str] = None) -> int:
        return (
            await self.hot_count(user_id)
            + await self.cold_count(user_id)
            + await self.stub_count(user_id)
        )

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
        self._links.clear()
