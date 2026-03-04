"""
Core engine for cognitive-memory.

Implements all mechanisms from the paper:
- Decay model with floors (Section 3.2, 3.3)
- Two-tier retrieval boosting (Section 3.5)
- Associative memory graph (Section 3.6)
- Consolidation (Section 3.7)
- Tiered storage with cold TTL (Section 3.8)
- Core memory promotion (Section 3.4)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from .types import (
    Memory,
    MemoryCategory,
    CognitiveMemoryConfig,
    Association,
    SearchResult,
)
from .adapters.base import MemoryAdapter
from .embeddings import EmbeddingProvider, cosine_similarity


class CognitiveEngine:
    """
    The computational core. Operates on a MemoryAdapter and applies
    all the temporal dynamics described in the paper.
    """

    def __init__(self, adapter: MemoryAdapter, config: CognitiveMemoryConfig):
        self.adapter = adapter
        self.config = config

    # ------------------------------------------------------------------
    # Decay model - Equation 1
    # ------------------------------------------------------------------

    def compute_retention(self, memory: Memory, now: datetime) -> float:
        """
        R(m) = max(floor, exp(-dt / (S * B * beta_c)))

        Equation 1 in the paper.
        """
        if memory.is_stub:
            return 0.0

        last = memory.last_accessed_at or memory.created_at
        if last is None:
            return memory.floor

        dt_days = max(0.0, (now - last).total_seconds() / 86400.0)

        beta_c = memory.base_decay_rate
        if beta_c == float("inf"):
            return 1.0  # procedural memories don't decay

        S = max(memory.stability, 0.01)  # avoid division by zero
        B = 1.0 + (memory.importance * 2.0)
        B = min(B, 3.0)

        effective_rate = S * B * beta_c
        raw = math.exp(-dt_days / effective_rate)

        return max(memory.floor, raw)

    # ------------------------------------------------------------------
    # Retrieval scoring - Equation 3
    # ------------------------------------------------------------------

    def score_memory(
        self,
        memory: Memory,
        relevance: float,
        now: datetime,
    ) -> float:
        """
        score(m, q) = sim(m, q) * R(m)^alpha

        Equation 3 in the paper, with configurable exponent to control
        how aggressively decay suppresses retrieval.
        """
        retention = self.compute_retention(memory, now)
        alpha = self.config.retrieval_score_exponent
        return relevance * (retention ** alpha)

    # ------------------------------------------------------------------
    # Retrieval boosting - Section 3.5
    # ------------------------------------------------------------------

    def _spaced_rep_factor(self, memory: Memory, now: datetime) -> float:
        """
        Spaced repetition multiplier: min(2.0, dt / 7)
        Memories accessed after a longer gap get bigger boosts.
        """
        last = memory.last_accessed_at or memory.created_at
        if last is None:
            return 1.0
        dt_days = max(0.0, (now - last).total_seconds() / 86400.0)
        return min(
            self.config.max_spaced_rep_multiplier,
            dt_days / self.config.spaced_rep_interval_days,
        )

    def apply_direct_boost(self, memory: Memory, now: datetime, session_id: Optional[str] = None):
        """
        Direct retrieval boost (Section 3.5, Equation 4-5).

        stability += 0.1 * min(2.0, dt/7)
        access_count += 1
        """
        factor = self._spaced_rep_factor(memory, now)
        memory.stability = min(1.0, memory.stability + self.config.direct_boost * factor)
        memory.access_count += 1
        memory.last_accessed_at = now
        if session_id:
            memory.session_ids.add(session_id)

    def apply_associative_boost(self, memory: Memory, now: datetime, session_id: Optional[str] = None):
        """
        Associative retrieval boost (Section 3.5, Equation 6-7).

        stability += 0.03 * min(2.0, dt/7)
        access_count += 1
        """
        factor = self._spaced_rep_factor(memory, now)
        memory.stability = min(1.0, memory.stability + self.config.associative_boost * factor)
        memory.access_count += 1
        memory.last_accessed_at = now
        if session_id:
            memory.session_ids.add(session_id)

    # ------------------------------------------------------------------
    # Core memory promotion - Section 3.4
    # ------------------------------------------------------------------

    def check_core_promotion(self, memory: Memory) -> bool:
        """
        Promote to core if:
        1. access_count > threshold (default 10)
        2. stability >= threshold (default 0.85)
        3. accessed across >= threshold distinct sessions (default 3)
        """
        if memory.category == MemoryCategory.CORE:
            return False  # already core

        if (
            memory.access_count >= self.config.core_access_threshold
            and memory.stability >= self.config.core_stability_threshold
            and len(memory.session_ids) >= self.config.core_session_threshold
        ):
            memory.category = MemoryCategory.CORE
            return True
        return False

    # ------------------------------------------------------------------
    # Associative graph - Section 3.6
    # ------------------------------------------------------------------

    def strengthen_association(
        self,
        mem_a: Memory,
        mem_b: Memory,
        now: datetime,
    ):
        """
        When two memories are co-retrieved:
        w(a,b) += 0.1, capped at 1.0
        Bidirectional.
        """
        amount = self.config.association_strengthen_amount

        if mem_b.id not in mem_a.associations:
            mem_a.associations[mem_b.id] = Association(
                target_id=mem_b.id, weight=0.0, created_at=now,
            )
        assoc_ab = mem_a.associations[mem_b.id]
        assoc_ab.weight = min(1.0, assoc_ab.weight + amount)
        assoc_ab.last_co_retrieval = now

        if mem_a.id not in mem_b.associations:
            mem_b.associations[mem_a.id] = Association(
                target_id=mem_a.id, weight=0.0, created_at=now,
            )
        assoc_ba = mem_b.associations[mem_a.id]
        assoc_ba.weight = min(1.0, assoc_ba.weight + amount)
        assoc_ba.last_co_retrieval = now

    def decay_association(self, assoc: Association, now: datetime) -> float:
        """
        w(a,b) *= exp(-dt / 90)

        Equation 8 in the paper.
        """
        if assoc.last_co_retrieval is None:
            return assoc.weight
        dt_days = max(0.0, (now - assoc.last_co_retrieval).total_seconds() / 86400.0)
        tau = self.config.association_decay_constant_days
        decayed = assoc.weight * math.exp(-dt_days / tau)
        assoc.weight = decayed
        return decayed

    def get_associated_memories(
        self,
        memory: Memory,
        now: datetime,
    ) -> list[tuple[Memory, float]]:
        """
        Get memories associated with a given memory.
        Returns (memory, association_weight) for weights above threshold.
        Includes cold store lookups by ID (Section 3.8).

        NOTE: This is synchronous for backward compat with engine internals.
        For InMemoryAdapter, accesses dicts directly to avoid nested async.
        """
        results = []
        threshold = self.config.association_retrieval_threshold

        for assoc in list(memory.associations.values()):
            weight = self.decay_association(assoc, now)
            if weight < threshold:
                continue

            target = self._sync_get(assoc.target_id)
            if target is None or target.is_stub:
                continue

            results.append((target, weight))

        return results

    def _sync_get(self, memory_id: str) -> Optional[Memory]:
        """Synchronous get for in-memory adapter (avoids nested async)."""
        adapter = self.adapter
        # For InMemoryAdapter, access dicts directly
        if hasattr(adapter, 'hot'):
            if memory_id in adapter.hot:
                return adapter.hot[memory_id]
            if memory_id in adapter.cold:
                return adapter.cold[memory_id]
            if memory_id in adapter.stubs:
                return adapter.stubs[memory_id]
        return None

    # ------------------------------------------------------------------
    # Full retrieval pipeline
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        now: datetime,
        top_k: int = 10,
        session_id: Optional[str] = None,
        deep_recall: bool = False,
    ) -> list[SearchResult]:
        """
        Full retrieval pipeline:
        1. Similarity search in hot store
        2. Score by retention * relevance
        3. Collect associated memories (including from cold store)
        4. Apply direct/associative boosts
        5. Check core promotion
        6. Return sorted results
        """
        # Step 1: Similarity search in hot store
        include_superseded = deep_recall
        candidates = await self.adapter.search_similar(
            query_embedding, top_k=top_k * 3, include_superseded=include_superseded,
        )

        # Step 2: Score candidates
        alpha = self.config.retrieval_score_exponent
        scored: list[SearchResult] = []
        for mem, relevance in candidates:
            retention = self.compute_retention(mem, now)
            combined = relevance * (retention ** alpha)

            # Deep recall penalty for superseded memories
            if mem.is_superseded and deep_recall:
                combined *= self.config.deep_recall_penalty

            scored.append(SearchResult(
                memory=mem,
                relevance_score=relevance,
                retention_score=retention,
                combined_score=combined,
                is_associative=False,
                via_deep_recall=mem.is_superseded and deep_recall,
            ))

        # Sort by combined score
        scored.sort(key=lambda x: x.combined_score, reverse=True)

        # Take top-k direct results
        direct_results = scored[:top_k]

        # Step 3: Collect associated memories
        seen_ids = {r.memory.id for r in direct_results}
        associative_results: list[SearchResult] = []

        for result in direct_results:
            associated = self.get_associated_memories(result.memory, now)
            for assoc_mem, assoc_weight in associated:
                if assoc_mem.id in seen_ids:
                    continue
                seen_ids.add(assoc_mem.id)

                # Score the associated memory
                if assoc_mem.embedding is not None:
                    relevance = cosine_similarity(query_embedding, assoc_mem.embedding)
                else:
                    relevance = 0.1  # cold memory without embedding
                retention = self.compute_retention(assoc_mem, now)
                combined = relevance * (retention ** alpha) * assoc_weight

                associative_results.append(SearchResult(
                    memory=assoc_mem,
                    relevance_score=relevance,
                    retention_score=retention,
                    combined_score=combined,
                    is_associative=True,
                ))

        # Step 4: Apply boosts
        for result in direct_results:
            self.apply_direct_boost(result.memory, now, session_id)
            # If it was cold and got retrieved, migrate back to hot
            if result.memory.is_cold:
                await self.adapter.migrate_to_hot(result.memory.id)

        for result in associative_results:
            self.apply_associative_boost(result.memory, now, session_id)
            if result.memory.is_cold:
                await self.adapter.migrate_to_hot(result.memory.id)

        # Step 5: Check core promotions
        for result in direct_results + associative_results:
            self.check_core_promotion(result.memory)

        # Step 6: Strengthen associations between co-retrieved memories
        all_direct_mems = [r.memory for r in direct_results]
        for i in range(len(all_direct_mems)):
            for j in range(i + 1, len(all_direct_mems)):
                self.strengthen_association(all_direct_mems[i], all_direct_mems[j], now)

        # Combine and sort
        all_results = direct_results + associative_results
        all_results.sort(key=lambda x: x.combined_score, reverse=True)
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Cold storage management - Section 3.8
    # ------------------------------------------------------------------

    async def run_cold_migration(self, now: datetime):
        """
        Move memories to cold storage if they've been at floor
        for cold_migration_days consecutive days.
        Core memories are exempt.
        """
        threshold_days = self.config.cold_migration_days

        for mem in await self.adapter.all_hot():
            if mem.category == MemoryCategory.CORE:
                continue
            if mem.is_superseded:
                # superseded originals go to cold immediately
                await self.adapter.migrate_to_cold(mem.id, now)
                continue

            retention = self.compute_retention(mem, now)
            at_floor = abs(retention - mem.floor) < 0.001

            if at_floor:
                mem.days_at_floor += 1
            else:
                mem.days_at_floor = 0

            if mem.days_at_floor >= threshold_days:
                await self.adapter.migrate_to_cold(mem.id, now)

    async def run_cold_ttl_expiry(self, now: datetime):
        """
        Permanently remove cold memories that have exceeded the TTL.
        Before deletion, create a lightweight summary stub.
        """
        ttl_days = self.config.cold_storage_ttl_days

        for mem in await self.adapter.all_cold():
            if mem.cold_since is None:
                continue
            if mem.category == MemoryCategory.CORE:
                continue

            days_cold = (now - mem.cold_since).total_seconds() / 86400.0
            if days_cold >= ttl_days:
                # Create stub before deletion
                stub_content = f"[archived] {mem.content[:200]}"
                await self.adapter.convert_to_stub(mem.id, stub_content)

    # ------------------------------------------------------------------
    # Consolidation - Section 3.7
    # ------------------------------------------------------------------

    async def run_consolidation(
        self,
        now: datetime,
        embedder: EmbeddingProvider,
        llm_compress: callable = None,
    ):
        """
        Cluster fading memories by semantic similarity and compress
        groups into summaries. Originals are preserved in cold storage.
        """
        threshold = self.config.consolidation_retention_threshold
        group_size = self.config.consolidation_group_size
        sim_threshold = self.config.consolidation_similarity_threshold

        # Find fading non-core, non-superseded memories in hot store
        fading = []
        for mem in await self.adapter.all_hot():
            if mem.is_superseded or mem.category == MemoryCategory.CORE:
                continue
            retention = self.compute_retention(mem, now)
            if retention < threshold:
                fading.append(mem)

        if len(fading) < group_size:
            return

        # Group by category, then cluster by embedding similarity
        by_category: dict[MemoryCategory, list[Memory]] = {}
        for mem in fading:
            by_category.setdefault(mem.category, []).append(mem)

        for category, mems in by_category.items():
            if len(mems) < group_size:
                continue

            # Simple greedy clustering
            used = set()
            groups = []

            for i, mem_i in enumerate(mems):
                if mem_i.id in used:
                    continue
                group = [mem_i]
                for j, mem_j in enumerate(mems):
                    if i == j or mem_j.id in used:
                        continue
                    if mem_i.embedding and mem_j.embedding:
                        sim = cosine_similarity(mem_i.embedding, mem_j.embedding)
                        if sim >= sim_threshold:
                            group.append(mem_j)
                            if len(group) >= group_size:
                                break

                if len(group) >= group_size:
                    groups.append(group[:group_size])
                    for m in group[:group_size]:
                        used.add(m.id)

            # Create summaries for each group
            for group in groups:
                contents = [m.content for m in group]

                if llm_compress:
                    summary_text = llm_compress(contents)
                else:
                    summary_text = "Summary: " + " | ".join(contents)

                # Create summary memory
                summary = Memory(
                    content=summary_text,
                    category=category,
                    importance=max(m.importance for m in group),
                    stability=sum(m.stability for m in group) / len(group),
                    access_count=max(m.access_count for m in group),
                    created_at=now,
                    last_accessed_at=now,
                    embedding=embedder.embed(summary_text),
                )
                await self.adapter.create(summary)

                # Supersede originals and move to cold
                for m in group:
                    m.is_superseded = True
                    m.superseded_by = summary.id
                    await self.adapter.migrate_to_cold(m.id, now)

                    # Create association from summary to original
                    summary.associations[m.id] = Association(
                        target_id=m.id, weight=0.8, created_at=now,
                        last_co_retrieval=now,
                    )

    # ------------------------------------------------------------------
    # Maintenance tick
    # ------------------------------------------------------------------

    async def tick(self, now: datetime, embedder: EmbeddingProvider, llm_compress: callable = None):
        """
        Run all periodic maintenance:
        1. Cold migration
        2. Cold TTL expiry
        3. Consolidation (if enough fading memories)
        """
        await self.run_cold_migration(now)
        await self.run_cold_ttl_expiry(now)
        await self.run_consolidation(now, embedder, llm_compress)
