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
import time as _time
from datetime import datetime, timedelta
from itertools import combinations
from typing import Optional

from .types import (
    Memory,
    MemoryCategory,
    CognitiveMemoryConfig,
    Association,
    SearchResult,
    SearchResponse,
    SearchTrace,
    StageTrace,
)
from .adapters.base import MemoryAdapter
from .embeddings import EmbeddingProvider, cosine_similarity

_EXPIRABLE_TYPES = {"plan", "transient_state"}


def _ensure_bidirectional_association(
    mem_a: Memory,
    mem_b: Memory,
    weight: float,
    now: datetime,
    *,
    strengthen: bool = False,
) -> None:
    """Create or strengthen a bidirectional association between two memories."""
    for src, tgt in ((mem_a, mem_b), (mem_b, mem_a)):
        if tgt.id not in src.associations:
            src.associations[tgt.id] = Association(
                target_id=tgt.id, weight=0.0 if strengthen else weight,
                created_at=now, last_co_retrieval=now,
            )
        if strengthen:
            assoc = src.associations[tgt.id]
            assoc.weight = min(1.0, assoc.weight + weight)
            assoc.last_co_retrieval = now


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

        if self.config.decay_model == "power":
            raw = (1.0 + dt_days / effective_rate) ** (-self.config.power_decay_gamma)
        else:
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
        _ensure_bidirectional_association(mem_a, mem_b, amount, now, strengthen=True)

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

        for assoc in memory.associations.values():
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
    # Graph expansion / bridge discovery (v6)
    # ------------------------------------------------------------------

    def _expand_graph(
        self,
        anchors: list[Memory],
        now: datetime,
        seen_ids: set[str],
        max_hops: int,
    ) -> list[tuple[Memory, float]]:
        """BFS expansion through association graph."""
        frontier = anchors[:]
        results: list[tuple[Memory, float]] = []

        for _hop in range(max_hops):
            next_frontier: list[Memory] = []
            for mem in frontier:
                for assoc in mem.associations.values():
                    weight = self.decay_association(assoc, now)
                    if weight < self.config.min_bridge_edge_weight:
                        continue
                    if assoc.target_id in seen_ids:
                        continue

                    target = self._sync_get(assoc.target_id)
                    if target is None or target.is_stub:
                        continue

                    seen_ids.add(target.id)
                    results.append((target, weight))
                    next_frontier.append(target)
            frontier = next_frontier

        return results

    def _find_bridge_paths(
        self,
        anchors: list[Memory],
        now: datetime,
    ) -> list[list[str]]:
        """Find short weighted paths connecting top anchor nodes."""
        if len(anchors) < 2:
            return []

        chains: list[list[str]] = []
        anchor_ids = [m.id for m in anchors[:3]]  # top-3

        for a_id, b_id in combinations(anchor_ids, 2):
            path = self._bfs_path(a_id, b_id, now, max_depth=3)
            if path and len(path) > 2:  # non-trivial path
                chains.append(path)
                if len(chains) >= self.config.max_bridge_paths:
                    return chains
        return chains

    def _bfs_path(
        self,
        start_id: str,
        end_id: str,
        now: datetime,
        max_depth: int = 3,
    ) -> Optional[list[str]]:
        """BFS shortest path between two memory IDs through associations."""
        from collections import deque
        queue: deque[list[str]] = deque([[start_id]])
        visited = {start_id}

        while queue:
            path = queue.popleft()
            if len(path) > max_depth + 1:
                break

            current_id = path[-1]
            current = self._sync_get(current_id)
            if current is None:
                continue

            for assoc in current.associations.values():
                weight = self.decay_association(assoc, now)
                if weight < self.config.min_bridge_edge_weight:
                    continue
                if assoc.target_id in visited:
                    continue

                new_path = path + [assoc.target_id]
                if assoc.target_id == end_id:
                    return new_path

                visited.add(assoc.target_id)
                queue.append(new_path)

        return None

    # ------------------------------------------------------------------
    # Validity filtering (v6)
    # ------------------------------------------------------------------

    def _is_expired(self, memory: Memory, now: datetime) -> bool:
        """Check if a plan/transient_state memory has expired."""
        if memory.memory_type not in _EXPIRABLE_TYPES:
            return False
        if memory.valid_until is not None and now > memory.valid_until:
            return True
        if memory.ttl_seconds is not None and memory.created_at is not None:
            expiry = memory.created_at + timedelta(seconds=memory.ttl_seconds)
            if now > expiry:
                return True
        return False

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
        query_text: Optional[str] = None,
        trace: bool = False,
        extractor=None,
    ) -> SearchResponse:
        """
        Full retrieval pipeline:
        1. Similarity search in hot store
        1b. Lexical search (if hybrid enabled)
        2. Score by retention * relevance
        2b. Validity filtering
        3. Collect associated memories (including from cold store)
        3b. Graph expansion / bridge discovery
        4. Apply direct/associative boosts
        5. Check core promotion
        6. Return sorted results
        """
        search_trace = SearchTrace() if trace else None
        t_total = _time.monotonic() if trace else 0.0

        # Step 1: Similarity search in hot store
        t0 = _time.monotonic() if trace else 0.0
        include_superseded = deep_recall
        candidates = await self.adapter.search_similar(
            query_embedding, top_k=top_k * 3, include_superseded=include_superseded,
        )
        if trace and search_trace is not None:
            search_trace.stages["vector_search"] = StageTrace(
                name="vector_search",
                wall_ms=(_time.monotonic() - t0) * 1000,
                candidate_count=len(candidates),
            )

        # Step 1b: Lexical search (if hybrid enabled)
        if self.config.hybrid_search and query_text:
            t0 = _time.monotonic() if trace else 0.0
            lexical_results = await self.adapter.search_lexical(
                query_text, top_k=self.config.k_sparse, include_superseded=include_superseded,
            )
            # Merge: add lexical-only candidates
            seen_ids = {mem.id for mem, _ in candidates}
            lexical_added = 0
            for mem, _bm25_score in lexical_results:
                if mem.id not in seen_ids:
                    if mem.embedding is not None:
                        sim = cosine_similarity(query_embedding, mem.embedding)
                    else:
                        sim = 0.1
                    candidates.append((mem, sim))
                    seen_ids.add(mem.id)
                    lexical_added += 1
            if trace and search_trace is not None:
                search_trace.stages["lexical_search"] = StageTrace(
                    name="lexical_search",
                    wall_ms=(_time.monotonic() - t0) * 1000,
                    candidate_count=lexical_added,
                )

        # Step 2: Score candidates
        t0 = _time.monotonic() if trace else 0.0
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

        # Step 2b: Validity filtering
        if self.config.filter_expired_transients:
            filtered = []
            for r in scored:
                if self._is_expired(r.memory, now):
                    if deep_recall and self.config.include_expired_in_deep_recall:
                        r.combined_score *= self.config.deep_recall_penalty
                        r.via_deep_recall = True
                        filtered.append(r)
                    # else: exclude
                else:
                    filtered.append(r)
            scored = filtered

        if trace and search_trace is not None:
            search_trace.stages["scoring"] = StageTrace(
                name="scoring",
                wall_ms=(_time.monotonic() - t0) * 1000,
                candidate_count=len(scored),
            )

        # Sort by combined score
        scored.sort(key=lambda x: x.combined_score, reverse=True)

        # Step 2c: LLM rerank (if enabled and extractor provided)
        if self.config.rerank_enabled and extractor is not None and query_text and len(scored) > 1:
            t0 = _time.monotonic() if trace else 0.0
            k_rerank = min(self.config.k_rerank, len(scored))
            to_rerank = scored[:k_rerank]

            reranked_indices, usage = extractor.rerank_candidates(
                query_text,
                [r.memory.content for r in to_rerank],
            )

            # Rebuild: reranked items first, then remainder
            reranked = []
            used_indices = set()
            for idx in reranked_indices:
                if idx < len(to_rerank):
                    reranked.append(to_rerank[idx])
                    used_indices.add(idx)
            # Append omitted
            for i in range(len(to_rerank)):
                if i not in used_indices:
                    reranked.append(to_rerank[i])
            scored = reranked + scored[k_rerank:]

            if trace and search_trace is not None:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                search_trace.stages["rerank"] = StageTrace(
                    name="rerank",
                    wall_ms=(_time.monotonic() - t0) * 1000,
                    candidate_count=k_rerank,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                search_trace.total_tokens += prompt_tokens + completion_tokens

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

        # Step 3b: Graph expansion (if configured)
        if self.config.graph_expansion_hops > 0:
            expanded = self._expand_graph(
                [r.memory for r in direct_results],
                now, seen_ids, self.config.graph_expansion_hops,
            )
            for exp_mem, exp_weight in expanded:
                if exp_mem.embedding is not None:
                    relevance = cosine_similarity(query_embedding, exp_mem.embedding)
                else:
                    relevance = 0.1
                retention = self.compute_retention(exp_mem, now)
                combined = relevance * (retention ** alpha) * exp_weight

                associative_results.append(SearchResult(
                    memory=exp_mem,
                    relevance_score=relevance,
                    retention_score=retention,
                    combined_score=combined,
                    is_associative=True,
                ))

        # Bridge discovery
        evidence_chains: list[list[str]] = []
        if self.config.bridge_discovery:
            evidence_chains = self._find_bridge_paths(
                [r.memory for r in direct_results], now,
            )

        # Step 4: Apply boosts
        modified: list[Memory] = []
        for result in direct_results:
            self.apply_direct_boost(result.memory, now, session_id)
            modified.append(result.memory)
            if result.memory.is_cold:
                await self.adapter.migrate_to_hot(result.memory.id)

        for result in associative_results:
            self.apply_associative_boost(result.memory, now, session_id)
            modified.append(result.memory)
            if result.memory.is_cold:
                await self.adapter.migrate_to_hot(result.memory.id)

        # Step 5: Check core promotions
        for result in direct_results + associative_results:
            self.check_core_promotion(result.memory)

        # Step 6: Strengthen associations between co-retrieved memories
        all_direct_mems = [r.memory for r in direct_results]
        for mem_a, mem_b in combinations(all_direct_mems, 2):
            self.strengthen_association(mem_a, mem_b, now)

        # Persist all boost/promotion/association mutations
        if modified:
            await self.adapter.batch_update(modified)

        # Combine and sort
        all_results = direct_results + associative_results
        all_results.sort(key=lambda x: x.combined_score, reverse=True)

        final = all_results[:top_k]

        # Attach evidence chains to top result
        if evidence_chains and final:
            final[0].evidence_chains = evidence_chains

        if trace and search_trace is not None:
            search_trace.total_wall_ms = (_time.monotonic() - t_total) * 1000

        return SearchResponse(
            results=final,
            evidence_chains=evidence_chains,
            trace=search_trace,
        )

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
        updated: list[Memory] = []

        for mem in await self.adapter.all_hot():
            if mem.category == MemoryCategory.CORE:
                continue
            if mem.is_superseded:
                await self.adapter.migrate_to_cold(mem.id, now)
                continue

            retention = self.compute_retention(mem, now)
            at_floor = abs(retention - mem.floor) < 0.001

            if at_floor:
                mem.days_at_floor += 1
            else:
                mem.days_at_floor = 0

            updated.append(mem)

            if mem.days_at_floor >= threshold_days:
                await self.adapter.migrate_to_cold(mem.id, now)

        if updated:
            await self.adapter.batch_update(updated)

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

                # Persist summary associations added after create()
                await self.adapter.update(summary)

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
