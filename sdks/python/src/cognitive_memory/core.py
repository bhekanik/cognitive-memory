"""
CognitiveMemory - the main public API.

This is the class users interact with. It wires together the adapter,
engine, extractor, and embedder into a coherent interface.

All public methods are async. For sync usage, use SyncCognitiveMemory.
"""

from __future__ import annotations

import logging
from datetime import datetime
from itertools import combinations
from typing import Optional, Literal

from .types import (
    Memory,
    MemoryCategory,
    CognitiveMemoryConfig,
    SearchResult,
    SearchResponse,
)
from .adapters.base import MemoryAdapter
from .adapters.memory import InMemoryAdapter
from .engine import CognitiveEngine, _ensure_bidirectional_association
from .extraction import MemoryExtractor
from .llm import LLMProvider
from .embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HashEmbeddings,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.85
STABILITY_REINFORCEMENT_THRESHOLD = 0.75
INGESTION_ASSOCIATION_THRESHOLD = 0.4
INGESTION_ASSOCIATION_BASE_WEIGHT = 0.2


def _session_roots(session_ids: set[str]) -> set[str]:
    """Extract session roots by stripping '_perspective_*' suffixes."""
    import re
    return {re.sub(r"_perspective_.*$", "", sid) for sid in session_ids}


def _close_validity_window(old: Memory, new: Memory, now: datetime, relation_type: str) -> None:
    """Preserve old state while marking temporal supersession for the experiment path."""
    boundary = new.valid_from or new.created_at or now
    old.valid_until = boundary
    if not isinstance(old.temporal, dict):
        old.temporal = {}
    valid_time = old.temporal.get("valid_time")
    if not isinstance(valid_time, dict):
        valid_time = {}
    valid_time["valid_to"] = boundary.isoformat()
    valid_time["status"] = "superseded"
    old.temporal["valid_time"] = valid_time
    relations = old.temporal.get("relations")
    if not isinstance(relations, list):
        relations = []
    relations.append({
        "type": relation_type,
        "target_memory_id": new.id,
        "confidence": 0.8,
    })
    old.temporal["relations"] = relations

    if new.valid_from is None:
        new.valid_from = boundary
    if not isinstance(new.temporal, dict):
        new.temporal = {}
    new_valid_time = new.temporal.get("valid_time")
    if not isinstance(new_valid_time, dict):
        new_valid_time = {}
    new_valid_time.setdefault("valid_from", boundary.isoformat())
    if new_valid_time.get("status") in (None, "", "unknown"):
        new_valid_time["status"] = "current"
    new.temporal["valid_time"] = new_valid_time


class CognitiveMemory:
    """
    Main interface for the cognitive-memory system.

    Usage:
        from cognitive_memory import CognitiveMemory

        mem = CognitiveMemory()  # uses OpenAI embeddings + extraction
        mem = CognitiveMemory(embedder="hash")  # offline testing

        # Ingest
        await mem.ingest("User said they are allergic to peanuts",
                    session_id="s1", timestamp=datetime.now())

        # Or extract from conversation
        await mem.extract_and_store(conversation_text, session_id="s1", ...)

        # Search
        results = await mem.search("what is the user allergic to?")

        # Maintenance
        await mem.tick()  # run cold migration, TTL expiry, consolidation
    """

    def __init__(
        self,
        config: Optional[CognitiveMemoryConfig] = None,
        embedder: Optional[EmbeddingProvider | Literal["openai", "hash"]] = None,
        adapter: Optional[MemoryAdapter] = None,
        llm: Optional[LLMProvider] = None,
        user_id: str = "default",
    ):
        self.config = config or CognitiveMemoryConfig()
        self._adapter = adapter or InMemoryAdapter()
        self._user_id = user_id
        self._engine = CognitiveEngine(self._adapter, self.config, user_id=user_id)
        self._extractor = MemoryExtractor(self.config, llm=llm)

        if embedder is None or embedder == "openai":
            self._embedder = OpenAIEmbeddings(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
            )
        elif embedder == "hash":
            self._embedder = HashEmbeddings(dimensions=384)
        elif isinstance(embedder, EmbeddingProvider):
            self._embedder = embedder
        else:
            raise ValueError(f"Unknown embedder: {embedder}")

        self._tick_counter = 0
        self._conflict_queue: list[tuple[str, str, float]] = []  # (new_id, existing_id, similarity)

    # ------------------------------------------------------------------
    # Low-level: add a memory directly
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.EPISODIC,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Memory:
        """
        Add a single memory directly (bypassing LLM extraction).
        Useful for testing or when you've already extracted memories.
        """
        now = timestamp or datetime.now()

        mem = Memory(
            user_id=self._user_id,
            content=content,
            category=category,
            importance=importance,
            stability=0.1 + (importance * 0.3),
            created_at=now,
            last_accessed_at=now,
            embedding=self._embedder.embed(content),
        )
        if session_id:
            mem.session_ids.add(session_id)

        # Tag potential conflicts for deferred resolution at tick
        if mem.embedding is not None:
            similar = await self._adapter.search_similar(
                mem.embedding, top_k=5, user_id=self._user_id,
            )
            for existing_mem, sim in similar:
                if existing_mem.id != mem.id and sim > CONFLICT_SIMILARITY_THRESHOLD:
                    self._conflict_queue.append((mem.id, existing_mem.id, sim))

        await self._adapter.create(mem)
        return mem

    async def add_memory_object(self, memory: Memory) -> Memory:
        """Add a pre-built Memory object. Embeds if needed.

        If the memory carries the placeholder user_id ('default'), it inherits
        this CognitiveMemory's configured user_id; an explicit non-default
        user_id is preserved.
        """
        if memory.user_id == "default":
            memory.user_id = self._user_id
        if memory.embedding is None:
            memory.embedding = self._embedder.embed(memory.content)
        await self._adapter.create(memory)
        return memory

    # ------------------------------------------------------------------
    # High-level: extract + store from conversation
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        conversation_text: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
        run_tick: bool = True,
    ) -> list[Memory]:
        """
        Extract memories from conversation text, embed them, and store.

        Behavior depends on ``config.extraction_mode``:
        - ``"semantic"`` (default): LLM extracts structured facts.
        - ``"raw"``: each conversation turn stored verbatim (no LLM).
        - ``"hybrid"``: both semantic extraction AND raw turns stored.
        """
        mode = self.config.extraction_mode
        if mode not in ("raw", "semantic", "hybrid"):
            raise ValueError(f"Invalid extraction_mode: {mode!r}. Must be 'raw', 'semantic', or 'hybrid'.")

        import time as _time
        now = timestamp or datetime.now()
        stored: list[Memory] = []

        # --- Semantic extraction (modes: semantic, hybrid) ---
        if mode in ("semantic", "hybrid"):
            logger.info(f"[ingest:{session_id}] Starting LLM extraction")
            _t0 = _time.time()
            memories = self._extractor.extract_from_conversation(
                conversation_text, session_id, now,
            )
            logger.info(f"[ingest:{session_id}] Extracted {len(memories)} memories in {_time.time()-_t0:.1f}s")
            if memories:
                _t0 = _time.time()
                contents = [m.content for m in memories]
                embeddings = self._embedder.embed_batch(contents)
                logger.info(f"[ingest:{session_id}] Embedded {len(contents)} memories in {_time.time()-_t0:.1f}s")
                for mem, emb in zip(memories, embeddings):
                    mem.embedding = emb

                _queued = 0
                for mem in memories:
                    mem.user_id = self._user_id
                    if mem.embedding is not None:
                        similar = await self._adapter.search_similar(
                            mem.embedding, top_k=5, user_id=self._user_id,
                        )
                        reinforced = []
                        for existing_mem, sim in similar:
                            if existing_mem.id == mem.id:
                                continue
                            # Tag potential conflicts for deferred LLM resolution at tick
                            if sim > CONFLICT_SIMILARITY_THRESHOLD:
                                # Skip if same session root (e.g. dual-perspective of same conversation)
                                if _session_roots(mem.session_ids) & _session_roots(existing_mem.session_ids):
                                    continue
                                self._conflict_queue.append((mem.id, existing_mem.id, sim))
                                _queued += 1
                            # Stability reinforcement
                            if sim > STABILITY_REINFORCEMENT_THRESHOLD:
                                existing_mem.stability = min(1.0, existing_mem.stability + 0.05)
                                reinforced.append(existing_mem)
                        if reinforced:
                            await self._adapter.batch_update(reinforced)
                    await self._adapter.create(mem)
                    stored.append(mem)
                logger.info(f"[ingest:{session_id}] Stored {len(stored)} memories, queued {_queued} conflict candidates (queue size: {len(self._conflict_queue)})")

        # --- Raw turn storage (modes: raw, hybrid) ---
        if mode in ("raw", "hybrid"):
            raw_memories = self._extractor.extract_raw_turns(
                conversation_text, session_id, now,
            )
            if raw_memories:
                raw_contents = [m.content for m in raw_memories]
                raw_embeddings = self._embedder.embed_batch(raw_contents)
                for mem, emb in zip(raw_memories, raw_embeddings):
                    mem.user_id = self._user_id
                    mem.embedding = emb
                    await self._adapter.create(mem)
                    stored.append(mem)

        if not stored:
            return []

        # Synaptic tagging: link co-ingested memories gated by similarity.
        # Writes to BOTH the per-memory `Memory.associations` cache (used at
        # retrieval-time decay) AND the adapter-level link table (spec contract).
        if len(stored) > 1:
            for mem_a, mem_b in combinations(stored, 2):
                if mem_a.embedding is None or mem_b.embedding is None:
                    continue
                sim = cosine_similarity(mem_a.embedding, mem_b.embedding)
                if sim >= INGESTION_ASSOCIATION_THRESHOLD:
                    weight = min(0.5, INGESTION_ASSOCIATION_BASE_WEIGHT + (sim - INGESTION_ASSOCIATION_THRESHOLD) * 0.5)
                    _ensure_bidirectional_association(mem_a, mem_b, weight, now)
                    await self._adapter.create_or_strengthen_link(
                        mem_a.id, mem_b.id, weight,
                    )
            await self._adapter.batch_update(stored)

        # Periodic maintenance (skip during batch benchmarks)
        if run_tick and self.config.run_maintenance_during_ingestion:
            self._tick_counter += 1
            if self._tick_counter % 5 == 0:  # every 5 ingestions
                await self.tick(now)

        return stored

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 10,
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None,
        deep_recall: bool = False,
        trace: bool = False,
    ) -> SearchResponse:
        """
        Search memories with full retention-weighted scoring.

        Args:
            query: natural language query
            top_k: max results
            timestamp: when the query happens (for decay calc). Defaults to now.
            session_id: current session (for boost tracking)
            deep_recall: include superseded originals (Section 3.8)
            trace: include per-stage instrumentation in response
        """
        now = timestamp or datetime.now()
        query_embedding = self._embedder.embed(query)

        return await self._engine.search(
            query_embedding=query_embedding,
            now=now,
            top_k=top_k,
            session_id=session_id,
            deep_recall=deep_recall,
            query_text=query,
            trace=trace,
            extractor=self._extractor if self.config.rerank_enabled else None,
        )

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    async def _check_conflicts(self, new_memory: Memory, now: datetime):
        """
        Check if new memory conflicts with existing memories.
        On contradiction/update: demote the old memory.
        """
        import time as _time
        all_hot = await self._adapter.all_hot()
        candidates = [
            m for m in all_hot
            if not m.is_superseded and not m.is_stub
            and (m.importance > 0.5 or m.category == MemoryCategory.CORE)
        ]

        if not candidates or new_memory.embedding is None:
            return

        _llm_calls = 0
        _sim_checks = 0
        demoted: list[Memory] = []
        for existing in candidates:
            if existing.embedding is None:
                continue
            sim = cosine_similarity(new_memory.embedding, existing.embedding)
            _sim_checks += 1
            if sim < CONFLICT_SIMILARITY_THRESHOLD:
                continue

            _llm_calls += 1
            _t0 = _time.time()
            conflict_type = self._extractor.detect_conflict(new_memory, existing)
            logger.debug(f"[conflict] detect_conflict call {_llm_calls} took {(_time.time()-_t0)*1000:.0f}ms (sim={sim:.3f})")

            if conflict_type in ("CONTRADICTION", "UPDATE"):
                logger.info(
                    f"Conflict detected ({conflict_type}): "
                    f"'{existing.content[:50]}' -> '{new_memory.content[:50]}'"
                )
                # Capture category before demotion
                was_core = existing.category == MemoryCategory.CORE
                if was_core:
                    existing.category = MemoryCategory.SEMANTIC
                existing.contradicted_by = new_memory.id
                _close_validity_window(existing, new_memory, now, "updates")
                new_memory.importance = max(new_memory.importance, existing.importance)
                if conflict_type == "CONTRADICTION" and was_core:
                    new_memory.category = MemoryCategory.CORE
                demoted.append(existing)

        if demoted:
            await self._adapter.batch_update(demoted)

        if _llm_calls > 0:
            logger.info(f"[conflict] {_sim_checks} sim checks, {_llm_calls} LLM calls for '{new_memory.content[:40]}...'")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def tick(self, now: Optional[datetime] = None):
        """Run periodic maintenance: cold migration, TTL expiry, consolidation, conflict resolution."""
        now = now or datetime.now()

        # Process deferred conflict queue (capped per tick)
        await self._resolve_conflict_queue(now, max_per_tick=50)

        await self._engine.tick(now, self._embedder, self._extractor.compress_memories)

    async def _resolve_conflict_queue(self, now: datetime, max_per_tick: int = 50):
        """Process pending conflict candidates with LLM verification."""
        if not self._conflict_queue:
            return

        # Sort by similarity descending — highest similarity = most likely conflict
        self._conflict_queue.sort(key=lambda x: x[2], reverse=True)

        processed = 0
        resolved = 0
        remaining = []
        demoted: list[Memory] = []

        for new_id, existing_id, sim in self._conflict_queue:
            if processed >= max_per_tick:
                remaining.append((new_id, existing_id, sim))
                continue

            # Look up both memories (they may have been deleted/superseded since queuing)
            new_mem = await self._adapter.get(new_id)
            existing_mem = await self._adapter.get(existing_id)

            if new_mem is None or existing_mem is None:
                processed += 1
                continue
            if existing_mem.is_superseded or new_mem.is_superseded:
                processed += 1
                continue

            conflict_type = self._extractor.detect_conflict(new_mem, existing_mem)
            processed += 1

            if conflict_type in ("CONTRADICTION", "UPDATE"):
                logger.info(
                    f"[tick:conflict] {conflict_type}: "
                    f"'{existing_mem.content[:50]}' -> '{new_mem.content[:50]}'"
                )
                was_core = existing_mem.category == MemoryCategory.CORE
                if was_core:
                    existing_mem.category = MemoryCategory.SEMANTIC
                existing_mem.contradicted_by = new_mem.id
                _close_validity_window(existing_mem, new_mem, now, "updates")
                new_mem.importance = max(new_mem.importance, existing_mem.importance)
                if conflict_type == "CONTRADICTION" and was_core:
                    new_mem.category = MemoryCategory.CORE
                demoted.extend([existing_mem, new_mem])
                resolved += 1

        if demoted:
            await self._adapter.batch_update(demoted)

        self._conflict_queue = remaining
        if processed > 0:
            logger.info(f"[tick:conflict] Processed {processed} candidates, {resolved} conflicts resolved, {len(remaining)} remaining")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Return current memory system statistics scoped to this user."""
        now = datetime.now()
        all_mems = await self._adapter.all_active(user_id=self._user_id)

        core_count = sum(1 for m in all_mems if m.category == MemoryCategory.CORE)
        faint_count = sum(
            1 for m in all_mems
            if not m.is_stub and self._engine.compute_retention(m, now) < self.config.faint_threshold
        )
        retentions = [
            self._engine.compute_retention(m, now)
            for m in all_mems if not m.is_stub
        ]
        avg_retention = sum(retentions) / len(retentions) if retentions else 0.0

        return {
            "total_memories": await self._adapter.total_count(user_id=self._user_id),
            "hot_memories": await self._adapter.hot_count(user_id=self._user_id),
            "cold_memories": await self._adapter.cold_count(user_id=self._user_id),
            "stub_memories": await self._adapter.stub_count(user_id=self._user_id),
            "core_memories": core_count,
            "faint_memories": faint_count,
            "avg_retention": avg_retention,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    async def clear(self):
        """Clear all memories."""
        await self._adapter.clear()
        self._tick_counter = 0

    # ------------------------------------------------------------------
    # CLI-equivalent surface (paper §3.4 / §3.6 + adapter pass-throughs)
    # ------------------------------------------------------------------

    async def store_core(
        self,
        content: str,
        importance: float = 0.7,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Memory:
        """Store a memory tagged as ``core`` at encoding time.

        Synaptic tagging shortcut (paper §3.4): identity-critical
        information receives the protected core retention floor (0.6 by
        default) immediately rather than earning core status through
        repeated retrieval. Use sparingly — most memories should reach
        core status emergently.
        """
        return await self.add(
            content=content,
            category=MemoryCategory.CORE,
            importance=importance,
            session_id=session_id,
            timestamp=timestamp,
        )

    async def store_batch(
        self,
        contents: list[str],
        category: MemoryCategory = MemoryCategory.SEMANTIC,
        importance: float = 0.5,
        link_weight: float = 0.5,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[Memory]:
        """Store multiple memories together; auto-create bidirectional
        associations between every pair (paper §3.6: "memories form
        bidirectional associations when they are retrieved together OR
        created in the same context").

        For ``RemoteAdapter`` (daemon-backed), delegates to the daemon's
        native ``StoreBatch`` for atomicity. For in-process adapters,
        creates each memory then strengthens every pair via
        ``create_or_strengthen_link``.
        """
        if not contents:
            return []

        # Fast path: adapter has a native batch implementation (daemon).
        native_batch = getattr(self._adapter, "create_batch", None)
        if callable(native_batch):
            now = timestamp or datetime.now()
            mem_objs = [
                Memory(
                    user_id=self._user_id,
                    content=c,
                    category=category,
                    importance=importance,
                    stability=0.1 + (importance * 0.3),
                    created_at=now,
                    last_accessed_at=now,
                )
                for c in contents
            ]
            result = await native_batch(mem_objs, initial_link_weight=link_weight)
            ids = result["ids"]
            # Re-fetch the persisted records so callers get accurate state.
            return await self._adapter.get_batch(ids)

        # Fan-out path: in-process adapter. Insert each, then link pairs.
        mems = []
        for c in contents:
            mems.append(
                await self.add(
                    content=c,
                    category=category,
                    importance=importance,
                    session_id=session_id,
                    timestamp=timestamp,
                )
            )
        for a, b in combinations(mems, 2):
            await self._adapter.create_or_strengthen_link(a.id, b.id, link_weight)
            await self._adapter.create_or_strengthen_link(b.id, a.id, link_weight)
        return mems

    # -- CRUD pass-throughs --

    async def get(self, memory_id: str) -> Optional[Memory]:
        return await self._adapter.get(memory_id)

    async def get_many(self, memory_ids: list[str]) -> list[Memory]:
        return await self._adapter.get_batch(memory_ids)

    async def list(
        self,
        *,
        category: Optional[MemoryCategory] = None,
        memory_type: Optional[str] = None,
        include_cold: bool = False,
        include_stubs: bool = False,
        include_superseded: bool = False,
    ) -> list[Memory]:
        """Enumerate memories. Filters apply post-fetch where the adapter
        doesn't support them natively."""
        if include_cold:
            mems = await self._adapter.all_active(user_id=self._user_id)
        else:
            mems = await self._adapter.all_hot(user_id=self._user_id)
        if not include_stubs:
            mems = [m for m in mems if not m.is_stub]
        if not include_superseded:
            mems = [m for m in mems if not m.is_superseded]
        if category is not None:
            mems = [m for m in mems if m.category == category]
        if memory_type is not None:
            mems = [m for m in mems if m.memory_type == memory_type]
        return mems

    async def update_memory(self, memory: Memory) -> None:
        """Persist updates to a memory you've already mutated locally.

        For partial updates without a local Memory object, fetch first:
        ``mem = await cm.get(id); mem.importance = 0.9; await cm.update_memory(mem)``.
        """
        if memory.user_id == "default":
            memory.user_id = self._user_id
        await self._adapter.update(memory)

    async def delete(self, memory_id: str) -> None:
        await self._adapter.delete(memory_id)

    async def delete_many(self, memory_ids: list[str]) -> None:
        await self._adapter.delete_batch(memory_ids)

    # -- Search variants --

    async def search_lexical(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Memory, float]]:
        """BM25-only search. Returns (memory, score) pairs."""
        return await self._adapter.search_lexical(
            query, top_k=top_k, user_id=self._user_id,
        )

    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_cold: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Search by raw embedding vector — for callers that have already
        embedded the query."""
        return await self._adapter.search_similar(
            embedding,
            top_k=top_k,
            include_cold=include_cold,
            user_id=self._user_id,
        )

    # -- Links --

    async def link(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.1,
    ) -> None:
        """Create or strengthen a bidirectional association."""
        await self._adapter.create_or_strengthen_link(source_id, target_id, weight)
        # Bidirectional — also strengthen the reverse edge.
        if hasattr(self._adapter, "_supports_bidirectional_link") or True:
            # `create_or_strengthen_link` is documented as bidirectional in
            # the SDK's ABC; in-process adapters honour that. RemoteAdapter
            # is bidirectional by default. No second call needed.
            pass

    async def unlink(self, source_id: str, target_id: str) -> None:
        """Delete an association (bidirectional in the sense that the
        adapter contract is bidirectional)."""
        await self._adapter.delete_link(source_id, target_id)

    async def linked(
        self,
        memory_id: str,
        min_strength: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        return await self._adapter.get_linked_memories(memory_id, min_weight=min_strength)

    async def linked_many(
        self,
        memory_ids: list[str],
        min_strength: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        # Adapter has it as ``get_linked_memories_multiple``? The Python
        # base is single-source-only; fall back to N calls with dedup.
        seen: dict[str, tuple[Memory, float]] = {}
        for mid in memory_ids:
            for mem, w in await self._adapter.get_linked_memories(
                mid, min_weight=min_strength,
            ):
                # Keep the strongest weight per target.
                if mem.id not in seen or seen[mem.id][1] < w:
                    seen[mem.id] = (mem, w)
        return list(seen.values())

    # -- Lifecycle / consolidation --

    async def find_fading(
        self,
        max_retention: float,
        exclude_core: bool = True,
    ) -> list[Memory]:
        return await self._adapter.find_fading(
            threshold=max_retention,
            exclude_core=exclude_core,
        )

    async def find_stable(
        self,
        min_stability: float,
        min_access_count: int,
    ) -> list[Memory]:
        return await self._adapter.find_stable(
            min_stability=min_stability,
            min_access_count=min_access_count,
        )

    async def mark_superseded(
        self,
        memory_ids: list[str],
        summary_id: str,
    ) -> None:
        await self._adapter.mark_superseded(memory_ids, summary_id)

    async def migrate_to_cold(
        self,
        memory_id: str,
        cold_since: Optional[datetime] = None,
    ) -> None:
        await self._adapter.migrate_to_cold(
            memory_id, cold_since or datetime.now(),
        )

    async def migrate_to_hot(self, memory_id: str) -> None:
        await self._adapter.migrate_to_hot(memory_id)

    async def convert_to_stub(self, memory_id: str, stub_content: str) -> None:
        await self._adapter.convert_to_stub(memory_id, stub_content)

    # -- Retention --

    async def set_retention(self, memory_id: str, floor: float) -> None:
        """Set the retention floor for one memory."""
        await self._adapter.update_retention_scores({memory_id: floor})

    async def set_retentions(self, updates: dict[str, float]) -> None:
        """Atomically set retention floors for many memories."""
        await self._adapter.update_retention_scores(updates)

    # -- Counts --

    async def counts(self) -> dict[str, int]:
        """Per-user tier counts: hot, cold, stub, total."""
        return {
            "hot": await self._adapter.hot_count(user_id=self._user_id),
            "cold": await self._adapter.cold_count(user_id=self._user_id),
            "stub": await self._adapter.stub_count(user_id=self._user_id),
            "total": await self._adapter.total_count(user_id=self._user_id),
        }

    # -- Daemon-only extras (RemoteAdapter) --

    async def mint_bridge_token(
        self,
        scope: Literal["read", "write", "admin"] = "write",
        ttl_seconds: int = 30 * 24 * 3600,
    ) -> dict:
        """Mint a bearer token for the cm-http bridge. Only available
        when this CognitiveMemory is configured with ``RemoteAdapter``.
        """
        mint = getattr(self._adapter, "mint_bridge_token", None)
        if not callable(mint):
            raise RuntimeError(
                "mint_bridge_token requires RemoteAdapter; current adapter "
                f"is {type(self._adapter).__name__}"
            )
        return await mint(scope=scope, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> MemoryAdapter:
        return self._adapter

    @property
    def engine(self) -> CognitiveEngine:
        return self._engine

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder
