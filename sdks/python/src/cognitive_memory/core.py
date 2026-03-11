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
from .embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HashEmbeddings,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.6
STABILITY_REINFORCEMENT_THRESHOLD = 0.75
INGESTION_ASSOCIATION_THRESHOLD = 0.4
INGESTION_ASSOCIATION_BASE_WEIGHT = 0.2


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
    ):
        self.config = config or CognitiveMemoryConfig()
        self._adapter = adapter or InMemoryAdapter()
        self._engine = CognitiveEngine(self._adapter, self.config)
        self._extractor = MemoryExtractor(self.config)

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

        await self._check_conflicts(mem, now)

        await self._adapter.create(mem)
        return mem

    async def add_memory_object(self, memory: Memory) -> Memory:
        """Add a pre-built Memory object. Embeds if needed."""
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

                _hot_count = len(await self._adapter.all_hot())
                logger.info(f"[ingest:{session_id}] Conflict detection: {len(memories)} new × {_hot_count} hot memories")
                for _mi, mem in enumerate(memories):
                    _t0 = _time.time()
                    await self._check_conflicts(mem, now)
                    _conflict_ms = (_time.time()-_t0)*1000
                    if _conflict_ms > 1000:
                        logger.warning(f"[ingest:{session_id}] conflict check {_mi+1}/{len(memories)} took {_conflict_ms:.0f}ms (SLOW)")
                    if mem.embedding is not None:
                        similar = await self._adapter.search_similar(mem.embedding, top_k=3)
                        reinforced = []
                        for existing_mem, sim in similar:
                            if sim > STABILITY_REINFORCEMENT_THRESHOLD and existing_mem.id != mem.id:
                                existing_mem.stability = min(1.0, existing_mem.stability + 0.05)
                                reinforced.append(existing_mem)
                        if reinforced:
                            await self._adapter.batch_update(reinforced)
                    await self._adapter.create(mem)
                    stored.append(mem)
                logger.info(f"[ingest:{session_id}] Stored {len(stored)} memories, hot total now: {len(await self._adapter.all_hot())}")

        # --- Raw turn storage (modes: raw, hybrid) ---
        if mode in ("raw", "hybrid"):
            raw_memories = self._extractor.extract_raw_turns(
                conversation_text, session_id, now,
            )
            if raw_memories:
                raw_contents = [m.content for m in raw_memories]
                raw_embeddings = self._embedder.embed_batch(raw_contents)
                for mem, emb in zip(raw_memories, raw_embeddings):
                    mem.embedding = emb
                    await self._adapter.create(mem)
                    stored.append(mem)

        if not stored:
            return []

        # Synaptic tagging: link co-ingested memories gated by similarity
        if len(stored) > 1:
            for mem_a, mem_b in combinations(stored, 2):
                if mem_a.embedding is None or mem_b.embedding is None:
                    continue
                sim = cosine_similarity(mem_a.embedding, mem_b.embedding)
                if sim >= INGESTION_ASSOCIATION_THRESHOLD:
                    weight = min(0.5, INGESTION_ASSOCIATION_BASE_WEIGHT + (sim - INGESTION_ASSOCIATION_THRESHOLD) * 0.5)
                    _ensure_bidirectional_association(mem_a, mem_b, weight, now)
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
        """Run periodic maintenance: cold migration, TTL expiry, consolidation."""
        now = now or datetime.now()
        await self._engine.tick(now, self._embedder, self._extractor.compress_memories)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Return current memory system statistics."""
        now = datetime.now()
        all_mems = await self._adapter.all_active()

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
            "total_memories": await self._adapter.total_count(),
            "hot_memories": await self._adapter.hot_count(),
            "cold_memories": await self._adapter.cold_count(),
            "stub_memories": await self._adapter.stub_count(),
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
