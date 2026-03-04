"""
CognitiveMemory - the main public API.

This is the class users interact with. It wires together the adapter,
engine, extractor, and embedder into a coherent interface.

All public methods are async. For sync usage, use SyncCognitiveMemory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Literal

from .types import (
    Memory,
    MemoryCategory,
    CognitiveMemoryConfig,
    SearchResult,
)
from .adapters.base import MemoryAdapter
from .adapters.memory import InMemoryAdapter
from .engine import CognitiveEngine
from .extraction import MemoryExtractor
from .embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HashEmbeddings,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


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

        # Resolve embedder
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

        # Check for conflicts with existing memories
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
        Extract memories from conversation text using LLM, embed them,
        check for conflicts, and store.
        """
        now = timestamp or datetime.now()

        # Extract via LLM
        memories = self._extractor.extract_from_conversation(
            conversation_text, session_id, now,
        )

        if not memories:
            return []

        # Batch embed
        contents = [m.content for m in memories]
        embeddings = self._embedder.embed_batch(contents)
        for mem, emb in zip(memories, embeddings):
            mem.embedding = emb

        # Check conflicts, boost similar existing memories, and store
        stored = []
        for mem in memories:
            await self._check_conflicts(mem, now)

            # Ingestion-time similarity boost: if this topic was mentioned
            # before, boost existing memory's stability (repeated exposure)
            if mem.embedding is not None:
                similar = await self._adapter.search_similar(mem.embedding, top_k=3)
                for existing_mem, sim in similar:
                    if sim > 0.75 and existing_mem.id != mem.id:
                        existing_mem.stability = min(1.0, existing_mem.stability + 0.05)

            await self._adapter.create(mem)
            stored.append(mem)

        # Synaptic tagging: create associations between memories from the
        # same session. In neuroscience, memories encoded close together in
        # time are linked so recalling one activates the others. We gate by
        # semantic similarity to avoid noise — only link memories that share
        # some topical overlap (sim > 0.4).
        from .embeddings import cosine_similarity as _cos_sim
        from .types import Association as _Assoc
        ingestion_assoc_threshold = 0.4
        ingestion_assoc_weight = 0.2
        for i in range(len(stored)):
            if stored[i].embedding is None:
                continue
            for j in range(i + 1, len(stored)):
                if stored[j].embedding is None:
                    continue
                sim = _cos_sim(stored[i].embedding, stored[j].embedding)
                if sim >= ingestion_assoc_threshold:
                    # Create bidirectional association
                    weight = min(0.5, ingestion_assoc_weight + (sim - ingestion_assoc_threshold) * 0.5)
                    if stored[j].id not in stored[i].associations:
                        stored[i].associations[stored[j].id] = _Assoc(
                            target_id=stored[j].id, weight=weight,
                            created_at=now, last_co_retrieval=now,
                        )
                    if stored[i].id not in stored[j].associations:
                        stored[j].associations[stored[i].id] = _Assoc(
                            target_id=stored[i].id, weight=weight,
                            created_at=now, last_co_retrieval=now,
                        )

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
    ) -> list[SearchResult]:
        """
        Search memories with full retention-weighted scoring.

        Args:
            query: natural language query
            top_k: max results
            timestamp: when the query happens (for decay calc). Defaults to now.
            session_id: current session (for boost tracking)
            deep_recall: include superseded originals (Section 3.8)
        """
        now = timestamp or datetime.now()
        query_embedding = self._embedder.embed(query)

        return await self._engine.search(
            query_embedding=query_embedding,
            now=now,
            top_k=top_k,
            session_id=session_id,
            deep_recall=deep_recall,
        )

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    async def _check_conflicts(self, new_memory: Memory, now: datetime):
        """
        Check if new memory conflicts with existing memories.
        On contradiction/update: demote the old memory.
        """
        # Only check against high-importance or core memories for efficiency
        all_hot = await self._adapter.all_hot()
        candidates = [
            m for m in all_hot
            if not m.is_superseded and not m.is_stub
            and (m.importance > 0.5 or m.category == MemoryCategory.CORE)
        ]

        if not candidates or new_memory.embedding is None:
            return

        # Find semantically similar existing memories
        for existing in candidates:
            if existing.embedding is None:
                continue
            sim = cosine_similarity(new_memory.embedding, existing.embedding)
            if sim < 0.6:  # only check high-similarity pairs
                continue

            conflict_type = self._extractor.detect_conflict(new_memory, existing)

            if conflict_type in ("CONTRADICTION", "UPDATE"):
                logger.info(
                    f"Conflict detected ({conflict_type}): "
                    f"'{existing.content[:50]}' -> '{new_memory.content[:50]}'"
                )
                # Demote the old memory
                if existing.category == MemoryCategory.CORE:
                    existing.category = MemoryCategory.SEMANTIC
                existing.contradicted_by = new_memory.id
                # New memory inherits importance
                new_memory.importance = max(new_memory.importance, existing.importance)
                if conflict_type == "CONTRADICTION" and existing.category == MemoryCategory.CORE:
                    new_memory.category = MemoryCategory.CORE

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
