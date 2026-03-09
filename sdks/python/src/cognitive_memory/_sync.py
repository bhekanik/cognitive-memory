"""
Synchronous wrapper around async CognitiveMemory.

Provides the same API but runs everything through asyncio.run().
Useful for scripts, notebooks, and benchmark compatibility.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional, Literal

from .types import Memory, MemoryCategory, CognitiveMemoryConfig, SearchResult, SearchResponse
from .adapters.base import MemoryAdapter
from .embeddings import EmbeddingProvider
from .core import CognitiveMemory


def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Jupyter).
        # Create a new loop in a thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class SyncCognitiveMemory:
    """
    Synchronous wrapper around CognitiveMemory.

    Usage:
        from cognitive_memory import SyncCognitiveMemory

        mem = SyncCognitiveMemory(embedder="hash")
        mem.add("User likes coffee", importance=0.5)
        results = mem.search("coffee")
    """

    def __init__(
        self,
        config: Optional[CognitiveMemoryConfig] = None,
        embedder: Optional[EmbeddingProvider | Literal["openai", "hash"]] = None,
        adapter: Optional[MemoryAdapter] = None,
    ):
        self._async = CognitiveMemory(
            config=config,
            embedder=embedder,
            adapter=adapter,
        )

    def add(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.EPISODIC,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Memory:
        return _run(self._async.add(
            content=content,
            category=category,
            importance=importance,
            session_id=session_id,
            timestamp=timestamp,
        ))

    def add_memory_object(self, memory: Memory) -> Memory:
        return _run(self._async.add_memory_object(memory))

    def extract_and_store(
        self,
        conversation_text: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
        run_tick: bool = True,
    ) -> list[Memory]:
        return _run(self._async.extract_and_store(
            conversation_text=conversation_text,
            session_id=session_id,
            timestamp=timestamp,
            run_tick=run_tick,
        ))

    def search(
        self,
        query: str,
        top_k: int = 10,
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None,
        deep_recall: bool = False,
        trace: bool = False,
    ) -> SearchResponse:
        return _run(self._async.search(
            query=query,
            top_k=top_k,
            timestamp=timestamp,
            session_id=session_id,
            deep_recall=deep_recall,
            trace=trace,
        ))

    def tick(self, now: Optional[datetime] = None):
        return _run(self._async.tick(now))

    def get_stats(self) -> dict:
        return _run(self._async.get_stats())

    def clear(self):
        return _run(self._async.clear())

    @property
    def adapter(self) -> MemoryAdapter:
        return self._async.adapter

    @property
    def engine(self):
        return self._async.engine

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._async.embedder

    @property
    def config(self) -> CognitiveMemoryConfig:
        return self._async.config
