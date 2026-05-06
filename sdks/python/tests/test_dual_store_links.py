"""
Tests for the dual-store association invariant.

The Python SDK keeps links in two places:
  1. ``Memory.associations`` — per-memory cache, used at retrieval-time for
     weight decay (90-day exponential).
  2. The adapter-level link table (spec contract) — durable backend store.

Both stores must stay in sync after every ingestion and every co-retrieval
strengthening, so a Postgres-backed adapter (item 6) sees the same edges as
the in-memory cache.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pytest

from cognitive_memory import (
    CognitiveMemory,
    CognitiveMemoryConfig,
    EmbeddingProvider,
    InMemoryAdapter,
)
from cognitive_memory.llm import LLMProvider, LLMUsage


class TwoMemoryLLM(LLMProvider):
    """Returns two extracted memories with identical embeddings (sim = 1.0)."""

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps(
            [
                {"content": "alpha", "category": "semantic", "importance": 0.5},
                {"content": "beta", "category": "semantic", "importance": 0.5},
            ]
        )

    def complete_with_usage(
        self, prompt: str, **kwargs
    ) -> tuple[str, LLMUsage]:
        return self.complete(prompt), {}


class FixedVectorEmbedder(EmbeddingProvider):
    """Returns the same embedding for every input — guarantees sim=1.0
    between any pair of extracted memories so synaptic tagging fires."""

    @property
    def dimensions(self) -> int:
        return 4

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_ingestion_writes_link_to_adapter_store():
    """After extract_and_store ingests two similar memories, the adapter-level
    link table reflects the bidirectional link, not just Memory.associations."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)
    mem = CognitiveMemory(
        config=config, adapter=adapter, embedder=FixedVectorEmbedder(), llm=TwoMemoryLLM()
    )

    stored = await mem.extract_and_store(
        "User: alpha. User: beta.",
        session_id="s1",
        timestamp=datetime(2026, 1, 1),
    )
    assert len(stored) == 2

    # Wipe the per-memory cache to prove the adapter store alone holds the
    # link. If the adapter were still relying on Memory.associations, this
    # assertion would fail.
    for m in stored:
        m.associations.clear()
    await adapter.batch_update(stored)

    linked = await adapter.get_linked_memories(stored[0].id, min_weight=0.0)
    target_ids = {m.id for m, _ in linked}
    assert stored[1].id in target_ids


@pytest.mark.asyncio
async def test_ingestion_link_weight_matches_synaptic_formula():
    """Verifies the spec formula `min(0.5, 0.2 + (sim - 0.4) * 0.5)` is
    applied at ingestion. With sim = 1.0, weight = 0.5."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)
    mem = CognitiveMemory(
        config=config, adapter=adapter, embedder=FixedVectorEmbedder(), llm=TwoMemoryLLM()
    )

    stored = await mem.extract_and_store(
        "User: alpha. User: beta.",
        session_id="s1",
        timestamp=datetime(2026, 1, 1),
    )

    linked = await adapter.get_linked_memories(stored[0].id, min_weight=0.0)
    weight = next(w for m, w in linked if m.id == stored[1].id)
    assert weight == pytest.approx(0.5)
