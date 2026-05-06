"""
Tests for the Python SDK's multi-tenancy contract.

Two CognitiveMemory instances scoped to different `user_id`s sharing the same
storage adapter must not bleed memories across users. Mirrors the TS SDK's
mandatory userId — see spec/memory-schema.md.
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
    MemoryCategory,
)
from cognitive_memory.llm import LLMProvider, LLMUsage


class IdentityEmbedder(EmbeddingProvider):
    """A deterministic embedder so tests don't hit the network."""

    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, text: str) -> list[float]:
        # Use a tiny content-derived hash so different texts get different vectors.
        v = [0.0] * 8
        for i, ch in enumerate(text):
            v[i % 8] += float(ord(ch) % 17) / 17.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class StubExtractLLM(LLMProvider):
    def __init__(self, content: str):
        self._content = content

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps(
            [
                {
                    "content": self._content,
                    "category": "semantic",
                    "importance": 0.5,
                    "memory_type": "fact",
                }
            ]
        )

    def complete_with_usage(self, prompt: str, **kwargs) -> tuple[str, LLMUsage]:
        return self.complete(prompt), {}


@pytest.mark.asyncio
async def test_search_does_not_leak_across_users():
    """Alice and Bob share an adapter; Alice's search must not surface Bob's
    memories even when their content is identical."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)

    alice = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=IdentityEmbedder(),
        llm=StubExtractLLM("alice's secret"),
        user_id="alice",
    )
    bob = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=IdentityEmbedder(),
        llm=StubExtractLLM("bob's secret"),
        user_id="bob",
    )

    await alice.add(
        "alice's secret",
        category=MemoryCategory.SEMANTIC,
        timestamp=datetime(2026, 1, 1),
    )
    await bob.add(
        "bob's secret",
        category=MemoryCategory.SEMANTIC,
        timestamp=datetime(2026, 1, 1),
    )

    alice_results = await alice.search("secret", timestamp=datetime(2026, 1, 2))
    bob_results = await bob.search("secret", timestamp=datetime(2026, 1, 2))

    alice_contents = {r.memory.content for r in alice_results.results}
    bob_contents = {r.memory.content for r in bob_results.results}

    assert "alice's secret" in alice_contents
    assert "bob's secret" not in alice_contents
    assert "bob's secret" in bob_contents
    assert "alice's secret" not in bob_contents


@pytest.mark.asyncio
async def test_default_user_id_preserves_back_compat():
    """Constructing CognitiveMemory without user_id places memories under
    'default' so existing callers (like the benchmarks) keep working."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)

    mem = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=IdentityEmbedder(),
        llm=StubExtractLLM("hello"),
    )
    stored = await mem.add("hello", timestamp=datetime(2026, 1, 1))

    assert stored.user_id == "default"


@pytest.mark.asyncio
async def test_get_stats_is_per_user():
    """get_stats() must report only the user's own memories."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)

    alice = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=IdentityEmbedder(),
        llm=StubExtractLLM("a"),
        user_id="alice",
    )
    bob = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=IdentityEmbedder(),
        llm=StubExtractLLM("b"),
        user_id="bob",
    )

    for i in range(3):
        await alice.add(f"alice-{i}", timestamp=datetime(2026, 1, 1))
    for i in range(2):
        await bob.add(f"bob-{i}", timestamp=datetime(2026, 1, 1))

    alice_stats = await alice.get_stats()
    bob_stats = await bob.get_stats()

    assert alice_stats["total_memories"] == 3
    assert bob_stats["total_memories"] == 2
