"""
Adapter conformance tests.

These tests verify that any adapter implementation correctly implements
the MemoryAdapter ABC. Run against all registered adapters via the
parametrized `adapter` fixture.
"""

import pytest
from datetime import datetime
from cognitive_memory.types import Memory, MemoryCategory


@pytest.mark.asyncio
async def test_create_and_get(adapter):
    """Adapter can create and retrieve a memory."""
    mem = Memory(
        content="Test memory",
        category=MemoryCategory.EPISODIC,
        importance=0.5,
        stability=0.3,
        created_at=datetime(2024, 1, 1),
        last_accessed_at=datetime(2024, 1, 1),
        embedding=[0.1, 0.2, 0.3],
    )
    await adapter.create(mem)
    retrieved = await adapter.get(mem.id)
    assert retrieved is not None
    assert retrieved.content == "Test memory"
    assert retrieved.id == mem.id


@pytest.mark.asyncio
async def test_delete(adapter):
    """Adapter can delete a memory."""
    mem = Memory(content="To delete", created_at=datetime(2024, 1, 1))
    await adapter.create(mem)
    await adapter.delete(mem.id)
    assert await adapter.get(mem.id) is None


@pytest.mark.asyncio
async def test_counts(adapter):
    """Adapter reports correct counts."""
    assert await adapter.total_count() == 0
    assert await adapter.hot_count() == 0

    mem = Memory(content="Count test", created_at=datetime(2024, 1, 1))
    await adapter.create(mem)

    assert await adapter.total_count() == 1
    assert await adapter.hot_count() == 1


@pytest.mark.asyncio
async def test_clear(adapter):
    """Adapter can clear all data."""
    mem = Memory(content="Clear test", created_at=datetime(2024, 1, 1))
    await adapter.create(mem)
    assert await adapter.total_count() == 1

    await adapter.clear()
    assert await adapter.total_count() == 0


@pytest.mark.asyncio
async def test_search_similar(adapter):
    """Adapter can perform similarity search."""
    mem1 = Memory(
        content="Coffee lover",
        embedding=[1.0, 0.0, 0.0],
        created_at=datetime(2024, 1, 1),
    )
    mem2 = Memory(
        content="Tea drinker",
        embedding=[0.0, 1.0, 0.0],
        created_at=datetime(2024, 1, 1),
    )
    await adapter.create(mem1)
    await adapter.create(mem2)

    results = await adapter.search_similar([1.0, 0.0, 0.0], top_k=1)
    assert len(results) >= 1
    assert results[0][0].content == "Coffee lover"


@pytest.mark.asyncio
async def test_cold_migration(adapter):
    """Adapter can migrate memories between tiers."""
    mem = Memory(content="Migrate me", created_at=datetime(2024, 1, 1))
    await adapter.create(mem)

    assert await adapter.hot_count() == 1
    assert await adapter.cold_count() == 0

    await adapter.migrate_to_cold(mem.id, datetime(2024, 6, 1))

    assert await adapter.hot_count() == 0
    assert await adapter.cold_count() == 1

    await adapter.migrate_to_hot(mem.id)

    assert await adapter.hot_count() == 1
    assert await adapter.cold_count() == 0
