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


# ----------------------------------------------------------------------
# Adapter-level link table (spec contract: spec/adapter-interface.md:147-177)
# ----------------------------------------------------------------------


async def _seed_pair(adapter) -> tuple[Memory, Memory]:
    a = Memory(content="A", created_at=datetime(2024, 1, 1))
    b = Memory(content="B", created_at=datetime(2024, 1, 1))
    await adapter.create(a)
    await adapter.create(b)
    return a, b


@pytest.mark.asyncio
async def test_create_and_get_link(adapter):
    """create_or_strengthen_link persists a link that get_linked_memories surfaces."""
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.4)

    linked = await adapter.get_linked_memories(a.id, min_weight=0.0)
    ids = {m.id for m, _w in linked}
    assert b.id in ids
    weight = next(w for m, w in linked if m.id == b.id)
    assert weight == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_link_strengthens_additively_with_cap(adapter):
    """Strengthening an existing link adds weight, capped at 1.0.

    Matches the engine's expectation in retrieval-time co-retrieval boosting.
    The spec's prior wording ('max(existing, new)') is being corrected as part
    of this work; this test pins the current engine behaviour.
    """
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.4)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.3)

    linked = await adapter.get_linked_memories(a.id, min_weight=0.0)
    weight = next(w for m, w in linked if m.id == b.id)
    assert weight == pytest.approx(0.7)

    # Cap test: total beyond 1.0 saturates.
    await adapter.create_or_strengthen_link(a.id, b.id, 0.6)
    linked = await adapter.get_linked_memories(a.id, min_weight=0.0)
    weight = next(w for m, w in linked if m.id == b.id)
    assert weight == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_link_is_symmetric(adapter):
    """A link from a→b is also visible querying from b's side."""
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.5)

    from_b = await adapter.get_linked_memories(b.id, min_weight=0.0)
    assert any(m.id == a.id for m, _ in from_b)


@pytest.mark.asyncio
async def test_delete_link_is_idempotent(adapter):
    """delete_link removes the link; calling it again is a no-op."""
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.5)

    await adapter.delete_link(a.id, b.id)
    assert await adapter.get_linked_memories(a.id, min_weight=0.0) == []

    # Idempotent: second call must not raise.
    await adapter.delete_link(a.id, b.id)
    await adapter.delete_link("does-not-exist", "also-not-there")


@pytest.mark.asyncio
async def test_link_weight_filter(adapter):
    """get_linked_memories(min_weight=W) excludes links below W."""
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.2)

    above = await adapter.get_linked_memories(a.id, min_weight=0.0)
    below = await adapter.get_linked_memories(a.id, min_weight=0.5)

    assert any(m.id == b.id for m, _ in above)
    assert all(m.id != b.id for m, _ in below)


@pytest.mark.asyncio
async def test_clear_wipes_links(adapter):
    """clear() must drop all links, not just memories."""
    a, b = await _seed_pair(adapter)
    await adapter.create_or_strengthen_link(a.id, b.id, 0.5)

    await adapter.clear()

    # After clear, recreate the memories so get_linked_memories has something
    # to look up — and confirm there are no links pointing at it.
    await adapter.create(a)
    await adapter.create(b)
    assert await adapter.get_linked_memories(a.id, min_weight=0.0) == []
