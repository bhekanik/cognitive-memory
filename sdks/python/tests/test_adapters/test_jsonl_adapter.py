"""
JSONL file adapter — persistence via append-only event log.

Mirrors the TypeScript JsonlFileAdapter contract: every mutation appends an
event to disk; on construction, events replay into in-memory state. Closing
and reopening the same path must yield identical observable state.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cognitive_memory.types import Memory, MemoryCategory


def _make_memory(content: str, **kwargs) -> Memory:
    base = dict(
        content=content,
        user_id="alice",
        category=MemoryCategory.SEMANTIC,
        importance=0.5,
        stability=0.3,
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2026, 1, 1),
        last_accessed_at=datetime(2026, 1, 1),
    )
    base.update(kwargs)
    return Memory(**base)


@pytest.mark.asyncio
async def test_jsonl_persists_memories_across_reopen(tmp_path: Path):
    """Memories created against one adapter instance are visible to a
    second instance opened on the same file."""
    from cognitive_memory.adapters.jsonl import JsonlFileAdapter

    path = tmp_path / "store.jsonl"
    a = JsonlFileAdapter(str(path))
    mem = _make_memory("persist me")
    await a.create(mem)

    b = JsonlFileAdapter(str(path))
    restored = await b.get(mem.id)
    assert restored is not None
    assert restored.content == "persist me"
    assert restored.user_id == "alice"


@pytest.mark.asyncio
async def test_jsonl_persists_links_across_reopen(tmp_path: Path):
    """Adapter-level links round-trip through file replay."""
    from cognitive_memory.adapters.jsonl import JsonlFileAdapter

    path = tmp_path / "store.jsonl"
    a = JsonlFileAdapter(str(path))
    m1 = _make_memory("A")
    m2 = _make_memory("B")
    await a.create(m1)
    await a.create(m2)
    await a.create_or_strengthen_link(m1.id, m2.id, 0.5)

    b = JsonlFileAdapter(str(path))
    linked = await b.get_linked_memories(m1.id, min_weight=0.0)
    assert any(m.id == m2.id and w == pytest.approx(0.5) for m, w in linked)


@pytest.mark.asyncio
async def test_jsonl_persists_tier_changes_across_reopen(tmp_path: Path):
    """Cold migration + stub conversion are durable across reopens."""
    from cognitive_memory.adapters.jsonl import JsonlFileAdapter

    path = tmp_path / "store.jsonl"
    a = JsonlFileAdapter(str(path))
    m = _make_memory("cold soon")
    await a.create(m)
    await a.migrate_to_cold(m.id, datetime(2026, 2, 1))

    b = JsonlFileAdapter(str(path))
    assert await b.cold_count() == 1
    assert await b.hot_count() == 0


@pytest.mark.asyncio
async def test_jsonl_handles_delete(tmp_path: Path):
    """Deleted memories don't return after reopen."""
    from cognitive_memory.adapters.jsonl import JsonlFileAdapter

    path = tmp_path / "store.jsonl"
    a = JsonlFileAdapter(str(path))
    m = _make_memory("delete me")
    await a.create(m)
    await a.delete(m.id)

    b = JsonlFileAdapter(str(path))
    assert await b.get(m.id) is None
    assert await b.total_count() == 0


@pytest.mark.asyncio
async def test_jsonl_search_and_user_filter_work_after_reopen(tmp_path: Path):
    """Vector search and user_id filter are honoured after replay (the
    adapter is not just a record-keeper; queries must work)."""
    from cognitive_memory.adapters.jsonl import JsonlFileAdapter

    path = tmp_path / "store.jsonl"
    a = JsonlFileAdapter(str(path))
    await a.create(_make_memory("alice fact", user_id="alice"))
    await a.create(_make_memory("bob fact", user_id="bob"))

    b = JsonlFileAdapter(str(path))
    # Vector search restricted to alice should only return alice's memory.
    results = await b.search_similar(
        [0.1, 0.2, 0.3], top_k=5, user_id="alice"
    )
    contents = {m.content for m, _ in results}
    assert "alice fact" in contents
    assert "bob fact" not in contents
