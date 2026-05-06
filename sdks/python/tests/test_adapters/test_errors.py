"""
Adapters must raise typed errors on contract violations (spec line 332).

- MemoryNotFoundError when update() targets a missing id.
- AdapterError as the umbrella class for connection / backend failures
  (covered by the type, not a behaviour test in the in-memory adapter).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cognitive_memory.types import Memory


@pytest.mark.asyncio
async def test_update_missing_id_raises_memory_not_found(adapter):
    """update() of an unknown id must raise MemoryNotFoundError, not return
    silently. Callers depend on this to detect stale references."""
    from cognitive_memory.adapters import MemoryNotFoundError

    ghost = Memory(content="ghost", created_at=datetime(2026, 1, 1))
    with pytest.raises(MemoryNotFoundError):
        await adapter.update(ghost)


@pytest.mark.asyncio
async def test_memory_not_found_error_carries_the_id(adapter):
    """The exception message must include the missing id for debuggability."""
    from cognitive_memory.adapters import MemoryNotFoundError

    ghost = Memory(content="ghost", created_at=datetime(2026, 1, 1))
    with pytest.raises(MemoryNotFoundError) as excinfo:
        await adapter.update(ghost)
    assert ghost.id in str(excinfo.value)


def test_typed_errors_share_a_base_class():
    """MemoryNotFoundError must inherit from AdapterError so callers can
    catch all backend issues with a single except clause."""
    from cognitive_memory.adapters import AdapterError, MemoryNotFoundError

    assert issubclass(MemoryNotFoundError, AdapterError)


@pytest.mark.asyncio
async def test_create_rejects_duplicate_id(adapter):
    """Per spec Implementation Note 5, create() must reject a memory whose
    id already exists in any tier — silent overwrite would corrupt the
    audit trail (e.g. accidentally clobber a contradicted memory)."""
    from cognitive_memory.adapters import AdapterError

    mem = Memory(content="original", created_at=datetime(2026, 1, 1))
    await adapter.create(mem)

    duplicate = Memory(
        id=mem.id, content="impostor", created_at=datetime(2026, 1, 1)
    )
    with pytest.raises(AdapterError):
        await adapter.create(duplicate)

    # The original is intact (the duplicate didn't slip through):
    restored = await adapter.get(mem.id)
    assert restored is not None
    assert restored.content == "original"
