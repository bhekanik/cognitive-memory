"""
Typed adapter errors.

Per spec/adapter-interface.md Implementation Note 1: adapters should raise
typed errors rather than generic exceptions, so callers can catch backend
issues precisely.
"""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for all adapter-level failures.

    Use this as the catch-all for connection drops, transaction conflicts,
    schema-mismatch errors, and any other backend issue surfacing from a
    storage adapter.
    """


class MemoryNotFoundError(AdapterError):
    """Raised when an operation references a memory id that does not exist.

    Distinct from a normal "not found" lookup — this is an *operation* error
    (e.g. update or migrate against a stale id), not a query result. Adapters
    that silently no-op on missing ids must wrap such cases here when the
    caller's intent requires the id to exist.
    """

    def __init__(self, memory_id: str):
        super().__init__(f"Memory not found: {memory_id}")
        self.memory_id = memory_id


class DuplicateMemoryError(AdapterError):
    """Raised when create() is called with an id that already exists.

    Silent overwrite would corrupt the audit trail — e.g. a contradicted
    memory could be accidentally clobbered by a re-ingest of the same id.
    """

    def __init__(self, memory_id: str):
        super().__init__(f"Memory already exists: {memory_id}")
        self.memory_id = memory_id
