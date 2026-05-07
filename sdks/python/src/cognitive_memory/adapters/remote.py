"""Cognitive Memory System - Remote Adapter (Python).

Talks to a ``cm-daemon`` over a Unix domain socket using length-delimited
JSON. Implements the full ``MemoryAdapter`` abstract base — drop-in
replacement for the in-process adapters when the user wants the daemon
deployment shape.

Plus paper-faithful extras (``create_batch`` for co-creation associations,
``mint_bridge_token`` for cm-http) that aren't on the base class.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ..types import Association, Memory, MemoryCategory
from .base import MemoryAdapter

IPC_PROTOCOL_VERSION = 1

T = TypeVar("T")


class RemoteAdapterError(RuntimeError):
    """Connection / protocol failure talking to the cognitive-memory daemon."""


def _default_socket_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "cognitive-memory" / "cm.sock"
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "cognitive-memory" / "cm.sock"
    import tempfile

    return Path(tempfile.gettempdir()) / "cognitive-memory" / "cm.sock"


def _to_unix(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _from_unix(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _wire_to_memory(d: dict[str, Any]) -> Memory:
    """Convert a daemon wire ``MemoryData`` dict into a typed ``Memory``."""
    metadata: dict[str, Any] = {}
    if d.get("metadata"):
        try:
            metadata = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    m = Memory(
        id=d["id"],
        user_id=d["user_id"],
        content=d["content"],
        category=MemoryCategory(d["category"]),
        importance=d.get("importance", 0.0),
        stability=d.get("stability", 0.5),
        access_count=d.get("retrieval_count", 0),
        last_accessed_at=_from_unix(d.get("last_accessed_at")),
        created_at=_from_unix(d.get("created_at")),
        embedding=None,
        is_cold=d.get("is_cold", False),
        cold_since=_from_unix(d.get("cold_since")),
        days_at_floor=0,
        is_superseded=d.get("is_superseded", False),
        superseded_by=d.get("superseded_by"),
        contradicted_by=None,
        is_stub=d.get("is_stub", False),
        memory_type=d.get("memory_type", "fact"),
        valid_from=_from_unix(d.get("valid_from")),
        valid_until=_from_unix(d.get("valid_until")),
        ttl_seconds=None,
    )
    # Memory carries a metadata field via a side-channel (not declared in
    # types.py) — set if present so downstream consumers can read it.
    if metadata:
        try:
            m.metadata = metadata  # type: ignore[attr-defined]
        except Exception:
            pass
    return m


class RemoteAdapter(MemoryAdapter):
    """``MemoryAdapter`` that talks to a running ``cm-daemon``.

    The daemon enforces tenancy via ``user_id``. This adapter is scoped to
    a single ``user_id`` per connection; methods that take a ``user_id``
    kwarg must match (or omit, in which case the constructor's value is used).
    """

    def __init__(
        self,
        user_id: str = "default",
        socket_path: Path | None = None,
        client_label: str = "cognitive-memory-sdk-py",
    ) -> None:
        self._user_id = user_id
        self._socket_path = Path(socket_path) if socket_path else _default_socket_path()
        self._client_label = client_label
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()

    # -- Connection lifecycle --

    async def connect(self) -> None:
        if self._writer is not None:
            return
        try:
            reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
        except OSError as e:
            raise RemoteAdapterError(
                f"failed to connect to daemon at {self._socket_path}: {e}"
            ) from e
        await self._write_frame(writer, {
            "kind": "Hello",
            "client": self._client_label,
            "protocol_version": IPC_PROTOCOL_VERSION,
            "user_id": self._user_id,
        })
        welcome = await self._read_frame(reader)
        if welcome.get("protocol_version") != IPC_PROTOCOL_VERSION:
            writer.close()
            await writer.wait_closed()
            raise RemoteAdapterError(
                f"daemon protocol mismatch: client v{IPC_PROTOCOL_VERSION}, "
                f"daemon v{welcome.get('protocol_version')}"
            )
        self._reader = reader
        self._writer = writer

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    @staticmethod
    async def _write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        header = struct.pack(">I", len(body))
        writer.write(header + body)
        await writer.drain()

    @staticmethod
    async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
        header = await reader.readexactly(4)
        (length,) = struct.unpack(">I", header)
        body = await reader.readexactly(length)
        return json.loads(body.decode("utf-8"))

    async def _send(self, request: dict[str, Any]) -> dict[str, Any]:
        await self.connect()
        if self._writer is None or self._reader is None:
            raise RemoteAdapterError("not connected")
        async with self._lock:
            request_id = self._next_id
            self._next_id += 1
            await self._write_frame(self._writer, {
                "id": request_id,
                "payload": {"kind": "Request", "body": request},
            })
            reply = await self._read_frame(self._reader)
            if reply.get("id") != request_id:
                raise RemoteAdapterError(
                    f"id mismatch: expected {request_id}, got {reply.get('id')}"
                )
            payload = reply.get("payload", {})
            if payload.get("kind") != "Response":
                raise RemoteAdapterError(f"unexpected payload kind: {payload.get('kind')}")
            response = payload.get("body", {})
            if not response.get("ok", False):
                err = response.get("error", {})
                raise RemoteAdapterError(
                    f"daemon error ({err.get('kind', '?')}): {err.get('message', '?')}"
                )
            return response

    def _data(self, response: dict[str, Any], expected_kind: str | None = None) -> dict[str, Any]:
        data = response.get("data") or {}
        if expected_kind and data.get("kind") != expected_kind:
            raise RemoteAdapterError(f"expected {expected_kind}, got {data.get('kind')}")
        return data

    def _check_user(self, user_id: Optional[str]) -> str:
        if user_id is not None and user_id != self._user_id:
            raise RemoteAdapterError(
                f"adapter scoped to {self._user_id!r}, can't operate on {user_id!r}"
            )
        return self._user_id

    # =====================================================================
    # MemoryAdapter abstract methods
    # =====================================================================

    # -- CRUD --

    async def create(self, memory: Memory) -> None:
        """Persist a memory and write the daemon-assigned id back onto
        ``memory.id``.

        The daemon issues its own ULID at storage time (it does not accept
        client-supplied ids in protocol v1). Without this write-back the
        local ``Memory`` object would drift from the persisted record and
        subsequent ``cm.delete(memory.id)`` calls would silently miss.
        """
        self._check_user(memory.user_id)
        metadata = getattr(memory, "metadata", {}) or {}
        resp = await self._send({
            "bucket": "Memory", "op": "Store", "user_id": memory.user_id,
            "content": memory.content,
            "category": memory.category.value,
            "memory_type": memory.memory_type,
            "metadata": json.dumps(metadata),
        })
        data = self._data(resp, "MemoryStored")
        memory.id = data["id"]

    async def get(self, memory_id: str) -> Optional[Memory]:
        try:
            resp = await self._send({
                "bucket": "Memory", "op": "Get",
                "user_id": self._user_id, "id": memory_id,
            })
        except RemoteAdapterError as e:
            if "NotFound" in str(e):
                return None
            raise
        return _wire_to_memory(self._data(resp, "Memory"))

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        resp = await self._send({
            "bucket": "Memory", "op": "GetMany",
            "user_id": self._user_id, "ids": memory_ids,
        })
        return [_wire_to_memory(m) for m in self._data(resp, "Memories").get("memories", [])]

    async def update(self, memory: Memory) -> None:
        self._check_user(memory.user_id)
        metadata = getattr(memory, "metadata", {}) or {}
        await self._send({
            "bucket": "Memory", "op": "Update",
            "user_id": memory.user_id, "id": memory.id,
            "content": memory.content,
            "category": memory.category.value,
            "memory_type": memory.memory_type,
            "metadata": json.dumps(metadata),
            "importance": memory.importance,
            "stability": memory.stability,
            "valid_until": _to_unix(memory.valid_until),
        })

    async def delete(self, memory_id: str) -> None:
        await self._send({
            "bucket": "Memory", "op": "Delete",
            "user_id": self._user_id, "id": memory_id,
        })

    async def delete_batch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        await self._send({
            "bucket": "Memory", "op": "DeleteMany",
            "user_id": self._user_id, "ids": memory_ids,
        })

    # -- Vector search --

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_superseded: bool = False,
        include_cold: bool = False,
        include_stubs: bool = False,
        user_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        uid = self._check_user(user_id)
        resp = await self._send({
            "bucket": "Memory", "op": "VectorSearch", "user_id": uid,
            "embedding": list(query_embedding),
            "embedding_provider": "local",
            "embedding_model": "bge-small-en-v1.5",
            "limit": top_k,
            "deep_recall": include_superseded or include_cold or include_stubs,
        })
        results = self._data(resp, "MemorySearchResults").get("results", [])
        # The wire SearchHit only carries id/content/cat/type/score —
        # synthesise a minimal Memory and let callers refetch if they need
        # the full record.
        out: list[tuple[Memory, float]] = []
        for hit in results:
            m = Memory(
                id=hit["memory_id"],
                user_id=uid,
                content=hit["content"],
                category=MemoryCategory(hit["category"]),
                memory_type=hit["memory_type"],
            )
            out.append((m, hit["score"]))
        return out

    async def search_lexical(
        self,
        query: str,
        top_k: int = 10,
        include_superseded: bool = False,
        include_cold: bool = False,
        include_stubs: bool = False,
        user_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        uid = self._check_user(user_id)
        resp = await self._send({
            "bucket": "Memory", "op": "SearchLexical",
            "user_id": uid, "query": query, "limit": top_k,
        })
        ids = self._data(resp, "LexicalIds").get("ids", [])
        memories = await self.get_batch(ids)
        # Wire returns ids ranked by BM25; we synthesise a descending score.
        return [(m, float(len(memories) - i)) for i, m in enumerate(memories)]

    # -- Tiered storage --

    async def migrate_to_cold(self, memory_id: str, cold_since: "datetime") -> None:
        await self._send({
            "bucket": "Lifecycle", "op": "MigrateToCold",
            "user_id": self._user_id, "id": memory_id,
            "cold_since": _to_unix(cold_since) or 0,
        })

    async def migrate_to_hot(self, memory_id: str) -> None:
        await self._send({
            "bucket": "Lifecycle", "op": "MigrateToHot",
            "user_id": self._user_id, "id": memory_id,
        })

    async def convert_to_stub(self, memory_id: str, stub_content: str) -> None:
        await self._send({
            "bucket": "Lifecycle", "op": "ConvertToStub",
            "user_id": self._user_id, "id": memory_id,
            "stub_content": stub_content,
        })

    # -- Links --

    async def create_or_strengthen_link(
        self, source_id: str, target_id: str, weight: float,
    ) -> None:
        await self._send({
            "bucket": "Memory", "op": "Link",
            "user_id": self._user_id,
            "source_id": source_id, "target_id": target_id,
            "strength": weight, "bidirectional": True, "kind": "explicit",
        })

    async def get_linked_memories(
        self, memory_id: str, min_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        resp = await self._send({
            "bucket": "Memory", "op": "GetLinked",
            "user_id": self._user_id,
            "source_id": memory_id, "min_strength": min_weight,
        })
        rows = self._data(resp, "LinkedMemories").get("memories", [])
        return [(_wire_to_memory(r["memory"]), r["link_strength"]) for r in rows]

    async def delete_link(self, source_id: str, target_id: str) -> None:
        await self._send({
            "bucket": "Memory", "op": "Unlink",
            "user_id": self._user_id,
            "source_id": source_id, "target_id": target_id,
            "bidirectional": True,
        })

    # -- Consolidation helpers --

    async def find_fading(
        self, threshold: float, exclude_core: bool = True,
    ) -> list[Memory]:
        resp = await self._send({
            "bucket": "Lifecycle", "op": "FindFading",
            "user_id": self._user_id,
            "max_retention": threshold, "limit": 100,
        })
        out = [_wire_to_memory(m) for m in self._data(resp, "Memories").get("memories", [])]
        if exclude_core:
            out = [m for m in out if m.category != MemoryCategory.CORE]
        return out

    async def find_stable(
        self, min_stability: float, min_access_count: int,
    ) -> list[Memory]:
        resp = await self._send({
            "bucket": "Lifecycle", "op": "FindStable",
            "user_id": self._user_id,
            "min_stability": min_stability,
            "min_access_count": min_access_count,
            "limit": 100,
        })
        return [_wire_to_memory(m) for m in self._data(resp, "Memories").get("memories", [])]

    async def mark_superseded(
        self, memory_ids: list[str], summary_id: str,
    ) -> None:
        if not memory_ids:
            return
        await self._send({
            "bucket": "Lifecycle", "op": "MarkSuperseded",
            "user_id": self._user_id,
            "ids": memory_ids, "summary_id": summary_id,
        })

    # -- Traversal --

    async def all_active(self, user_id: Optional[str] = None) -> list[Memory]:
        uid = self._check_user(user_id)
        return await self._list({"user_id": uid, "include_cold": True, "include_stubs": False})

    async def all_hot(self, user_id: Optional[str] = None) -> list[Memory]:
        uid = self._check_user(user_id)
        return await self._list({"user_id": uid, "include_cold": False, "include_stubs": False})

    async def all_cold(self, user_id: Optional[str] = None) -> list[Memory]:
        uid = self._check_user(user_id)
        out = await self._list({"user_id": uid, "include_cold": True, "include_stubs": False})
        return [m for m in out if m.is_cold]

    async def _list(self, payload: dict[str, Any]) -> list[Memory]:
        body = {
            "bucket": "Memory", "op": "List",
            "include_superseded": False,
            "include_cold": False,
            "include_stubs": False,
            "limit": 1000,
        }
        body.update(payload)
        resp = await self._send(body)
        return [_wire_to_memory(m) for m in self._data(resp, "Memories").get("memories", [])]

    # -- Counts --

    async def hot_count(self, user_id: Optional[str] = None) -> int:
        uid = self._check_user(user_id)
        return (await self._counts(uid))["hot"]

    async def cold_count(self, user_id: Optional[str] = None) -> int:
        uid = self._check_user(user_id)
        return (await self._counts(uid))["cold"]

    async def stub_count(self, user_id: Optional[str] = None) -> int:
        uid = self._check_user(user_id)
        return (await self._counts(uid))["stub"]

    async def total_count(self, user_id: Optional[str] = None) -> int:
        uid = self._check_user(user_id)
        return (await self._counts(uid))["total"]

    async def _counts(self, user_id: str) -> dict[str, int]:
        resp = await self._send({"bucket": "Diagnostics", "op": "Counts", "user_id": user_id})
        return self._data(resp, "Counts")

    # -- Batch --

    async def batch_update(self, memories: list[Memory]) -> None:
        # The daemon's BatchUpdate is retention-only; for richer fields we
        # fall back to N serial Updates (same default as the base class).
        for m in memories:
            await self.update(m)

    async def update_retention_scores(self, updates: dict[str, float]) -> None:
        if not updates:
            return
        await self._send({
            "bucket": "Memory", "op": "BatchUpdate",
            "user_id": self._user_id,
            "updates": [
                {"id": id_, "retention_floor": floor}
                for id_, floor in updates.items()
            ],
        })

    # -- Transaction --

    async def transaction(self, callback: Callable[["MemoryAdapter"], Awaitable[T]]) -> T:
        # No daemon-side transaction primitive in v1; run the callback
        # against this adapter (no isolation guarantees beyond per-request
        # atomicity).
        return await callback(self)

    # -- Reset --

    async def clear(self) -> None:
        await self._send({
            "bucket": "Lifecycle", "op": "Clear",
            "user_id": self._user_id, "confirm": True,
        })

    # =====================================================================
    # Daemon-only extras (not on the abstract base)
    # =====================================================================

    async def create_batch(
        self,
        memories: list[Memory],
        initial_link_weight: float = 0.5,
    ) -> dict[str, Any]:
        """Paper-faithful batch storage with co-creation associations
        (paper §3.6). Returns ``{ids, associations_created}``.
        """
        for m in memories:
            self._check_user(m.user_id)
        resp = await self._send({
            "bucket": "Memory", "op": "StoreBatch",
            "user_id": self._user_id,
            "memories": [
                {
                    "content": m.content,
                    "category": m.category.value,
                    "memory_type": m.memory_type,
                    "metadata": json.dumps(getattr(m, "metadata", {}) or {}),
                }
                for m in memories
            ],
            "initial_link_weight": initial_link_weight,
        })
        data = self._data(resp, "MemoryStoredBatch")
        return {"ids": data["ids"], "associations_created": data["associations_created"]}

    async def mint_bridge_token(
        self, scope: str = "write", ttl_seconds: int = 30 * 24 * 3600,
    ) -> dict[str, Any]:
        resp = await self._send({
            "bucket": "Diagnostics", "op": "MintBridgeToken",
            "user_id": self._user_id, "scope": scope, "ttl_seconds": ttl_seconds,
        })
        data = self._data(resp, "BridgeToken")
        return {"token": data["token"], "expires_at_unix": data["expires_at_unix"]}
