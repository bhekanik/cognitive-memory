"""
JSONL file adapter — append-only event log + in-memory index.

Each mutating operation appends an event to the file. Construction reads the
file end-to-end and replays events into in-memory state. Reads are served
from the in-memory store (inherited from :class:`InMemoryAdapter`).

This adapter is the simplest durable backend that ships with the SDK; it has
no external dependencies and is well-suited to single-process workloads,
local development, and reproducible test fixtures.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

from ..types import Association, Memory, MemoryCategory
from .memory import InMemoryAdapter


def _serialize_memory(mem: Memory) -> dict:
    return {
        "id": mem.id,
        "user_id": mem.user_id,
        "content": mem.content,
        "category": mem.category.value,
        "importance": mem.importance,
        "stability": mem.stability,
        "access_count": mem.access_count,
        "last_accessed_at": mem.last_accessed_at.isoformat() if mem.last_accessed_at else None,
        "created_at": mem.created_at.isoformat() if mem.created_at else None,
        "embedding": mem.embedding,
        "session_ids": list(mem.session_ids),
        "is_cold": mem.is_cold,
        "cold_since": mem.cold_since.isoformat() if mem.cold_since else None,
        "days_at_floor": mem.days_at_floor,
        "is_superseded": mem.is_superseded,
        "superseded_by": mem.superseded_by,
        "contradicted_by": mem.contradicted_by,
        "is_stub": mem.is_stub,
        "memory_type": mem.memory_type,
        "valid_from": mem.valid_from.isoformat() if mem.valid_from else None,
        "valid_until": mem.valid_until.isoformat() if mem.valid_until else None,
        "ttl_seconds": mem.ttl_seconds,
        "source_turn_ids": list(mem.source_turn_ids),
    }


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _deserialize_memory(payload: dict) -> Memory:
    mem = Memory(
        id=payload["id"],
        user_id=payload.get("user_id", "default"),
        content=payload["content"],
        category=MemoryCategory(payload["category"]),
        importance=payload["importance"],
        stability=payload["stability"],
        access_count=payload.get("access_count", 0),
        last_accessed_at=_parse_dt(payload.get("last_accessed_at")),
        created_at=_parse_dt(payload.get("created_at")),
        embedding=payload.get("embedding"),
        is_cold=payload.get("is_cold", False),
        cold_since=_parse_dt(payload.get("cold_since")),
        days_at_floor=payload.get("days_at_floor", 0),
        is_superseded=payload.get("is_superseded", False),
        superseded_by=payload.get("superseded_by"),
        contradicted_by=payload.get("contradicted_by"),
        is_stub=payload.get("is_stub", False),
        memory_type=payload.get("memory_type", "other"),
        valid_from=_parse_dt(payload.get("valid_from")),
        valid_until=_parse_dt(payload.get("valid_until")),
        ttl_seconds=payload.get("ttl_seconds"),
        source_turn_ids=payload.get("source_turn_ids", []),
    )
    mem.session_ids = set(payload.get("session_ids", []))
    return mem


class JsonlFileAdapter(InMemoryAdapter):
    """File-backed adapter using an append-only JSONL event log.

    Construction loads and replays the file. All :class:`InMemoryAdapter`
    behaviour is inherited; mutating methods are wrapped to also write
    events to the log so the next process can rebuild identical state.
    """

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._loading = False  # suppress event writes during replay
        if os.path.exists(self._path):
            self._replay()
        else:
            # Touch the file so concurrent readers don't see "missing".
            with open(self._path, "a"):
                pass

    # ------------------------------------------------------------------
    # Replay / persistence
    # ------------------------------------------------------------------

    def _append(self, event: dict) -> None:
        if self._loading:
            return
        with open(self._path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _replay(self) -> None:
        self._loading = True
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._apply_event(event)
        finally:
            self._loading = False

    def _apply_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "create":
            mem = _deserialize_memory(event["memory"])
            if mem.is_stub:
                self.stubs[mem.id] = mem
            elif mem.is_cold:
                self.cold[mem.id] = mem
            else:
                self.hot[mem.id] = mem
        elif kind == "update":
            mem = _deserialize_memory(event["memory"])
            for tier in (self.hot, self.cold, self.stubs):
                if mem.id in tier:
                    tier[mem.id] = mem
                    break
        elif kind == "delete":
            self.hot.pop(event["id"], None)
            self.cold.pop(event["id"], None)
            self.stubs.pop(event["id"], None)
        elif kind == "migrate_to_cold":
            mem = self.hot.pop(event["id"], None)
            if mem is not None:
                mem.is_cold = True
                mem.cold_since = _parse_dt(event.get("cold_since"))
                self.cold[mem.id] = mem
        elif kind == "migrate_to_hot":
            mem = self.cold.pop(event["id"], None)
            if mem is not None:
                mem.is_cold = False
                mem.cold_since = None
                mem.days_at_floor = 0
                self.hot[mem.id] = mem
        elif kind == "convert_to_stub":
            mem_id = event["id"]
            existing = self.cold.pop(mem_id, None) or self.hot.pop(mem_id, None)
            if existing is not None:
                stub = Memory(
                    id=existing.id,
                    user_id=existing.user_id,
                    content=event["content"],
                    category=existing.category,
                    importance=existing.importance,
                    stability=0.0,
                    created_at=existing.created_at,
                    is_stub=True,
                    is_cold=False,
                    embedding=None,
                )
                self.stubs[mem_id] = stub
        elif kind == "link":
            key = self._link_key(event["a"], event["b"])
            existing = self._links.get(key)
            base = existing["weight"] if existing else 0.0
            self._links[key] = {
                "weight": min(1.0, base + event["weight"]),
                "link_type": event.get("link_type", "association"),
                "created_at": _parse_dt(event["created_at"]) or datetime.now(),
                "updated_at": _parse_dt(event["updated_at"]) or datetime.now(),
            }
        elif kind == "unlink":
            self._links.pop(self._link_key(event["a"], event["b"]), None)
        elif kind == "clear":
            self.hot.clear()
            self.cold.clear()
            self.stubs.clear()
            self._links.clear()
        # Unknown event types are skipped silently — lets us add new event
        # types without breaking older log files.

    # ------------------------------------------------------------------
    # Wrapped mutators
    # ------------------------------------------------------------------

    async def create(self, memory: Memory) -> None:
        await super().create(memory)
        self._append({"type": "create", "memory": _serialize_memory(memory)})

    async def update(self, memory: Memory) -> None:
        await super().update(memory)
        self._append({"type": "update", "memory": _serialize_memory(memory)})

    async def delete(self, memory_id: str) -> None:
        await super().delete(memory_id)
        self._append({"type": "delete", "id": memory_id})

    async def delete_batch(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            await self.delete(mid)

    async def batch_update(self, memories: list[Memory]) -> None:
        for mem in memories:
            await self.update(mem)

    async def migrate_to_cold(
        self, memory_id: str, cold_since: datetime
    ) -> None:
        await super().migrate_to_cold(memory_id, cold_since)
        self._append(
            {
                "type": "migrate_to_cold",
                "id": memory_id,
                "cold_since": cold_since.isoformat(),
            }
        )

    async def migrate_to_hot(self, memory_id: str) -> None:
        await super().migrate_to_hot(memory_id)
        self._append({"type": "migrate_to_hot", "id": memory_id})

    async def convert_to_stub(
        self, memory_id: str, stub_content: str
    ) -> None:
        await super().convert_to_stub(memory_id, stub_content)
        self._append(
            {
                "type": "convert_to_stub",
                "id": memory_id,
                "content": stub_content,
            }
        )

    async def create_or_strengthen_link(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        link_type: str = "association",
    ) -> None:
        await super().create_or_strengthen_link(
            source_id, target_id, weight, link_type
        )
        # Look up the canonical row to capture timestamps.
        row = self._links[self._link_key(source_id, target_id)]
        self._append(
            {
                "type": "link",
                "a": source_id,
                "b": target_id,
                "weight": weight,
                "link_type": link_type,
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
        )

    async def delete_link(self, source_id: str, target_id: str) -> None:
        await super().delete_link(source_id, target_id)
        self._append({"type": "unlink", "a": source_id, "b": target_id})

    async def mark_superseded(
        self, memory_ids: list[str], summary_id: str
    ) -> None:
        await super().mark_superseded(memory_ids, summary_id)
        # Re-emit affected memories as updates so replay reflects state.
        for mid in memory_ids:
            mem = await self.get(mid)
            if mem is not None:
                self._append(
                    {"type": "update", "memory": _serialize_memory(mem)}
                )

    async def clear(self) -> None:
        await super().clear()
        self._append({"type": "clear"})
