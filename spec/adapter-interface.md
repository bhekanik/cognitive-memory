# Adapter Interface Specification

This document defines the canonical adapter contract that both the Python and TypeScript SDKs must implement. Any storage backend (SQLite, PostgreSQL, in-memory, etc.) must conform to this interface to be used with Cognitive Memory.

## Purpose

The adapter layer abstracts storage so the core engine never touches databases directly. Both SDKs must keep their adapters functionally equivalent — same methods, same semantics, same guarantees. This spec is the source of truth.

## Convention

- All methods are async.
- Python signatures use snake_case. TypeScript equivalents use camelCase.
- `Memory` refers to the SDK's memory object (see [memory-schema.md](./memory-schema.md)).
- `embedding` is a list/array of floats (the vector representation).
- IDs are strings (UUIDs).
- Timestamps are ISO 8601 strings or native datetime objects depending on the SDK.

---

## CRUD

### create

Store a new memory.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def create(self, memory: Memory) -> Memory` | `create(memory: Memory): Promise<Memory>` |
| **Args** | `memory` — fully populated Memory object including embedding | `memory` — fully populated Memory object including embedding |
| **Returns** | The stored Memory with any adapter-assigned fields (e.g. confirmed id) | The stored Memory |
| **Notes** | Must persist all fields. Must not overwrite an existing id. | Same. |

### get

Retrieve a single memory by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get(self, memory_id: str) -> Memory \| None` | `get(memoryId: string): Promise<Memory \| null>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Returns** | The Memory if found, `None`/`null` otherwise | Same |

### get_batch / getBatch

Retrieve multiple memories by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get_batch(self, memory_ids: list[str]) -> list[Memory]` | `getBatch(memoryIds: string[]): Promise<Memory[]>` |
| **Args** | `memory_ids` — list of UUIDs | `memoryIds` — array of UUIDs |
| **Returns** | List of found memories. Missing ids are silently omitted. | Same |

### update

Update an existing memory in place.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def update(self, memory: Memory) -> Memory` | `update(memory: Memory): Promise<Memory>` |
| **Args** | `memory` — Memory with updated fields. `id` must match an existing record. | Same |
| **Returns** | The updated Memory | Same |
| **Notes** | Must overwrite all fields. If memory does not exist, raise/throw. | Same |

### delete

Delete a single memory by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete(self, memory_id: str) -> None` | `delete(memoryId: string): Promise<void>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Returns** | Nothing. No-op if id does not exist. | Same |

### delete_batch / deleteBatch

Delete multiple memories by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete_batch(self, memory_ids: list[str]) -> None` | `deleteBatch(memoryIds: string[]): Promise<void>` |
| **Args** | `memory_ids` — list of UUIDs | `memoryIds` — array of UUIDs |
| **Returns** | Nothing. Missing ids are silently ignored. | Same |

---

## Vector Search

### search_similar / searchSimilar

Find memories closest to a given embedding vector.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def search_similar(self, embedding: list[float], top_k: int, include_superseded: bool = False) -> list[tuple[Memory, float]]` | `searchSimilar(embedding: number[], topK: number, includeSuperseded?: boolean): Promise<[Memory, number][]>` |
| **Args** | `embedding` — query vector. `top_k` — max results. `include_superseded` — if True, include superseded memories (for deep recall). | Same |
| **Returns** | List of `(Memory, similarity_score)` tuples, sorted by score descending. | Array of `[Memory, score]` tuples, sorted by score descending. |
| **Notes** | Similarity metric is cosine similarity. Superseded memories, if included, should be returned alongside active ones without special ordering. | Same |

---

## Tiering

### migrate_to_cold / migrateToCold

Move a memory from hot to cold storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def migrate_to_cold(self, memory_id: str) -> None` | `migrateToCold(memoryId: string): Promise<void>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Notes** | Sets `is_cold = True` and `cold_since` to current timestamp. | Sets equivalent fields. |

### migrate_to_hot / migrateToHot

Promote a memory from cold back to hot storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def migrate_to_hot(self, memory_id: str) -> None` | `migrateToHot(memoryId: string): Promise<void>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Notes** | Sets `is_cold = False` and clears `cold_since`. | Sets equivalent fields. |

### convert_to_stub / convertToStub

Convert a memory to a lightweight stub (drops embedding and most metadata).

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def convert_to_stub(self, memory_id: str) -> None` | `convertToStub(memoryId: string): Promise<void>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Notes** | Sets `is_stub = True`. Clears embedding. Retains id, content summary, and link references. | Same |

---

## Links

### create_or_strengthen_link / createOrStrengthenLink

Create a weighted association between two memories, or strengthen an existing one.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def create_or_strengthen_link(self, source_id: str, target_id: str, weight: float, link_type: str = "association") -> None` | `createOrStrengthenLink(sourceId: string, targetId: string, weight: number, linkType?: string): Promise<void>` |
| **Args** | `source_id`, `target_id` — UUIDs. `weight` — float 0..1. `link_type` — category of relationship. | Same |
| **Notes** | If a link already exists between source and target, set weight to `max(existing, new)`. Links are directional. | Same |

### get_linked_memories / getLinkedMemories

Retrieve all memories linked to a given memory.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get_linked_memories(self, memory_id: str) -> list[tuple[Memory, float, str]]` | `getLinkedMemories(memoryId: string): Promise<[Memory, number, string][]>` |
| **Args** | `memory_id` — UUID | `memoryId` — UUID |
| **Returns** | List of `(Memory, weight, link_type)` tuples. | Array of `[Memory, weight, linkType]` tuples. |

### delete_link / deleteLink

Remove a link between two memories.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete_link(self, source_id: str, target_id: str) -> None` | `deleteLink(sourceId: string, targetId: string): Promise<void>` |
| **Args** | `source_id`, `target_id` — UUIDs | `sourceId`, `targetId` — UUIDs |
| **Returns** | Nothing. No-op if link does not exist. | Same |

---

## Consolidation

### find_fading / findFading

Find memories whose retention has dropped below a threshold.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def find_fading(self, threshold: float) -> list[Memory]` | `findFading(threshold: number): Promise<Memory[]>` |
| **Args** | `threshold` — retention score cutoff (e.g. 0.1) | Same |
| **Returns** | All active (non-stub, non-superseded) memories with retention below threshold. | Same |

### find_stable / findStable

Find memories that have remained above a retention threshold consistently.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def find_stable(self, threshold: float, min_access_count: int) -> list[Memory]` | `findStable(threshold: number, minAccessCount: number): Promise<Memory[]>` |
| **Args** | `threshold` — minimum retention. `min_access_count` — minimum times accessed. | Same |
| **Returns** | Memories meeting both criteria. Candidates for core promotion. | Same |

### mark_superseded / markSuperseded

Mark a memory as superseded by another (e.g. after contradiction resolution).

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def mark_superseded(self, memory_id: str, superseded_by: str) -> None` | `markSuperseded(memoryId: string, supersededBy: string): Promise<void>` |
| **Args** | `memory_id` — the old memory. `superseded_by` — the new memory that replaces it. | Same |
| **Notes** | Sets `is_superseded = True` and `superseded_by` field. Does not delete the old memory. | Same |

---

## Traversal

### all_active / allActive

Retrieve all non-superseded, non-stub memories.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_active(self) -> list[Memory]` | `allActive(): Promise<Memory[]>` |
| **Returns** | All memories where `is_superseded = False` and `is_stub = False`. | Same |

### all_hot / allHot

Retrieve all memories currently in hot storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_hot(self) -> list[Memory]` | `allHot(): Promise<Memory[]>` |
| **Returns** | All memories where `is_cold = False` and `is_stub = False`. | Same |

### all_cold / allCold

Retrieve all memories currently in cold storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_cold(self) -> list[Memory]` | `allCold(): Promise<Memory[]>` |
| **Returns** | All memories where `is_cold = True`. | Same |

---

## Counts

### hot_count / hotCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def hot_count(self) -> int` | `hotCount(): Promise<number>` |
| **Returns** | Number of hot (non-cold, non-stub) memories. | Same |

### cold_count / coldCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def cold_count(self) -> int` | `coldCount(): Promise<number>` |
| **Returns** | Number of cold memories. | Same |

### stub_count / stubCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def stub_count(self) -> int` | `stubCount(): Promise<number>` |
| **Returns** | Number of stub memories. | Same |

### total_count / totalCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def total_count(self) -> int` | `totalCount(): Promise<number>` |
| **Returns** | Total number of memories across all tiers and states. | Same |

---

## Batch Operations

### batch_update / batchUpdate

Update multiple memories in a single operation.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def batch_update(self, memories: list[Memory]) -> None` | `batchUpdate(memories: Memory[]): Promise<void>` |
| **Args** | `memories` — list of Memory objects with updated fields | Same |
| **Notes** | Should be atomic where the backend supports it. All-or-nothing preferred. | Same |

### update_retention_scores / updateRetentionScores

Bulk-update retention scores (used during decay passes).

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def update_retention_scores(self, updates: list[tuple[str, float]]) -> None` | `updateRetentionScores(updates: [string, number][]): Promise<void>` |
| **Args** | `updates` — list of `(memory_id, new_retention)` pairs | Same |
| **Notes** | Optimized for bulk writes. Should not trigger full Memory serialization. | Same |

---

## Transactions

### transaction

Execute a callback within a transaction. If the callback raises/throws, the transaction is rolled back.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def transaction(self, callback: Callable[[Adapter], Awaitable[T]]) -> T` | `transaction<T>(callback: (adapter: Adapter) => Promise<T>): Promise<T>` |
| **Args** | `callback` — async function that receives a transactional adapter instance | Same |
| **Returns** | The return value of the callback | Same |
| **Notes** | The adapter passed to the callback must use the same connection/transaction. If the backend does not support transactions, execute the callback directly (best-effort). | Same |

---

## Clear

### clear

Delete all memories and links. Used for testing and reset.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def clear(self) -> None` | `clear(): Promise<void>` |
| **Returns** | Nothing. | Same |
| **Notes** | Irreversible. Drops all data including stubs, cold memories, and links. | Same |

---

## Implementation Notes

1. **Error handling** — Adapters should raise/throw typed errors: `MemoryNotFoundError` for missing ids on update, `AdapterError` for connection failures.
2. **Thread safety** — Adapters must be safe for concurrent use. Connection pooling is recommended for database backends.
3. **Embedding storage** — Embeddings are stored as arrays of floats. Database adapters should use native vector types where available (pgvector for PostgreSQL) and fall back to serialized JSON otherwise.
4. **Link storage** — Links are stored separately from memories. The schema is: `(source_id, target_id, weight, link_type, created_at)`.
5. **Idempotency** — `delete`, `delete_batch`, and `delete_link` must be idempotent (no-op if target does not exist). `create` must reject duplicate ids.
