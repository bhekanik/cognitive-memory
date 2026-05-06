# Adapter Interface Specification

This document defines the canonical adapter contract that both the Python and TypeScript SDKs must implement. Any storage backend (SQLite, PostgreSQL, in-memory, etc.) must conform to this interface to be used with Cognitive Memory.

## Purpose

The adapter layer abstracts storage so the core engine never touches databases directly. Both SDKs must keep their adapters functionally equivalent — same methods, same semantics, same guarantees. This spec is the source of truth.

## Convention

- All methods are async.
- Python signatures use snake_case. TypeScript equivalents use camelCase, often with a verbose suffix (`createMemory` vs `create`, `findFadingMemories` vs `findFading`). The TypeScript names are canonical for that SDK; cross-references in this document show the pair.
- `Memory` refers to the SDK's memory object (see [memory-schema.md](./memory-schema.md)).
- `embedding` is a list/array of floats (the vector representation).
- IDs are strings (UUIDs).
- Timestamps: Python uses native `datetime` objects; TypeScript uses Unix epoch milliseconds (`number`).

---

## CRUD

### create / createMemory

Store a new memory.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def create(self, memory: Memory) -> None` | `createMemory(memory: Omit<Memory, "id" \| "createdAt" \| "updatedAt">): Promise<string>` |
| **Args** | `memory` — fully populated Memory object including embedding and id | `memory` — Memory fields except `id`, `createdAt`, `updatedAt` (assigned by adapter) |
| **Returns** | Nothing | The newly assigned `id` |
| **Notes** | Must persist all fields. Must reject duplicate ids by raising `DuplicateMemoryError`. | The TS API generates the id; duplicate detection still applies under composition or replay (throws `DuplicateMemoryError`). |

### get / getMemory

Retrieve a single memory by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get(self, memory_id: str) -> Memory \| None` | `getMemory(id: string): Promise<Memory \| null>` |
| **Args** | `memory_id` — UUID | `id` — UUID |
| **Returns** | The Memory if found, `None`/`null` otherwise | Same |

### get_batch / getMemories

Retrieve multiple memories by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get_batch(self, memory_ids: list[str]) -> list[Memory]` | `getMemories(ids: string[]): Promise<Memory[]>` |
| **Args** | `memory_ids` — list of UUIDs | `ids` — array of UUIDs |
| **Returns** | List of found memories. Missing ids are silently omitted. | Same |

### queryMemories *(TypeScript only)*

General-purpose filter query. The Python SDK satisfies the same use case via `all_active`, `all_hot`, `all_cold` plus the engine's filtering.

| | TypeScript |
|---|---|
| **Signature** | `queryMemories(filters: MemoryFilters): Promise<Memory[]>` |
| **Args** | `filters` — `userId?`, `categories?`, `minRetention?`, `minImportance?`, `createdAfter?`, `createdBefore?`, `limit?`, `offset?`, `includeSuperseded?`, `includeCold?`, `includeStubs?` |
| **Returns** | Memories matching the filter set. |

### update / updateMemory

Update an existing memory in place.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def update(self, memory: Memory) -> None` | `updateMemory(id: string, updates: Partial<Memory>): Promise<void>` |
| **Args** | `memory` — Memory with updated fields. `id` must match an existing record. | `id` — UUID. `updates` — subset of fields to overwrite. |
| **Returns** | Nothing | Nothing |
| **Notes** | If memory does not exist, raise `MemoryNotFoundError`. | Same — throws `MemoryNotFoundError`. |

### delete / deleteMemory

Delete a single memory by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete(self, memory_id: str) -> None` | `deleteMemory(id: string): Promise<void>` |
| **Args** | `memory_id` — UUID | `id` — UUID |
| **Returns** | Nothing. No-op if id does not exist. | Same |

### delete_batch / deleteMemories

Delete multiple memories by id.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete_batch(self, memory_ids: list[str]) -> None` | `deleteMemories(ids: string[]): Promise<void>` |
| **Args** | `memory_ids` — list of UUIDs | `ids` — array of UUIDs |
| **Returns** | Nothing. Missing ids are silently ignored. | Same |

---

## Vector Search

### search_similar / vectorSearch

Find memories closest to a given embedding vector.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def search_similar(self, query_embedding: list[float], top_k: int = 10, include_superseded: bool = False, include_cold: bool = False, include_stubs: bool = False, user_id: Optional[str] = None) -> list[tuple[Memory, float]]` | `vectorSearch(embedding: number[], filters?: MemoryFilters): Promise<ScoredMemory[]>` |
| **Args** | `query_embedding` — query vector. `top_k` — max results. `include_superseded` / `include_cold` / `include_stubs` — tier inclusion flags. `user_id` — optional multi-tenant filter (`None` = no filter). | `embedding` — query vector. `filters` — `limit`, `userId`, `categories`, `minRetention`, `includeSuperseded`, `includeCold`, `includeStubs`. |
| **Returns** | List of `(Memory, similarity_score)` tuples, sorted by score descending. | `ScoredMemory[]` (Memory plus `relevanceScore` and `finalScore`), sorted by score descending. |
| **Notes** | Similarity metric is cosine similarity. Normal search excludes cold, superseded, and stub records. Deep recall sets `include_superseded=True` and `include_cold=True` but still excludes stubs. | Same. The TS filter object is the multi-axis equivalent of Python's flat keyword arguments. |

### search_lexical / searchLexical

**(Optional)** Perform a keyword/lexical (non-vector) search over memory content.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def search_lexical(self, query: str, top_k: int = 10, include_superseded: bool = False, include_cold: bool = False, include_stubs: bool = False, user_id: Optional[str] = None) -> list[tuple[Memory, float]]` | `searchLexical(query: string, filters?: MemoryFilters): Promise<ScoredMemory[]>` |
| **Args** | `query` — keyword search string. Other args match `search_similar`. | `query` — keyword search string. `filters` — same shape as `vectorSearch`. |
| **Returns** | List of `(Memory, relevance_score)` tuples, sorted by score descending. | `ScoredMemory[]`, sorted by score descending. |
| **Notes** | Optional method. The Python `InMemoryAdapter` ships a BM25 implementation; the base-class default for adapters without lexical support returns `[]`. Used by the hybrid search pipeline to combine with vector results. | Same. The TS base class default returns `[]`. |

---

## Tiering

### migrate_to_cold / migrateToCold

Move a memory from hot to cold storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def migrate_to_cold(self, memory_id: str, cold_since: datetime) -> None` | `migrateToCold(memoryId: string, coldSince: number): Promise<void>` |
| **Args** | `memory_id` — UUID. `cold_since` — timestamp recorded on the memory. | `memoryId` — UUID. `coldSince` — Unix ms timestamp. |
| **Notes** | Sets `is_cold = True` and `cold_since` to the supplied timestamp. The engine passes `now` from the maintenance tick so deterministic-time tests work. | Sets equivalent fields. |

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
| **Signature** | `async def create_or_strengthen_link(self, source_id: str, target_id: str, weight: float, link_type: str = "association") -> None` | `createOrStrengthenLink(sourceId: string, targetId: string, strength: number): Promise<void>` |
| **Args** | `source_id`, `target_id` — UUIDs. `weight` — float 0..1. `link_type` — category of relationship. | `sourceId`, `targetId`, `strength`. |
| **Notes** | Strengthening adds the new weight to the existing weight, capped at `1.0`. Links are stored under a canonical (sorted) pair key, so a→b and b→a refer to the same row (treated as undirected). | Same. |

### get_linked_memories / getLinkedMemories

Retrieve all memories linked to a given memory.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def get_linked_memories(self, memory_id: str, min_weight: float = 0.3) -> list[tuple[Memory, float]]` | `getLinkedMemories(memoryId: string, minStrength?: number): Promise<Array<Memory & { linkStrength: number }>>` |
| **Args** | `memory_id` — UUID. `min_weight` — exclude links below this weight (default `0.3`). | `memoryId`. `minStrength` — same semantics, default `0.3`. |
| **Returns** | List of `(Memory, weight)` tuples. | Array of `Memory` objects with `linkStrength` merged in. |
| **Notes** | Bidirectional: a link from a→b surfaces when querying either side. The Python implementation reads the union of the adapter link table and the per-memory `Memory.associations` cache (per-target max weight). | Same. |

### getLinkedMemoriesMultiple *(TypeScript only)*

Batch lookup for the linked-memories of multiple source ids in one call.

| | TypeScript |
|---|---|
| **Signature** | `getLinkedMemoriesMultiple(memoryIds: string[], minStrength?: number): Promise<Array<Memory & { linkStrength: number }>>` |
| **Args** | `memoryIds` — array of UUIDs. `minStrength` — default `0.3`. |
| **Returns** | Deduplicated linked memories across all sources, with `linkStrength` set to the maximum across overlapping rows. |
| **Notes** | Used by the engine's graph-expansion stage to avoid N round-trips. Python achieves this by iterating `get_linked_memories` in the engine. |

### delete_link / deleteLink

Remove a link between two memories.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def delete_link(self, source_id: str, target_id: str) -> None` | `deleteLink(sourceId: string, targetId: string): Promise<void>` |
| **Args** | `source_id`, `target_id` — UUIDs | `sourceId`, `targetId` — UUIDs |
| **Returns** | Nothing. No-op if link does not exist. | Same |

---

## Consolidation

### find_fading / findFadingMemories

Find memories whose retention has dropped below a threshold.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def find_fading(self, threshold: float, exclude_core: bool = True) -> list[Memory]` | `findFadingMemories(userId: string, maxRetention: number): Promise<Memory[]>` |
| **Args** | `threshold` — retention score cutoff (e.g. 0.1). `exclude_core` — when `True` (default), excludes core memories so consolidation never folds them. | `userId` — required (TS enforces multi-tenancy). `maxRetention` — cutoff. |
| **Returns** | All active (non-stub, non-superseded) memories with retention below threshold. | Same. |
| **Notes** | The Python signature does not require a `user_id`; user scoping is layered on by the engine, which reads `self.user_id` from `CognitiveMemory`. The TS API surfaces it explicitly because TS makes multi-tenancy mandatory. | |

### find_stable / findStableMemories

Find memories that have remained above a retention threshold consistently.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def find_stable(self, min_stability: float, min_access_count: int) -> list[Memory]` | `findStableMemories(userId: string, minStability: number, minAccessCount: number): Promise<Memory[]>` |
| **Args** | `min_stability` — minimum stability score. `min_access_count` — minimum times accessed. | `userId` — required. `minStability`, `minAccessCount` — same semantics. |
| **Returns** | Memories meeting both criteria. Candidates for core promotion. | Same |

### mark_superseded / markSuperseded

Mark one or more memories as superseded by another (e.g. after consolidation or contradiction resolution).

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def mark_superseded(self, memory_ids: list[str], summary_id: str) -> None` | `markSuperseded(memoryIds: string[], summaryId: string): Promise<void>` |
| **Args** | `memory_ids` — list of memories to mark. `summary_id` — the replacement memory's id. | `memoryIds`, `summaryId`. |
| **Notes** | Sets `is_superseded = True` and `superseded_by = summary_id` on each memory. Does not delete the old memories. | Same. |

---

## Traversal

Python read methods accept an optional `user_id: Optional[str] = None` filter; `None` means no filter. The TypeScript equivalents do not currently take a user filter on these methods (the engine pre-filters via `queryMemories`), but implementations MAY add it.

### all_active / allActive

Retrieve all non-superseded, non-stub memories.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_active(self, user_id: Optional[str] = None) -> list[Memory]` | `allActive(): Promise<Memory[]>` |
| **Returns** | All memories where `is_superseded = False` and `is_stub = False`, optionally filtered by user. | Same (no user filter on this method in TS). |

### all_hot / allHot

Retrieve all memories currently in hot storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_hot(self, user_id: Optional[str] = None) -> list[Memory]` | `allHot(): Promise<Memory[]>` |
| **Returns** | All memories where `is_cold = False` and `is_stub = False`. | Same |

### all_cold / allCold

Retrieve all memories currently in cold storage.

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def all_cold(self, user_id: Optional[str] = None) -> list[Memory]` | `allCold(): Promise<Memory[]>` |
| **Returns** | All memories where `is_cold = True`. | Same |

---

## Counts

All count methods accept the same optional `user_id` filter on the Python side.

### hot_count / hotCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def hot_count(self, user_id: Optional[str] = None) -> int` | `hotCount(): Promise<number>` |
| **Returns** | Number of hot (non-cold, non-stub) memories. | Same |

### cold_count / coldCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def cold_count(self, user_id: Optional[str] = None) -> int` | `coldCount(): Promise<number>` |
| **Returns** | Number of cold memories. | Same |

### stub_count / stubCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def stub_count(self, user_id: Optional[str] = None) -> int` | `stubCount(): Promise<number>` |
| **Returns** | Number of stub memories. | Same |

### total_count / totalCount

| | Python | TypeScript |
|---|---|---|
| **Signature** | `async def total_count(self, user_id: Optional[str] = None) -> int` | `totalCount(): Promise<number>` |
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
| **Signature** | `async def update_retention_scores(self, updates: dict[str, float]) -> None` | `updateRetentionScores(updates: Map<string, number>): Promise<void>` |
| **Args** | `updates` — `{memory_id: new_retention}` mapping | `updates` — `Map<memory_id, new_retention>` |
| **Notes** | Optimized for bulk writes. Should not trigger full Memory serialization. Adapters that compute retention on-the-fly (e.g. InMemory) MAY implement this as a no-op. | Same |

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

1. **Error handling** — Adapters raise typed errors from `cognitive_memory.adapters` (Python) / `cognitive-memory` (TS):
   - `AdapterError` — base class for backend failures (connection drop, transaction conflict, schema mismatch).
   - `MemoryNotFoundError` (extends `AdapterError`) — thrown by `update`/`updateMemory` when the id does not exist.
   - `DuplicateMemoryError` (extends `AdapterError`) — thrown by `create`/`createMemory` when the id already exists in any tier.
2. **Concurrency model** — Both SDKs assume cooperative async (single event loop). Adapters do not need OS-thread locking; they must remain consistent under interleaved `await` points and across concurrent in-flight requests sharing the same instance. Database-backed adapters should use connection pooling and rely on the database for atomicity (transactions). Adapters that share mutable in-memory state (e.g. `InMemoryAdapter`, `JsonlFileAdapter`) are safe under the SDK's normal usage; sharing a single instance across multiple OS threads requires the caller to add locking.
3. **Embedding storage** — Embeddings are stored as arrays of floats. Database adapters should use native vector types where available (pgvector for PostgreSQL) and fall back to serialized JSON otherwise.
4. **Link storage** — Links live in an adapter-level store separate from `Memory.associations`, with row shape `(source_id, target_id, weight, link_type, created_at, updated_at)`. Implementations MAY use a canonical (sorted) pair key to make the store undirected — both Python `InMemoryAdapter` and TS `InMemoryAdapter` do this. The engine treats `Memory.associations` as a per-row cache and the adapter store as the durable record; ingestion and retrieval write through to both and read the union (per-target max weight). Strengthening adds the new weight to the existing weight, capped at `1.0`. Bidirectional queries: a link surfaces when querying from either endpoint.
5. **Idempotency** — `delete`, `delete_batch`, `deleteMemory`, `deleteMemories`, and `delete_link`/`deleteLink` are idempotent (no-op if target does not exist). `create`/`createMemory` rejects duplicate ids by raising `DuplicateMemoryError`.
6. **User scoping (multi-tenancy)** — Memories carry `user_id` (Python) / `userId` (TypeScript) as a top-level field. Python read methods (`search_similar`, `search_lexical`, `all_active`, `all_hot`, `all_cold`, the count methods) take an optional `user_id` filter (`None` = no filter). TypeScript surfaces user filtering primarily through `MemoryFilters.userId` on `vectorSearch`/`searchLexical`/`queryMemories`, and through required `userId` arguments on `findFadingMemories`/`findStableMemories`. Conflict resolution and consolidation are scoped per-user in the engine.
7. **Conflict resolution** — Adapters do not implement conflict logic; the engine uses `contradicted_by`, `is_superseded`, and `superseded_by` fields. On `CONTRADICTION` or `UPDATE`, the existing memory MUST be preserved (its `content` is never overwritten); only `category` may be demoted (core → semantic) and `contradicted_by` set to the new memory's id. The new memory's `importance` is lifted to `max(existing, new)`. `CONTRADICTION` of a previously-core memory promotes the new memory to core; `UPDATE` does not. This preserves an audit trail.
8. **Defaults for optional methods** — `searchLexical` (TS) and `search_lexical` (Python) have a base-class default that returns `[]`; only override when the backend supports lexical search. `batchUpdate` (TS) has a base-class default that loops `updateMemory`; override for true atomicity (e.g. wrapping in a single transaction).
