# Memory Schema Specification

This document defines the Memory object fields for both the Python and TypeScript SDKs. The two representations are conceptually equivalent but use language-appropriate naming and typing conventions.

---

## Python Memory Dataclass

```python
@dataclass
class Memory:
    id: str                              # UUID v4, primary key
    user_id: str                         # Owner of this memory; default "default" for single-tenant
    content: str                         # The stored text content
    category: str                        # Classification label (e.g. "preference", "fact", "event")
    importance: float                    # 0.0–1.0, how important this memory is (set at creation, can be promoted)
    stability: float                     # 0.0–1.0, how resistant to decay (increases with repeated access)
    access_count: int                    # Number of times this memory has been retrieved
    last_accessed_at: datetime           # Timestamp of last retrieval
    created_at: datetime                 # Timestamp of creation
    embedding: list[float] | None        # Vector embedding of content (None for stubs)
    associations: dict[str, float]       # Map of linked memory_id -> weight (0.0–1.0)
    session_ids: list[str]              # Sessions in which this memory was created or accessed
    is_cold: bool                        # True if memory has been migrated to cold storage
    cold_since: datetime | None          # Timestamp of cold migration (None if hot)
    days_at_floor: int                   # Number of decay cycles spent at the retention floor
    is_superseded: bool                  # True if this memory has been replaced by a newer one
    superseded_by: str | None            # ID of the memory that supersedes this one
    contradicted_by: str | None          # ID of the memory that contradicted this one
    is_stub: bool                        # True if converted to a lightweight stub
    memory_type: str                     # Semantic type: "fact", "preference", "plan", "transient_state", "other"
    valid_from: Optional[datetime]       # Start of temporal validity window (None = always valid)
    valid_until: Optional[datetime]      # End of temporal validity window (None = no expiry)
    ttl_seconds: Optional[int]           # Time-to-live in seconds from creation (None = no TTL)
    source_turn_ids: list[str]           # Conversation turn IDs that contributed to this memory
```

### Field Details

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | uuid4() | Unique identifier. Generated at creation. |
| `content` | `str` | required | The textual content of the memory. Extracted from conversations or provided directly. |
| `category` | `str` | `"general"` | Classification label used for filtering and organization. Common values: `"preference"`, `"fact"`, `"event"`, `"relationship"`, `"general"`. |
| `importance` | `float` | `0.5` | Initial importance score. Range 0.0–1.0. Set by the extraction LLM or manually. Memories above 0.8 are candidates for core promotion. |
| `stability` | `float` | `0.0` | Resistance to decay. Increases with each access. Higher stability means slower decay. Range 0.0–1.0. |
| `access_count` | `int` | `0` | Incremented each time the memory is returned in a search result. Used for core promotion thresholds. |
| `last_accessed_at` | `datetime` | creation time | Updated on each retrieval. Used to calculate time-since-last-access for the decay formula. |
| `created_at` | `datetime` | now | Immutable creation timestamp. |
| `embedding` | `list[float] \| None` | `None` | Vector embedding generated from `content`. Typically 1536 dimensions (OpenAI) or 384 (MiniLM). Set to `None` for stubs. |
| `associations` | `dict[str, float]` | `{}` | Weighted links to other memories. Keys are memory IDs, values are weights (0.0–1.0). Built automatically during ingestion and consolidation. |
| `session_ids` | `list[str]` | `[]` | Tracks which sessions contributed to or accessed this memory. Used for session-scoped queries. |
| `is_cold` | `bool` | `False` | Set to `True` when the memory is migrated to cold storage after prolonged low retention. |
| `cold_since` | `datetime \| None` | `None` | Timestamp of when the memory was moved to cold storage. `None` while hot. |
| `days_at_floor` | `int` | `0` | Counts consecutive decay cycles where retention was at the floor value. Used to trigger cold migration (e.g. after 30 days at floor). |
| `is_superseded` | `bool` | `False` | Set to `True` when a newer memory contradicts and replaces this one. Superseded memories are excluded from normal search but available via deep recall. |
| `superseded_by` | `str \| None` | `None` | ID of the replacement memory. Forms a chain for tracking memory evolution. |
| `contradicted_by` | `str \| None` | `None` | ID of the memory that introduced the contradiction. May differ from `superseded_by` if resolution created a third memory. |
| `is_stub` | `bool` | `False` | Set to `True` when the memory is converted to a stub. Stubs retain `id`, `content` (summary), and association references but drop the embedding. |
| `memory_type` | `str` | `"other"` | Semantic type of the memory. Values: `"fact"`, `"preference"`, `"plan"`, `"transient_state"`, `"other"`. Distinct from `category` which is a free-form classification label. |
| `valid_from` | `Optional[datetime]` | `None` | Start of the temporal validity window. `None` means the memory is valid from creation. |
| `valid_until` | `Optional[datetime]` | `None` | End of the temporal validity window. `None` means no scheduled expiry. |
| `ttl_seconds` | `Optional[int]` | `None` | Time-to-live in seconds from creation. When set, the memory should be considered expired after `created_at + ttl_seconds`. `None` means no TTL. |
| `source_turn_ids` | `list[str]` | `[]` | IDs of the conversation turns that contributed to extracting this memory. Used for provenance tracking. |

---

## TypeScript Memory Interface

The TypeScript `Memory` shape uses **flat top-level fields** for tiering and
lifecycle state. The `metadata` field is a free-form bag for caller-supplied
data, NOT a nested envelope for SDK fields.

```typescript
interface Memory {
  id: string;                           // UUID v4, primary key
  userId: string;                       // Owner of this memory
  content: string;                      // The stored text content
  embedding: number[];                  // Vector embedding of content
  category: MemoryCategory;             // Classification enum
  importance: number;                   // 0.0–1.0, importance score
  stability: number;                    // 0.0–1.0, resistance to decay
  accessCount: number;                  // Number of retrievals
  lastAccessed: number;                 // Unix ms — last retrieval timestamp
  retention: number;                    // Current computed retention score (0.0–1.0)
  createdAt: number;                    // Unix ms — creation timestamp
  updatedAt: number;                    // Unix ms — last mutation timestamp
  metadata?: Record<string, unknown>;   // Caller-supplied free-form bag
  // Tiering and lifecycle (flat, top-level)
  associations: Record<string, Association>;
  sessionIds: string[];
  isCold: boolean;
  coldSince: number | null;             // Unix ms when moved to cold
  daysAtFloor: number;
  isSuperseded: boolean;
  supersededBy: string | null;
  contradictedBy: string | null;
  isStub: boolean;
  // v6: validity window
  semanticType?: SemanticType;
  validFrom?: number | null;
  validUntil?: number | null;
  ttlSeconds?: number | null;
  sourceTurnIds?: string[];
}

enum SemanticType {
  Fact = "fact",
  Preference = "preference",
  Plan = "plan",
  TransientState = "transient_state",
  Other = "other",
}

type MemoryCategory = "core" | "semantic" | "episodic" | "procedural";
```

### Field Details

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `string` | uuid() | Unique identifier. |
| `userId` | `string` | required | User who owns this memory. Enables multi-tenant storage. |
| `content` | `string` | required | The textual content. |
| `embedding` | `number[]` | required | Vector embedding. Empty array for stubs. |
| `category` | `MemoryCategory` | `"semantic"` | Memory category: `"core"`, `"semantic"`, `"episodic"`, or `"procedural"`. |
| `importance` | `number` | `0.5` | Importance score, 0.0–1.0. |
| `stability` | `number` | `0.0` | Decay resistance, 0.0–1.0. |
| `accessCount` | `number` | `0` | Retrieval counter. |
| `lastAccessed` | `number` | now | Unix ms — last retrieval timestamp. |
| `retention` | `number` | `1.0` | Current materialised retention. Refreshed by the engine via `engine.computeRetention(memory, now)` and persisted on the row so storage backends can index/filter on it. Python computes the equivalent on the fly and does not store this field. |
| `createdAt` | `number` | now | Unix ms — immutable creation timestamp. |
| `updatedAt` | `number` | now | Unix ms — updated on any mutation. Python does not have an equivalent field (its adapters track this implicitly). |
| `metadata` | `Record<string, unknown>` | `undefined` | Caller-supplied free-form bag. Not used by the SDK; reserved for application data. |
| `associations` | `Record<string, Association>` | `{}` | Per-memory link cache. Each value carries `targetId`, `weight`, `lastCoRetrieval`, `createdAt`. The durable record lives in the adapter-level link store; this field is the engine's read cache. |
| `sessionIds` | `string[]` | `[]` | Sessions that created or accessed this memory. Used by the core-promotion cross-session check. |
| `isCold` | `boolean` | `false` | True after migration to cold storage. |
| `coldSince` | `number \| null` | `null` | Unix ms timestamp of the cold migration. `null` while hot. |
| `daysAtFloor` | `number` | `0` | Consecutive maintenance cycles spent at the retention floor. Used to trigger cold migration. |
| `isSuperseded` | `boolean` | `false` | True if a consolidation summary has replaced this memory. |
| `supersededBy` | `string \| null` | `null` | ID of the replacement memory. |
| `contradictedBy` | `string \| null` | `null` | ID of a contradicting memory (audit trail; the original is preserved). |
| `isStub` | `boolean` | `false` | True for lightweight stubs (no embedding, content collapsed to a summary). |
| `semanticType` | `SemanticType?` | `undefined` | `"fact"`, `"preference"`, `"plan"`, `"transient_state"`, or `"other"`. Drives v6 validity-window filtering. |
| `validFrom` | `number \| null` | `undefined` | Unix ms — start of the validity window. `null`/`undefined` = valid from creation. |
| `validUntil` | `number \| null` | `undefined` | Unix ms — end of the validity window. `null`/`undefined` = no scheduled expiry. |
| `ttlSeconds` | `number \| null` | `undefined` | Time-to-live in seconds from creation. The memory expires after `createdAt + ttlSeconds * 1000`. |
| `sourceTurnIds` | `string[]` | `[]` | Conversation turn IDs that contributed to extracting this memory. Provenance tracking. |

---

## Search Response Types (TypeScript)

```typescript
interface StageTrace {
  stage: string;                        // Name of the search pipeline stage (e.g. "vector", "lexical", "rerank")
  inputCount: number;                   // Number of candidates entering this stage
  outputCount: number;                  // Number of candidates leaving this stage
  durationMs: number;                   // Time spent in this stage in milliseconds
  promptTokens: number;                 // LLM prompt tokens used in this stage (0 for non-LLM stages)
  completionTokens: number;            // LLM completion tokens used in this stage (0 for non-LLM stages)
  metadata?: Record<string, unknown>;   // Optional stage-specific debug data
}

interface SearchTrace {
  totalDurationMs: number;              // Total wall-clock time for the search
  totalTokens: number;                  // Sum of all prompt + completion tokens across stages
  stages: StageTrace[];                 // Ordered list of pipeline stage traces
}

interface SearchResponse {
  results: SearchResult[];              // Ranked results
  evidenceChains: string[][];
  trace?: SearchTrace;                  // Optional pipeline trace (included when `debug: true`)
}

interface SearchResult {
  memory: Memory;
  relevanceScore: number;               // Cosine similarity to query
  retentionScore: number;               // R(m)
  combinedScore: number;                // semantic relevance weighted by retention^alpha
  isAssociative: boolean;
  viaDeepRecall: boolean;
}
```

---

## Field Mapping: Python <-> TypeScript

| Python Field | TypeScript Field | Notes |
|---|---|---|
| `id` | `id` | Same |
| `user_id` | `userId` | Both SDKs scope memories by user. Python defaults to `"default"`. |
| `content` | `content` | Same |
| `category` | `category` | Same value set: `"core"` / `"semantic"` / `"episodic"` / `"procedural"`. Python uses `MemoryCategory` enum, TypeScript uses a union string type. |
| `importance` | `importance` | Same |
| `stability` | `stability` | Same |
| `access_count` | `accessCount` | snake_case vs camelCase |
| `last_accessed_at` | `lastAccessed` | Python `datetime`, TypeScript Unix ms `number` |
| _(computed on the fly)_ | `retention` | TypeScript materialises retention on the row so backends like Postgres can index and filter on it. Python computes via `engine.compute_retention(memory, now)`; see Design Rationale. |
| `created_at` | `createdAt` | Python `datetime`, TypeScript Unix ms `number` |
| _(implicit)_ | `updatedAt` | TypeScript-only. Tracks last mutation; Python adapters track this implicitly. |
| `embedding` | `embedding` | `list[float]` vs `number[]` |
| `associations` | `associations` | Both top-level. Python: `dict[str, Association]`. TypeScript: `Record<string, Association>`. |
| `session_ids` | `sessionIds` | Both top-level. Python `set[str]`, TypeScript `string[]`. |
| `is_cold` | `isCold` | Both top-level. |
| `cold_since` | `coldSince` | Both top-level. |
| `days_at_floor` | `daysAtFloor` | Both top-level. |
| `is_superseded` | `isSuperseded` | Both top-level. |
| `superseded_by` | `supersededBy` | Both top-level. |
| `contradicted_by` | `contradictedBy` | Both top-level. |
| `is_stub` | `isStub` | Both top-level. |
| `memory_type` | `semanticType` | Python str field, TypeScript optional enum. Values: `"fact"`, `"preference"`, `"plan"`, `"transient_state"`, `"other"`. Used for v6 validity-window filtering. |
| `valid_from` | `validFrom` | Python `datetime`, TypeScript Unix ms `number`. |
| `valid_until` | `validUntil` | Python `datetime`, TypeScript Unix ms `number`. |
| `ttl_seconds` | `ttlSeconds` | Same semantics. |
| `source_turn_ids` | `sourceTurnIds` | snake_case vs camelCase |

### Design Rationale

**Flat top-level fields (both SDKs).** Both `Memory` shapes keep tiering and
lifecycle fields at the top level. The TypeScript `metadata?: Record<string,
unknown>` is a free-form bag for caller-supplied data, NOT an envelope for SDK
state. (An earlier draft of this spec described nested `MemoryMetadata`; the
shipping code has been flat since early releases — the spec has been corrected.)

**Materialised vs computed retention.** TypeScript stores `retention` on the
row because its bundled adapters (Postgres, Convex) benefit from indexed
retention filters at query time. Python's only bundled adapter is in-memory,
where on-the-fly compute is cheap; a future Python Postgres adapter will use
`update_retention_scores` to materialise the same field.
