# Memory Schema Specification

This document defines the Memory object fields for both the Python and TypeScript SDKs. The two representations are conceptually equivalent but use language-appropriate naming and typing conventions.

---

## Python Memory Dataclass

```python
@dataclass
class Memory:
    id: str                              # UUID v4, primary key
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

---

## TypeScript Memory Interface

```typescript
interface Memory {
  id: string;                           // UUID v4, primary key
  userId: string;                       // Owner of this memory
  content: string;                      // The stored text content
  embedding: number[] | null;           // Vector embedding of content (null for stubs)
  memoryType: MemoryType;               // Classification enum
  importance: number;                   // 0.0–1.0, importance score
  stability: number;                    // 0.0–1.0, resistance to decay
  accessCount: number;                  // Number of retrievals
  lastAccessed: Date;                   // Timestamp of last retrieval
  retention: number;                    // Current computed retention score (0.0–1.0)
  createdAt: Date;                      // Timestamp of creation
  updatedAt: Date;                      // Timestamp of last modification
  metadata: MemoryMetadata;             // Additional structured data
}

enum MemoryType {
  Preference = "preference",
  Fact = "fact",
  Event = "event",
  Relationship = "relationship",
  General = "general",
}

interface MemoryMetadata {
  sessionIds: string[];                 // Sessions that contributed to this memory
  associations: Record<string, number>; // Linked memory_id -> weight
  isCold: boolean;                      // Whether in cold storage
  coldSince: Date | null;               // When moved to cold storage
  daysAtFloor: number;                  // Decay cycles at retention floor
  isSuperseded: boolean;                // Whether replaced by newer memory
  supersededBy: string | null;          // ID of replacement
  contradictedBy: string | null;        // ID of contradicting memory
  isStub: boolean;                      // Whether converted to stub
}
```

### Field Details

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `string` | uuid() | Unique identifier. |
| `userId` | `string` | required | User who owns this memory. Enables multi-tenant storage. |
| `content` | `string` | required | The textual content. |
| `embedding` | `number[] \| null` | `null` | Vector embedding. `null` for stubs. |
| `memoryType` | `MemoryType` | `"general"` | Classification enum. Maps to Python's `category` field. |
| `importance` | `number` | `0.5` | Importance score, 0.0–1.0. |
| `stability` | `number` | `0.0` | Decay resistance, 0.0–1.0. |
| `accessCount` | `number` | `0` | Retrieval counter. |
| `lastAccessed` | `Date` | creation time | Last retrieval timestamp. |
| `retention` | `number` | `1.0` | Current computed retention. Updated by the decay engine each cycle. Derived from `R^alpha` formula. Not stored in Python (computed on the fly). |
| `createdAt` | `Date` | now | Immutable creation timestamp. |
| `updatedAt` | `Date` | now | Updated on any mutation. Not present in Python (tracked implicitly). |
| `metadata` | `MemoryMetadata` | defaults | Structured metadata. Groups tiering and lifecycle fields. |

---

## Field Mapping: Python <-> TypeScript

| Python Field | TypeScript Field | Notes |
|---|---|---|
| `id` | `id` | Same |
| _(no equivalent)_ | `userId` | TypeScript adds explicit multi-tenancy. Python passes user_id at the API level. |
| `content` | `content` | Same |
| `category` | `memoryType` | String in Python, enum in TypeScript. Same values. |
| `importance` | `importance` | Same |
| `stability` | `stability` | Same |
| `access_count` | `accessCount` | snake_case vs camelCase |
| `last_accessed_at` | `lastAccessed` | Python uses `datetime`, TypeScript uses `Date` |
| _(computed)_ | `retention` | TypeScript stores the current retention score. Python computes it on the fly in the engine. |
| `created_at` | `createdAt` | snake_case vs camelCase |
| _(no equivalent)_ | `updatedAt` | TypeScript-only. Tracks last mutation. |
| `embedding` | `embedding` | `list[float]` vs `number[]` |
| `associations` | `metadata.associations` | Top-level in Python, nested in TypeScript's metadata |
| `session_ids` | `metadata.sessionIds` | Top-level in Python, nested in TypeScript's metadata |
| `is_cold` | `metadata.isCold` | Top-level in Python, nested in TypeScript's metadata |
| `cold_since` | `metadata.coldSince` | Top-level in Python, nested in TypeScript's metadata |
| `days_at_floor` | `metadata.daysAtFloor` | Top-level in Python, nested in TypeScript's metadata |
| `is_superseded` | `metadata.isSuperseded` | Top-level in Python, nested in TypeScript's metadata |
| `superseded_by` | `metadata.supersededBy` | Top-level in Python, nested in TypeScript's metadata |
| `contradicted_by` | `metadata.contradictedBy` | Top-level in Python, nested in TypeScript's metadata |
| `is_stub` | `metadata.isStub` | Top-level in Python, nested in TypeScript's metadata |

### Design Rationale

The Python SDK uses a flat dataclass because it evolved from a research prototype where quick attribute access mattered. The TypeScript SDK groups lifecycle and tiering fields under `metadata` to keep the top-level interface clean for application developers who primarily interact with `content`, `importance`, and `retention`. The mapping table above ensures serialization layers can convert between the two representations without data loss.
