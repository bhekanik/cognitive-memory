# Migration Guide

## 0.3.0 → 0.4.0

This release brings the TypeScript SDK to behavioural parity with the
Python SDK around conflict resolution and synaptic tagging. The public
API is unchanged; behaviour at runtime is now stricter about preserving
audit trails on contradictions.

### Behavioural changes (no API impact)

#### Conflict resolution preserves the audit trail

When `tick()` resolves a `CONTRADICTION` or `UPDATE` from the deferred
conflict queue, the **existing memory's content is no longer overwritten**.
Instead:

| | Before (0.3.0) | After (0.4.0) |
|---|---|---|
| Existing `content` | Replaced with new memory's content | Preserved verbatim |
| Existing `category` | Untouched | Demoted from `core` → `semantic` (if applicable) |
| Existing `contradictedBy` | Set on `CONTRADICTION` only | Set on both `CONTRADICTION` and `UPDATE` |
| Existing `importance` | Lifted to max | Untouched |
| New memory `importance` | Untouched | Lifted to `max(existing, new)` |
| New memory `category` | Untouched | Promoted to `core` on `CONTRADICTION` of a previously-core memory |

The original memory remains queryable forever — it just has
`contradictedBy` set so callers can filter it out or surface it as
"superseded by …".

This matches the Python SDK's algorithm and aligns with how Mem0 and Letta
treat contradictions as first-class objects rather than silent overwrites.

#### Synaptic-tagging weight curve

The weight assigned to a new ingestion-time link is now
`min(0.5, 0.2 + (sim − 0.4) * 0.5)` with a non-strict `>=` gate at the
0.4 threshold. Above the threshold this is algebraically identical to the
prior `sim * 0.5` formula; the only observable change is at exactly
sim = 0.4 (now creates a link with weight 0.2, previously skipped).

### What's new in the Python SDK (relevant context)

For users running both SDKs against shared infrastructure, note that
Python 0.4.0 adds:

- `user_id` multi-tenancy (matching TypeScript's mandatory `userId`).
- `LLMProvider` interface (matching TypeScript's `LLMProvider`).
- `JsonlFileAdapter` (file-backed durability).
- A real adapter-level link table (previously a no-op in `InMemoryAdapter`).

The Python and TypeScript SDKs now produce identical observable state for
the cross-SDK parity scenario suite (`tests/parity-fixtures/`).

### Spec

The spec at `cognitive-memory-sdk/spec/` has been corrected to match what
the code actually does (Memory fields are flat top-level, not nested under
`metadata`).
