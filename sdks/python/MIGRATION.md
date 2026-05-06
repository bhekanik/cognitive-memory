# Migration Guide

## 0.3.0 → 0.4.0

This release brings the Python SDK to behavioural parity with the TypeScript
SDK and ships a new file-backed adapter. All changes are **additive** —
existing 0.3.0 callers continue to work without code changes.

### What's new

#### Pluggable LLM provider

The extractor and conflict-detector now talk to an `LLMProvider` interface
instead of an inlined OpenAI client. This means:

- The SDK can run without `OPENAI_API_KEY` when you wire your own provider.
- Tests don't need network access.
- Switching to Anthropic / local models / a gateway is a one-class change.

```python
from cognitive_memory import CognitiveMemory, OpenAILLMProvider, LLMProvider

# Default — unchanged behaviour from 0.3.0:
mem = CognitiveMemory()

# Or inject a custom provider:
class MyProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        # call your model here
        ...

mem = CognitiveMemory(llm=MyProvider())
```

#### Multi-tenancy via `user_id`

`Memory` now carries a `user_id` field (default `"default"`). Two
`CognitiveMemory` instances with different `user_id`s sharing one adapter
are fully isolated — alice's `search()` will not surface bob's memories.

```python
alice = CognitiveMemory(adapter=shared_adapter, user_id="alice")
bob = CognitiveMemory(adapter=shared_adapter, user_id="bob")

await alice.add("alice's secret")
await bob.add("bob's secret")

# alice.search() never returns bob's memories.
```

Adapter read methods (`search_similar`, `all_active`, count methods, etc.)
now accept an optional `user_id=` filter. Default behaviour (no filter)
preserves the 0.3.0 single-tenant assumption.

#### File-backed adapter

`JsonlFileAdapter` writes every mutation as a JSONL event and replays the
log on construction. Single-process production workloads that need
durability without a database now have a one-line option.

```python
from cognitive_memory import CognitiveMemory, JsonlFileAdapter

mem = CognitiveMemory(
    adapter=JsonlFileAdapter("/var/lib/myapp/memories.jsonl"),
)
```

#### Adapter-level link table

`InMemoryAdapter.create_or_strengthen_link()` was previously a no-op (links
lived only on `Memory.associations`). It now persists in a separate link
store as the spec requires. The engine reads the union of both stores at
retrieval time. No call-site changes needed; the bug is fixed transparently.

### Behavioural changes (no API impact)

- **Synaptic-tagging weight curve** at ingestion now uses
  `min(0.5, 0.2 + (sim − 0.4) · 0.5)` with a non-strict `>=` threshold,
  matching the paper's formulation. Above the 0.4 threshold this is
  algebraically identical to the prior formula; the only observable change
  is at the boundary (exactly sim=0.4).

### Spec

The spec at `cognitive-memory-sdk/spec/` has been corrected to match what
the code actually does (TypeScript memory fields are flat top-level, not
nested under `metadata`; user_id filter on read methods; conflict-resolution
preserves the audit trail).

### Deferred to 0.4.1

- **Postgres adapter** with pgvector. JSONL covers the most common
  single-process production case; the Postgres adapter is staged for
  0.4.1 with a live integration test suite.
