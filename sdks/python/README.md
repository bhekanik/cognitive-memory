# cognitive-memory

Biologically-inspired agent memory with decay, consolidation, and tiered storage.

[![PyPI version](https://img.shields.io/pypi/v/cognitive-memory.svg)](https://pypi.org/project/cognitive-memory/)

Python SDK. v0.4.0 brings hybrid retrieval (BM25 + vector), power-law decay, graph expansion, LLM rerank, deferred conflict resolution, multi-tenancy, a pluggable `LLMProvider`, and a JSONL file adapter. Behavioural parity with the TypeScript SDK.

## Install

```bash
pip install cognitive-memory
```

For the OpenAI extractor and embedder:

```bash
pip install "cognitive-memory[openai]"
export OPENAI_API_KEY=sk-...
```

Requires Python 3.10+.

## Quick Start

```python
from cognitive_memory import SyncCognitiveMemory

mem = SyncCognitiveMemory(embedder="hash")  # zero-dep, deterministic

mem.add("User is allergic to shellfish", category="core", importance=0.95)

response = mem.search("what allergies does the user have?")
for r in response.results:
    print(r.memory.content, f"(score: {r.combined_score:.2f})")
```

For async code, use `CognitiveMemory` directly (same API, `await mem.add(...)` / `await mem.search(...)`).

## Production setup

```python
from cognitive_memory import CognitiveMemory, JsonlFileAdapter

mem = CognitiveMemory(
    embedder="openai",                                  # OpenAIEmbeddings, reads OPENAI_API_KEY
    adapter=JsonlFileAdapter("/var/lib/myapp/mem.jsonl"),  # durable, single-process
    user_id="alice",                                    # multi-tenant scoping
)

await mem.extract_and_store(conversation_text, session_id="sess-1")
results = await mem.search("UI preferences", deep_recall=True, rerank=True)
```

`extract_and_store(...)` runs the LLM extractor; `add(...)` skips it for pre-extracted facts.

## Custom LLM provider

The extractor and conflict-resolver talk to an `LLMProvider` interface — swap OpenAI for Anthropic, a local model, or a gateway:

```python
from cognitive_memory import CognitiveMemory, LLMProvider

class MyProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        ...  # your model

mem = CognitiveMemory(llm=MyProvider())
```

## Multi-tenancy

`user_id` namespaces every read and write. Two instances sharing an adapter are fully isolated.

```python
alice = CognitiveMemory(adapter=shared, user_id="alice")
bob   = CognitiveMemory(adapter=shared, user_id="bob")

await alice.add("alice's secret")
# bob.search() never returns alice's memories
```

## Adapters

- `InMemoryAdapter` — default, ephemeral
- `JsonlFileAdapter` — append-only event log, replay on startup
- Custom — implement `MemoryAdapter` from `cognitive_memory.adapters`
- Postgres (pgvector) — planned for 0.4.1

## Migration

See [`MIGRATION.md`](./MIGRATION.md) for the 0.3.0 → 0.4.0 changes (all additive — existing code keeps working).

## Docs

Full documentation, guides, concepts, and API reference: **[planetaryescape.github.io/cognitive-memory](https://planetaryescape.github.io/cognitive-memory)**.

## License

[MIT](../../LICENSE)
