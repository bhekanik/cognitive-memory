# Cognitive Memory

> Biologically-inspired agent memory with decay, consolidation, and tiered storage

[![npm version](https://img.shields.io/npm/v/cognitive-memory.svg)](https://www.npmjs.com/package/cognitive-memory)
[![PyPI version](https://img.shields.io/pypi/v/cognitive-memory.svg)](https://pypi.org/project/cognitive-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Memory that behaves like memory. Important things stick, irrelevant things fade, and contradictions get resolved instead of silently overwritten. Built for AI agents that need long-term memory across conversations.

**Current release:** v0.4.0 — Python and TypeScript SDKs at behavioural parity, hybrid retrieval (BM25 + vector), power-law decay, graph expansion, LLM rerank, deferred conflict resolution that preserves the audit trail.

**Benchmark highlights** (v6 retrieval pipeline, refresh of 2026-05-05/06):

| Benchmark | Result | Comparison |
| --- | --- | --- |
| **LoCoMo** (10 conv, 1540 QA) | 44.8% overall F1 · 48.5% multi-hop F1 | 1.7× Mem0's 28.4% multi-hop · 70% of LoCoMo oracle evidence context condition (63.9%) |
| **LongMemEval-S** (500 Q) | 71.6% task-averaged accuracy · 72.6% overall accuracy | Competitive with ENGRAM 71.4% · +15.4pp over full-context 56.2% |
| **LTI-Bench** (controlled, 42 probes) | 88.1% accuracy · 100% critical-fact retention | FadeMem 82.1% critical retention |

Methodology, parameters, and per-category breakdowns live in the [benchmark repo](https://github.com/planetaryescape/cognitive-memory-benchmarks).

## Install

**Python**

```bash
pip install cognitive-memory
# or, with the OpenAI extractor + embedder wired up:
pip install "cognitive-memory[openai]"
```

**TypeScript**

```bash
npm install cognitive-memory
# Optional adapters use peer dependencies:
npm install pg          # for PostgresAdapter
npm install convex      # for ConvexAdapter
```

Python 3.10+, Node 18+ (or Bun), ESM-only.

## Quick Start

### Python

```python
from cognitive_memory import SyncCognitiveMemory, MemoryCategory

mem = SyncCognitiveMemory(embedder="hash")  # swap "hash" → "openai" for production

# Store a memory
mem.add(
    "User prefers dark mode and compact layouts",
    category=MemoryCategory.SEMANTIC,
    importance=0.7,
)

# Search memories
response = mem.search("What are the user's UI preferences?")
for r in response.results:
    print(r.memory.content, f"(score: {r.combined_score:.2f})")
```

For async code, use `CognitiveMemory` directly. For multi-tenant deployments, pass `user_id="alice"` — searches are isolated per user even when adapters are shared.

### TypeScript

```typescript
import { CognitiveMemory, InMemoryAdapter, HashEmbeddingProvider } from "cognitive-memory";

const mem = new CognitiveMemory({
  adapter: new InMemoryAdapter(),
  embeddingProvider: new HashEmbeddingProvider(),
  userId: "user-1",
});

await mem.store({
  content: "User prefers dark mode and compact layouts",
  category: "semantic",
  importance: 0.7,
});

const { results } = await mem.search({ query: "What are the user's UI preferences?" });
for (const r of results) {
  console.log(r.memory.content, `(score: ${r.combinedScore.toFixed(2)})`);
}
```

For production with OpenAI embeddings and a durable adapter:

```typescript
import { CognitiveMemory, OpenAIEmbeddingProvider } from "cognitive-memory";
import { PostgresAdapter } from "cognitive-memory/adapters/postgres";
import { Pool } from "pg";

const mem = new CognitiveMemory({
  adapter: new PostgresAdapter({ pool: new Pool({ connectionString: process.env.DATABASE_URL }) }),
  embeddingProvider: new OpenAIEmbeddingProvider({ apiKey: process.env.OPENAI_API_KEY }),
  userId: "user-1",
});
```

## Key Features

- **Power-law decay** — Default `R(t) = (1 + t/s)^(−γ)`, with `exponential` available behind a config flag. Power-law matches the empirical forgetting curve and adds +3.2pp on LoCoMo over exponential.
- **Hybrid retrieval** — Vector similarity fused with BM25 lexical search; `hybridSearch: true` enables it. Surfaces both semantically-related and exact-keyword matches.
- **Graph expansion** — One-hop or two-hop traversal across the association graph at query time. Bridge discovery (`bridgeDiscovery: true`) finds evidence chains across multiple memories — the multi-hop reasoning lever.
- **LLM rerank** — Optional post-retrieval rerank via the LLM provider with a configurable candidate pool (`rerankFactor`). +1.9pp on LoCoMo headline.
- **Deferred conflict resolution** — Contradictions are queued at ingestion and resolved at `tick()` time. Conflicting memories are *demoted*, never overwritten — the original stays queryable with `contradictedBy` set, preserving the audit trail.
- **Two-tier retrieval** — `similarity × R^α` scoring lets faded but relevant memories surface. Deep recall (`deepRecall: true`) reaches into cold and superseded memories at a relevance penalty.
- **Core promotion** — Important or repeatedly accessed memories get promoted to `core` with a 0.60 retention floor, making them effectively permanent.
- **Tiered storage** — Hot / cold / stub. Stale memories migrate to cold; very old cold records become lightweight stubs after TTL expiry.
- **Pluggable LLM provider** — Built-in `OpenAILLMProvider`, or implement `LLMProvider` to wire Anthropic, local models, or a gateway. Tests run without network.
- **Multi-tenancy** — `user_id` (Python) / `userId` (TS) namespaces every read and write. Two `CognitiveMemory` instances over a shared adapter are fully isolated.
- **Cross-SDK parity** — Python and TypeScript SDKs produce identical observable state on the parity test suite.

## Adapters

| Adapter | Python | TypeScript |
| --- | --- | --- |
| In-memory | ✓ | ✓ |
| JSONL file (durable, single-process) | ✓ | ✓ |
| Postgres (pgvector) | 0.4.1 (planned) | ✓ |
| Convex | — | ✓ |
| Custom (`MemoryAdapter` interface) | ✓ | ✓ |

## Docs

Full documentation, guides, concepts, benchmarks, and API reference at **[planetaryescape.github.io/cognitive-memory](https://planetaryescape.github.io/cognitive-memory)**.

Migration: [Python 0.3.0 → 0.4.0](./sdks/python/MIGRATION.md) · [TypeScript 0.3.0 → 0.4.0](./sdks/typescript/MIGRATION.md).

## Daemon mode

For a long-running shared-process deployment — one embedding model loaded once, single SQLite writer, cross-agent visibility, central lifecycle scheduling — point both SDKs at the [**`cognitive-memory-daemon`**](https://github.com/planetaryescape/cognitive-memory-daemon) Rust service. Multiple AI clients on the same machine (Claude Code, Cursor, scripts, the SDK in `RemoteAdapter` mode) share one canonical store over a Unix socket.

```python
from cognitive_memory import CognitiveMemory
from cognitive_memory.adapters.remote import RemoteAdapter

cm = CognitiveMemory(adapter=RemoteAdapter(user_id="alice"), user_id="alice")
# Auto-spawns the daemon on first call. All in-process methods now go
# through the daemon: shared embedding cache, single writer, central tick.
```

```typescript
import { CognitiveMemory } from "cognitive-memory";
import { RemoteAdapter } from "cognitive-memory/adapters/remote";

const cm = new CognitiveMemory({
  adapter: new RemoteAdapter({ userId: "alice" }),
  // ...
});
```

The daemon also ships a `cm` CLI and a loopback HTTP bridge (`cm-http`) for browser clients. See the [daemon repo](https://github.com/planetaryescape/cognitive-memory-daemon) for the full architecture, `cm` command reference, and install instructions.

## Related repositories

The cognitive-memory project spans three repos:

| Repo | What it is |
| --- | --- |
| **[cognitive-memory](https://github.com/planetaryescape/cognitive-memory)** (this) | Python + TypeScript SDKs. The library you embed in your application. |
| **[cognitive-memory-daemon](https://github.com/planetaryescape/cognitive-memory-daemon)** | Rust daemon, `cm` CLI, and `cm-http` bridge. The shared-process deployment. |
| **[cognitive-memory-benchmarks](https://github.com/planetaryescape/cognitive-memory-benchmarks)** | LoCoMo, LongMemEval-S, LTI-Bench evaluation harness. Reproducible methodology and per-category breakdowns. |

## Repo Structure

```
cognitive-memory/
├── sdks/
│   ├── python/          # Python SDK (pip install cognitive-memory)
│   └── typescript/      # TypeScript SDK (npm install cognitive-memory)
├── spec/
│   ├── adapter-interface.md   # Canonical adapter contract
│   └── memory-schema.md       # Memory object field definitions
├── docs/                # Documentation site (Astro)
├── Makefile             # Monorepo task runner
├── LICENSE              # MIT
└── README.md
```

## License

[MIT](./LICENSE) — Copyright 2024-2026 Bhekani Khumalo
