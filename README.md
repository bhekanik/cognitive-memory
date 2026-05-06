# Cognitive Memory

> Biologically-inspired agent memory with decay, consolidation, and tiered storage

[![npm version](https://img.shields.io/npm/v/cognitive-memory.svg)](https://www.npmjs.com/package/cognitive-memory)
[![PyPI version](https://img.shields.io/pypi/v/cognitive-memory.svg)](https://pypi.org/project/cognitive-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Memory that behaves like memory. Cognitive Memory models how humans actually remember — important things stick, irrelevant things fade, and contradictions get resolved. Built for AI agents that need long-term memory across conversations.

**Benchmark highlights:** 48.5% multi-hop F1 on [LoCoMo](https://github.com/snap-research/locomo) — 1.7× Mem0's 28.4%. 70.2% task-averaged accuracy on [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval), competitive with ENGRAM-class single-stage memory systems. 100% critical-fact retention on a controlled 30-day long-term-interaction benchmark.

Benchmark methodology and caveats live in the [benchmark repo](https://github.com/planetaryescape/cognitive-memory-benchmarks).

## Install

**Python**

```bash
pip install cognitive-memory
```

**TypeScript**

```bash
npm install cognitive-memory
```

## Quick Start

### Python

```python
from cognitive_memory import SyncCognitiveMemory, MemoryCategory

mem = SyncCognitiveMemory(embedder="hash")

# Store a memory
mem.add("User prefers dark mode and compact layouts", category=MemoryCategory.SEMANTIC, importance=0.7)

# Search memories
response = mem.search("What are the user's UI preferences?")
for r in response.results:
    print(r.memory.content, f"(score: {r.combined_score:.2f})")
```

### TypeScript

```typescript
import { CognitiveMemory, InMemoryAdapter, HashEmbeddingProvider } from "cognitive-memory";

const mem = new CognitiveMemory({
  adapter: new InMemoryAdapter(),
  embeddingProvider: new HashEmbeddingProvider(),
  userId: "user-1",
});

// Store a memory
await mem.store({
  content: "User prefers dark mode and compact layouts",
  category: "semantic",
  importance: 0.7,
});

// Search memories
const { results } = await mem.search({ query: "What are the user's UI preferences?" });
for (const r of results) {
  console.log(r.memory.content, `(score: ${r.combinedScore.toFixed(2)})`);
}
```

## Key Features

- **Retention dynamics** — Memories decay over time, but floors, stability, and reinforcement keep important memories available.
- **Accessibility scoring** — Retrieval uses `similarity × R^alpha`, so faded but relevant memories can still surface.
- **Core promotion** — Important or repeatedly accessed memories get promoted to "core" status with a high retention floor (0.60), making them near-permanent.
- **Associations** — Memories automatically form weighted links to related memories, enabling graph-based traversal and richer recall.
- **Tiered storage** — Hot, cold, and stub tiers. Active memories stay hot. Stale or superseded memories migrate to cold storage, and very old cold records can become lightweight stubs after TTL expiry.
- **Deep recall** — Retrieve superseded and cold memories at a relevance penalty, keeping archived context available when you explicitly ask for it.
- **Adapters** — Pluggable storage backends. Ships with in-memory adapters, TypeScript Postgres/JSONL/Convex adapters, and adapter interfaces for custom backends.

## Docs

Full documentation, guides, and API reference at **[bhekanik.github.io/cognitive-memory](https://bhekanik.github.io/cognitive-memory)**.

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
