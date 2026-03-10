# Cognitive Memory

> Biologically-inspired agent memory with decay, consolidation, and tiered storage

[![npm version](https://img.shields.io/npm/v/cognitive-memory.svg)](https://www.npmjs.com/package/cognitive-memory)
[![PyPI version](https://img.shields.io/pypi/v/cognitive-memory.svg)](https://pypi.org/project/cognitive-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Memory that behaves like memory. Cognitive Memory models how humans actually remember — important things stick, irrelevant things fade, and contradictions get resolved. Built for AI agents that need long-term memory across conversations.

**Benchmark highlight:** 47.1% multi-hop accuracy on [LoCoMo](https://github.com/snap-research/locomo) — 66% ahead of Mem0.

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

- **Decay model** — Memories fade over time following a power-law curve (`R^alpha`), just like human memory. Frequently accessed memories decay slower.
- **Core promotion** — Important or repeatedly accessed memories get promoted to "core" status with a high retention floor (0.60), making them near-permanent.
- **Associations** — Memories automatically form weighted links to related memories, enabling graph-based traversal and richer recall.
- **Tiered storage** — Hot, cold, and stub tiers. Active memories stay hot. Stale memories migrate to cold storage. Superseded memories become lightweight stubs.
- **Deep recall** — Retrieve superseded and cold memories at a relevance penalty, so nothing is truly lost.
- **Adapters** — Pluggable storage backends. Ship with SQLite, PostgreSQL, and in-memory adapters. Bring your own by implementing the adapter interface.

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
