# cognitive-memory

Biologically-inspired agent memory with decay, consolidation, and tiered storage.

[![npm version](https://img.shields.io/npm/v/cognitive-memory.svg)](https://www.npmjs.com/package/cognitive-memory)

TypeScript SDK. v0.4.0 brings deferred conflict resolution that preserves the audit trail (no silent overwrites), and synaptic-tagging weight curve aligned to the paper. Behavioural parity with the Python SDK.

## Install

```bash
npm install cognitive-memory
```

Optional adapters use peer dependencies:

```bash
npm install pg          # for PostgresAdapter
npm install convex      # for ConvexAdapter
```

Requires Node.js 18+ or Bun. ESM-only.

## Quick Start

```typescript
import { CognitiveMemory, InMemoryAdapter, HashEmbeddingProvider } from "cognitive-memory";

const mem = new CognitiveMemory({
  adapter: new InMemoryAdapter(),
  embeddingProvider: new HashEmbeddingProvider(),
  userId: "user-1",
});

await mem.store({
  content: "User is allergic to shellfish",
  category: "core",
  importance: 0.95,
});

const { results } = await mem.search({ query: "what allergies?" });
for (const r of results) {
  console.log(r.memory.content, `(score: ${r.combinedScore.toFixed(2)})`);
}
```

## Production setup

```typescript
import { CognitiveMemory, OpenAIEmbeddingProvider } from "cognitive-memory";
import { PostgresAdapter } from "cognitive-memory/adapters/postgres";
import { Pool } from "pg";

const mem = new CognitiveMemory({
  adapter: new PostgresAdapter({
    pool: new Pool({ connectionString: process.env.DATABASE_URL }),
  }),
  embeddingProvider: new OpenAIEmbeddingProvider({ apiKey: process.env.OPENAI_API_KEY }),
  userId: "user-1",
});

await mem.extractAndStore({ conversationText, sessionId: "sess-1" });

const { results } = await mem.search({
  query: "UI preferences",
  deepRecall: true,
  rerank: true,
  hybridSearch: true,
});
```

`extractAndStore(...)` runs the LLM extractor. `store(...)` skips it for pre-extracted facts.

## Adapters

| Adapter | Import |
| --- | --- |
| `InMemoryAdapter` (default) | `cognitive-memory` |
| `JsonlFileAdapter` (durable, single-process) | `cognitive-memory/adapters/jsonl` |
| `PostgresAdapter` (pgvector) | `cognitive-memory/adapters/postgres` |
| `ConvexAdapter` | `cognitive-memory` |
| Custom | implement `MemoryAdapter` |

`PostgresAdapter` exports `postgresSchemaSql` — run it once to create the `memories` and `memory_links` tables with pgvector indexes.

## Multi-tenancy

`userId` namespaces every read and write. Two instances sharing an adapter are fully isolated:

```typescript
const alice = new CognitiveMemory({ adapter: shared, /* ... */ userId: "alice" });
const bob   = new CognitiveMemory({ adapter: shared, /* ... */ userId: "bob" });
// bob.search() never returns alice's memories
```

## Migration

See [`MIGRATION.md`](./MIGRATION.md) for the 0.3.0 → 0.4.0 behavioural changes (no API impact). Conflict resolution now preserves the audit trail — original memories are demoted, not overwritten, and remain queryable with `contradictedBy` set.

## Docs

Full documentation, guides, concepts, and API reference: **[bhekanik.github.io/cognitive-memory](https://bhekanik.github.io/cognitive-memory)**.

## License

[MIT](../../LICENSE)
