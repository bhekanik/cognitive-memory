# cognitive-memory

Biologically-inspired agent memory with decay, consolidation, and tiered storage.

[![npm version](https://img.shields.io/npm/v/cognitive-memory.svg)](https://www.npmjs.com/package/cognitive-memory)

## Install

```bash
npm install cognitive-memory
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

For production, use `OpenAIEmbeddingProvider` and `PostgresAdapter`:

```typescript
import { OpenAIEmbeddingProvider } from "cognitive-memory";
import { PostgresAdapter } from "cognitive-memory/adapters/postgres";
import { Pool } from "pg";

const mem = new CognitiveMemory({
  adapter: new PostgresAdapter({
    pool: new Pool({ connectionString: process.env.DATABASE_URL }),
  }),
  embeddingProvider: new OpenAIEmbeddingProvider({ apiKey: process.env.OPENAI_API_KEY }),
  userId: "user-1",
});
```

## Docs

Full documentation, guides, and API reference at **[bhekanik.github.io/cognitive-memory](https://bhekanik.github.io/cognitive-memory)**.

## License

[MIT](../../LICENSE)
