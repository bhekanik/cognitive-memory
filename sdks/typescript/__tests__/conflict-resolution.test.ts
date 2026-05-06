import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import type { EmbeddingProvider, Memory } from "../src/core/types";
import { createDefaultMemory } from "../src/core/types";
import type { LLMProvider } from "../src/core/extraction";

function providerFromMap(map: Map<string, number[]>): EmbeddingProvider {
  return {
    async embed(text: string) {
      const v = map.get(text);
      if (!v) throw new Error(`missing embedding for: ${text}`);
      return v;
    },
  };
}

function dispatchingLLM(
  conflictReply: string,
  extractedContent: string,
  category: Memory["category"] = "semantic",
  importance = 0.5,
): LLMProvider {
  return {
    async complete(prompt: string) {
      if (prompt.startsWith("Does the new memory contradict")) {
        return conflictReply;
      }
      // Extraction prompt
      return JSON.stringify([
        { content: extractedContent, category, importance },
      ]);
    },
  };
}

async function seedExistingMemory(
  adapter: InMemoryAdapter,
  opts: {
    content: string;
    embedding: number[];
    category?: Memory["category"];
    importance?: number;
    sessionIds?: string[];
  },
): Promise<string> {
  return adapter.createMemory({
    ...createDefaultMemory({
      id: "tmp",
      userId: "u1",
      content: opts.content,
      embedding: opts.embedding,
    }),
    category: opts.category ?? "semantic",
    importance: opts.importance ?? 0.5,
    sessionIds: opts.sessionIds ?? ["session-old"],
  });
}

describe("conflict resolution at tick", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  test("CONTRADICTION preserves the existing memory's content (audit trail)", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["Alex prefers tea", [1, 0]],
      ["Alex prefers coffee", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    const existingId = await seedExistingMemory(adapter, {
      content: "Alex prefers tea",
      embedding: [1, 0],
    });

    const llm = dispatchingLLM("CONTRADICTION", "Alex prefers coffee");
    const [storedId] = await memory.extractAndStore(
      "User: I prefer coffee now",
      "session-new",
      llm,
    );
    await memory.tick(llm);

    const existing = await adapter.getMemory(existingId);
    expect(existing).not.toBeNull();
    expect(existing?.content).toBe("Alex prefers tea");
    // Positive evidence the conflict was actually resolved (so this test isn't
    // satisfied by a no-op resolveConflictQueue):
    expect(existing?.contradictedBy).toBe(storedId);
  });

  test("CONTRADICTION on a core memory demotes existing to semantic and promotes new to core", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["I am vegan", [1, 0]],
      ["I eat meat now", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    const existingId = await seedExistingMemory(adapter, {
      content: "I am vegan",
      embedding: [1, 0],
      category: "core",
      importance: 0.9,
    });

    const llm = dispatchingLLM("CONTRADICTION", "I eat meat now");
    const [storedId] = await memory.extractAndStore(
      "User: I eat meat now",
      "session-new",
      llm,
    );
    await memory.tick(llm);

    const existing = await adapter.getMemory(existingId);
    const replacement = await adapter.getMemory(storedId);

    expect(existing?.category).toBe("semantic");
    expect(replacement?.category).toBe("core");
  });

  test("CONTRADICTION lifts the new memory's importance to max(existing, new)", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["important fact", [1, 0]],
      ["replacement", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    await seedExistingMemory(adapter, {
      content: "important fact",
      embedding: [1, 0],
      importance: 0.9,
    });

    // New memory's extracted importance is intentionally low (0.3); after
    // resolution it should be lifted to 0.9 (the existing memory's importance).
    const llm = dispatchingLLM("CONTRADICTION", "replacement", "semantic", 0.3);
    const [storedId] = await memory.extractAndStore(
      "User: replacement",
      "session-new",
      llm,
    );
    await memory.tick(llm);

    const replacement = await adapter.getMemory(storedId);
    expect(replacement?.importance).toBe(0.9);
  });

  test("UPDATE does NOT promote the new memory to core even if existing was core", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["I live in NYC", [1, 0]],
      ["I live in SF now", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    const existingId = await seedExistingMemory(adapter, {
      content: "I live in NYC",
      embedding: [1, 0],
      category: "core",
    });

    // UPDATE = "newer version of same fact", not contradiction.
    // Existing should still demote to semantic (audit trail), but new should
    // NOT auto-promote to core — the user's life-fact is updated, not
    // architecturally elevated.
    const llm = dispatchingLLM("UPDATE", "I live in SF now");
    const [storedId] = await memory.extractAndStore(
      "User: I live in SF now",
      "session-new",
      llm,
    );
    await memory.tick(llm);

    const existing = await adapter.getMemory(existingId);
    const replacement = await adapter.getMemory(storedId);

    expect(existing?.category).toBe("semantic"); // demoted
    expect(existing?.contradictedBy).toBe(storedId); // audit trail set
    expect(replacement?.category).not.toBe("core"); // NOT promoted
  });

  test("OVERLAP and NONE leave both memories untouched", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["fact A", [1, 0]],
      ["fact B", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    const existingId = await seedExistingMemory(adapter, {
      content: "fact A",
      embedding: [1, 0],
      category: "core",
      importance: 0.7,
    });

    const llm = dispatchingLLM("OVERLAP", "fact B");
    const [storedId] = await memory.extractAndStore(
      "User: fact B",
      "session-new",
      llm,
    );
    await memory.tick(llm);

    const existing = await adapter.getMemory(existingId);
    const replacement = await adapter.getMemory(storedId);

    expect(existing?.category).toBe("core");
    expect(existing?.content).toBe("fact A");
    expect(existing?.contradictedBy).toBeNull();
    expect(replacement?.category).toBe("semantic"); // as extracted
  });
});
