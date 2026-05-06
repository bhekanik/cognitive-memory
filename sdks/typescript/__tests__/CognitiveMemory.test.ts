import { InMemoryAdapter } from "../src/adapters/memory";
import { MemoryAdapter, type MemoryFilters } from "../src/adapters/base";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import type {
  EmbeddingProvider,
  Memory,
  MemoryCategory,
  ScoredMemory,
} from "../src/core/types";
import { createDefaultMemory } from "../src/core/types";

function providerFromMap(map: Map<string, number[]>): EmbeddingProvider {
  return {
    async embed(text: string) {
      const v = map.get(text);
      if (!v) throw new Error(`missing embedding for: ${text}`);
      return v;
    },
  };
}

class AdapterWithoutSyncMaps extends MemoryAdapter {
  constructor(private readonly inner: InMemoryAdapter) {
    super();
  }

  createMemory(memory: Omit<Memory, "id" | "createdAt" | "updatedAt">) {
    return this.inner.createMemory(memory);
  }

  getMemory(id: string) {
    return this.inner.getMemory(id);
  }

  getMemories(ids: string[]) {
    return this.inner.getMemories(ids);
  }

  queryMemories(filters: MemoryFilters) {
    return this.inner.queryMemories(filters);
  }

  updateMemory(id: string, updates: Partial<Memory>) {
    return this.inner.updateMemory(id, updates);
  }

  deleteMemory(id: string) {
    return this.inner.deleteMemory(id);
  }

  deleteMemories(ids: string[]) {
    return this.inner.deleteMemories(ids);
  }

  async vectorSearch(embedding: number[], filters?: MemoryFilters): Promise<ScoredMemory[]> {
    const results = await this.inner.vectorSearch(embedding, filters);
    return results.filter((memory) => memory.content === "anchor");
  }

  searchLexical(query: string, filters?: MemoryFilters) {
    return this.inner.searchLexical(query, filters);
  }

  updateRetentionScores(updates: Map<string, number>) {
    return this.inner.updateRetentionScores(updates);
  }

  createOrStrengthenLink(sourceId: string, targetId: string, strength: number) {
    return this.inner.createOrStrengthenLink(sourceId, targetId, strength);
  }

  getLinkedMemories(memoryId: string, minStrength?: number) {
    return this.inner.getLinkedMemories(memoryId, minStrength);
  }

  getLinkedMemoriesMultiple(memoryIds: string[], minStrength?: number) {
    return this.inner.getLinkedMemoriesMultiple(memoryIds, minStrength);
  }

  deleteLink(sourceId: string, targetId: string) {
    return this.inner.deleteLink(sourceId, targetId);
  }

  findFadingMemories(userId: string, maxRetention: number) {
    return this.inner.findFadingMemories(userId, maxRetention);
  }

  findStableMemories(userId: string, minStability: number, minAccessCount: number) {
    return this.inner.findStableMemories(userId, minStability, minAccessCount);
  }

  markSuperseded(memoryIds: string[], summaryId: string) {
    return this.inner.markSuperseded(memoryIds, summaryId);
  }

  migrateToCold(memoryId: string, coldSince: number) {
    return this.inner.migrateToCold(memoryId, coldSince);
  }

  migrateToHot(memoryId: string) {
    return this.inner.migrateToHot(memoryId);
  }

  convertToStub(memoryId: string, stubContent: string) {
    return this.inner.convertToStub(memoryId, stubContent);
  }

  allActive() {
    return this.inner.allActive();
  }

  allHot() {
    return this.inner.allHot();
  }

  allCold() {
    return this.inner.allCold();
  }

  hotCount() {
    return this.inner.hotCount();
  }

  coldCount() {
    return this.inner.coldCount();
  }

  stubCount() {
    return this.inner.stubCount();
  }

  totalCount() {
    return this.inner.totalCount();
  }

  clear() {
    return this.inner.clear();
  }

  transaction<T>(callback: (adapter: MemoryAdapter) => Promise<T>) {
    return callback(this);
  }
}

class FixedVectorAdapter extends MemoryAdapter {
  readonly memories: Memory[];

  constructor(count: number) {
    super();
    this.memories = Array.from({ length: count }, (_, i) =>
      createDefaultMemory({
        id: `m${i}`,
        userId: "u1",
        content: `memory-${i}`,
        embedding: [1, 0],
        retention: 1,
      }),
    );
  }

  async createMemory(): Promise<string> {
    throw new Error("not used");
  }

  async getMemory(id: string) {
    return this.memories.find((m) => m.id === id) ?? null;
  }

  async getMemories(ids: string[]) {
    return this.memories.filter((m) => ids.includes(m.id));
  }

  async queryMemories() {
    return this.memories;
  }

  async updateMemory(): Promise<void> {}

  async deleteMemory(): Promise<void> {}

  async deleteMemories(): Promise<void> {}

  async vectorSearch(_embedding: number[], filters?: MemoryFilters): Promise<ScoredMemory[]> {
    return this.memories.slice(0, filters?.limit ?? 5).map((memory, index) => ({
      ...memory,
      relevanceScore: 1 - index / 100,
      finalScore: 1 - index / 100,
    }));
  }

  async updateRetentionScores(): Promise<void> {}

  async createOrStrengthenLink(): Promise<void> {}

  async getLinkedMemories() {
    return [];
  }

  async getLinkedMemoriesMultiple() {
    return [];
  }

  async deleteLink(): Promise<void> {}

  async findFadingMemories() {
    return [];
  }

  async findStableMemories() {
    return [];
  }

  async markSuperseded(): Promise<void> {}

  async migrateToCold(): Promise<void> {}

  async migrateToHot(): Promise<void> {}

  async convertToStub(): Promise<void> {}

  async allActive() {
    return this.memories;
  }

  async allHot() {
    return this.memories;
  }

  async allCold() {
    return [];
  }

  async hotCount() {
    return this.memories.length;
  }

  async coldCount() {
    return 0;
  }

  async stubCount() {
    return 0;
  }

  async totalCount() {
    return this.memories.length;
  }

  async clear(): Promise<void> {}

  async transaction<T>(callback: (adapter: MemoryAdapter) => Promise<T>) {
    return callback(this);
  }
}

class MaintenanceCountingAdapter extends InMemoryAdapter {
  allHotCalls = 0;

  override async allHot(): Promise<Memory[]> {
    this.allHotCalls += 1;
    return super.allHot();
  }
}

describe("CognitiveMemory", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("store() applies defaults", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["a", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });

    const id = await memory.store({ content: "a" });
    const m = await adapter.getMemory(id);
    expect(m?.category).toBe("semantic");
    expect(m?.importance).toBe(0.5);
    expect(m?.stability).toBe(0.3);
    expect(m?.accessCount).toBe(0);
    expect(m?.retention).toBe(1.0);
  });

  test("retrieve() scores by relevance * retention and strengthens memories + links", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["q", [1, 0]],
      ["A", [1, 0]],
      ["B", [1, 0]],
      ["C", [0, 1]],
    ]);

    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });

    const now = Date.now();
    const aId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "A",
        embedding: embeddings.get("A")!,
      }),
      category: "episodic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 1 * 24 * 60 * 60 * 1000,
      retention: 1,
    });
    const bId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "B",
        embedding: embeddings.get("B")!,
      }),
      category: "episodic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 30 * 24 * 60 * 60 * 1000,
      retention: 1,
    });
    const cId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "C",
        embedding: embeddings.get("C")!,
      }),
      category: "semantic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 1 * 24 * 60 * 60 * 1000,
      retention: 1,
    });

    await adapter.createOrStrengthenLink(aId, cId, 0.4);

    const results = await memory.retrieve({
      query: "q",
      limit: 3,
      includeAssociations: true,
    });
    expect(results[0].id).toBe(aId);
    expect(results.some((r) => r.id === cId)).toBe(true);

    const a = await adapter.getMemory(aId);
    expect(a?.accessCount).toBe(1);
    expect(a?.stability).toBeGreaterThan(0.5);
  });

  test("retrieve() uses the same v6 scoring semantics as search()", async () => {
    async function buildMemorySystem() {
      const adapter = new InMemoryAdapter();
      const embeddings = new Map<string, number[]>([
        ["q", [1, 0]],
        ["fresh", [0.8, 0.6]],
        ["faded", [1, 0]],
      ]);
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: providerFromMap(embeddings),
        userId: "u1",
        config: {
          minRetention: 0,
          regularRetentionFloor: 0.02,
          retrievalScoreExponent: 0.3,
          decayRates: { semantic: 1 },
          runMaintenanceDuringIngestion: false,
        },
      });
      const now = Date.now();

      await adapter.createMemory({
        ...createDefaultMemory({
          id: "tmp",
          userId: "u1",
          content: "fresh",
          embedding: embeddings.get("fresh")!,
        }),
        category: "semantic",
        importance: 0.5,
        stability: 1,
        lastAccessed: now,
      });
      await adapter.createMemory({
        ...createDefaultMemory({
          id: "tmp",
          userId: "u1",
          content: "faded",
          embedding: embeddings.get("faded")!,
        }),
        category: "semantic",
        importance: 0,
        stability: 0.01,
        lastAccessed: now - 365 * 24 * 60 * 60 * 1000,
      });

      return memory;
    }

    const retrieved = await (await buildMemorySystem()).retrieve({
      query: "q",
      limit: 2,
      minRetention: 0,
    });
    const searched = await (await buildMemorySystem()).search({
      query: "q",
      limit: 2,
      deepRecall: false,
    });

    expect(retrieved.map((r) => r.content)).toEqual(
      searched.results.map((r) => r.memory.content),
    );
    expect(retrieved[0].finalScore).toBeCloseTo(
      searched.results[0].combinedScore,
      6,
    );

    const faded = retrieved.find((r) => r.content === "faded");
    expect(faded?.retention).toBeCloseTo(0.02, 6);
    expect(faded?.relevanceScore).toBeCloseTo(1, 6);
    expect(faded?.finalScore).toBeCloseTo(0.309, 3);
  });

  test("search() follows memory associations through the adapter interface", async () => {
    const inner = new InMemoryAdapter();
    const adapter = new AdapterWithoutSyncMaps(inner);
    const embeddings = new Map<string, number[]>([
      ["q", [1, 0]],
      ["anchor", [1, 0]],
      ["associated", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: {
        minRetention: 0,
        associationRetrievalThreshold: 0.3,
        graphExpansionHops: 0,
        runMaintenanceDuringIngestion: false,
      },
    });
    const now = Date.now();
    const associatedId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "associated",
        embedding: embeddings.get("associated")!,
      }),
      lastAccessed: now,
    });
    await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "anchor",
        embedding: embeddings.get("anchor")!,
      }),
      lastAccessed: now,
      associations: {
        [associatedId]: {
          targetId: associatedId,
          weight: 0.8,
          lastCoRetrieval: null,
          createdAt: now,
        },
      },
    });

    const results = await memory.search({
      query: "q",
      limit: 2,
    });

    expect(results.results.map((r) => r.memory.content)).toContain("anchor");
    expect(results.results.map((r) => r.memory.content)).toContain("associated");
  });

  test("rerankFactor controls candidate count and final order", async () => {
    const adapter = new FixedVectorAdapter(30);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(new Map([["q", [1, 0]]])),
      userId: "u1",
      config: {
        rerankEnabled: true,
        rerankFactor: 3,
        minRetention: 0,
        graphExpansionHops: 0,
        runMaintenanceDuringIngestion: false,
      },
    });
    const reranked = Array.from({ length: 10 }, (_, i) => 29 - i);
    const llm = {
      async complete() {
        return JSON.stringify(reranked);
      },
      async completeWithUsage() {
        return {
          text: JSON.stringify(reranked),
          usage: { promptTokens: 123, completionTokens: 4 },
        };
      },
    };

    const response = await memory.search(
      { query: "q", limit: 10, trace: true },
      llm,
    );

    expect(response.trace?.stages.rerank.candidateCount).toBe(30);
    expect(response.trace?.totalTokens).toBe(127);
    expect(response.results.map((r) => r.memory.content)).toEqual(
      reranked.map((i) => `memory-${i}`),
    );
  });

  test("extractAndStore reinforces similar existing memories during ingestion", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["existing fact", [1, 0]],
      ["new fact", [1, 0]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
      config: { runMaintenanceDuringIngestion: false },
    });
    const existingId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "existing fact",
        embedding: embeddings.get("existing fact")!,
      }),
      stability: 0.3,
      sessionIds: ["old-session"],
    });
    const llm = {
      async complete() {
        return JSON.stringify([
          { content: "new fact", category: "semantic", importance: 0.5 },
        ]);
      },
    };

    await memory.extractAndStore("User: new fact", "new-session", llm);

    const existing = await adapter.getMemory(existingId);
    expect(existing?.stability).toBeCloseTo(0.35, 6);
  });

  test("extractAndStore runs automatic maintenance every fifth ingestion", async () => {
    const adapter = new MaintenanceCountingAdapter();
    let count = 0;
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: {
        async embed() {
          return [1, 0];
        },
      },
      userId: "u1",
    });
    const llm = {
      async complete() {
        count += 1;
        return JSON.stringify([
          { content: `fact ${count}`, category: "semantic", importance: 0.5 },
        ]);
      },
    };

    for (let i = 0; i < 4; i++) {
      await memory.extractAndStore(`User: fact ${i}`, `s${i}`, llm);
    }
    expect(adapter.allHotCalls).toBe(0);

    await memory.extractAndStore("User: fact 5", "s5", llm);
    expect(adapter.allHotCalls).toBeGreaterThan(0);
  });

  test("get() strengthens a memory", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: embeddings.get("x")!,
      }),
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Date.now() - 10_000,
      retention: 1,
    });
    await memory.get(id);
    const m = await adapter.getMemory(id);
    expect(m?.accessCount).toBe(1);
  });

  test("queryMemories() strengthens returned memories", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: embeddings.get("x")!,
      }),
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Date.now() - 10_000,
      retention: 1,
    });

    await memory.queryMemories({ limit: 10 });
    const m = await adapter.getMemory(id);
    expect(m?.accessCount).toBe(1);
  });

  test("update() regenerates embedding", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["old", [1, 0]],
      ["new", [0, 1]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await memory.store({ content: "old" });
    await memory.update(id, "new");
    const m = await adapter.getMemory(id);
    expect(m?.embedding).toEqual([0, 1]);
  });

  test("consolidate() compresses groups and deletes stale", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    const now = Date.now();
    for (const c of [
      "coffee a",
      "coffee b",
      "coffee c",
      "coffee d",
      "coffee e",
    ]) {
      await adapter.createMemory({
        ...createDefaultMemory({
          id: "tmp",
          userId: "u1",
          content: c,
          embedding: [1, 0],
        }),
        stability: 0.3,
        accessCount: 0,
        lastAccessed: now - 200 * 24 * 60 * 60 * 1000,
        retention: 0.1,
      });
    }

    const staleId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "s",
        embedding: [1, 0],
      }),
      stability: 0.05,
      importance: 0.1,
      accessCount: 0,
      lastAccessed: now - 200 * 24 * 60 * 60 * 1000,
      retention: 0.01,
    });

    const result = await memory.consolidate();
    expect(result.compressed.length).toBe(1);
    expect(result.deleted).toBe(1);
    expect(await adapter.getMemory(staleId)).toBeNull();
  });

  test("consolidate() refreshes retention before finding fading memories", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [0, 0] },
      userId: "u1",
    });

    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: [1, 0],
      }),
      stability: 0.3,
      accessCount: 0,
      lastAccessed: Date.now() - 150 * 24 * 60 * 60 * 1000,
      retention: 1,
    });

    const result = await memory.consolidate();
    expect(result.decayed.map((d) => d.id)).toContain(id);
  });

  test("link() validates strength", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    await expect(memory.link("a", "b", 2)).rejects.toThrow(/Invalid strength/);
  });

  test("store() retries embedding up to 3 attempts", async () => {
    const adapter = new InMemoryAdapter();
    const embed = vi
      .fn<
        Parameters<EmbeddingProvider["embed"]>,
        ReturnType<EmbeddingProvider["embed"]>
      >()
      .mockRejectedValueOnce(new Error("rate limit"))
      .mockRejectedValueOnce(new Error("transient"))
      .mockResolvedValue([1, 0]);

    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed },
      userId: "u1",
    });

    const p = memory.store({ content: "x" });
    await vi.runAllTimersAsync();
    await p;
    expect(embed).toHaveBeenCalledTimes(3);
  });

  test("store() fails after 3 embedding attempts", async () => {
    const adapter = new InMemoryAdapter();
    const embed = vi.fn().mockRejectedValue(new Error("down"));
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed },
      userId: "u1",
    });

    const p = memory.store({ content: "x" });
    const ex = expect(p).rejects.toThrow(/Embedding failed/);
    await vi.runAllTimersAsync();
    await ex;
  });

  test("get() throws on invalid lastAccessed", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: [1, 0],
      }),
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Number.NaN,
      retention: 1,
    });

    await expect(memory.get(id)).rejects.toThrow(/Invalid lastAccessed/);
  });

  test("getStats() returns correct counts", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    await memory.store({ content: "hello", category: "semantic" });
    await memory.store({ content: "world", category: "core" });

    const stats = await memory.getStats();
    expect(stats.total).toBe(2);
    expect(stats.hot).toBe(2);
    expect(stats.cold).toBe(0);
    expect(stats.stub).toBe(0);
    expect(stats.core).toBe(1);
  });

  test("clear() removes all memories", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    await memory.store({ content: "hello" });
    await memory.store({ content: "world" });
    await memory.clear();

    const stats = await memory.getStats();
    expect(stats.total).toBe(0);
  });

  describe("extractionMode", () => {
    const fakeLlm = {
      async complete() {
        return JSON.stringify([
          { content: "Ross is a paleontologist", category: "core", importance: 0.9 },
          { content: "Rachel said nice to meet you", category: "episodic", importance: 0.3 },
        ]);
      },
    };

    test("raw mode stores turns verbatim without LLM", async () => {
      const adapter = new InMemoryAdapter();
      const embeddings = new Map<string, number[]>([
        ["Ross: I got you a present. It is a Slinky!", [1, 0]],
        ["Rachel: A Slinky? That is so thoughtful.", [0.9, 0.1]],
        ["Joey: Who wants pizza?", [0, 1]],
      ]);
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: providerFromMap(embeddings),
        userId: "u1",
        config: { extractionMode: "raw" },
      });

      const conversation =
        "[This conversation took place on 2024-12-14]\n" +
        "Ross: I got you a present. It is a Slinky!\n" +
        "Rachel: A Slinky? That is so thoughtful.\n" +
        "Joey: Who wants pizza?";

      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // Should store 3 raw turns (header skipped), no LLM call
      expect(ids).toHaveLength(3);

      const stats = await memory.getStats();
      expect(stats.total).toBe(3);

      // Verify content is verbatim
      const m = await adapter.getMemory(ids[0]);
      expect(m?.content).toContain("Slinky");
      expect(m?.stability).toBe(0.2); // raw turns get stability 0.2
    });

    test("hybrid mode stores both extracted facts and raw turns", async () => {
      const adapter = new InMemoryAdapter();
      const embeddingCalls: string[] = [];
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: {
          async embed(text: string) {
            embeddingCalls.push(text);
            return [Math.random(), Math.random()];
          },
        },
        userId: "u1",
        config: { extractionMode: "hybrid" },
      });

      const conversation =
        "Ross: My name is Ross and I am a paleontologist.\n" +
        "Rachel: Nice to meet you Ross!";

      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // 2 from LLM extraction + 2 raw turns = 4
      expect(ids.length).toBeGreaterThanOrEqual(4);

      const stats = await memory.getStats();
      expect(stats.total).toBeGreaterThanOrEqual(4);
    });

    test("semantic mode (default) does not store raw turns", async () => {
      const adapter = new InMemoryAdapter();
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: { embed: async () => [1, 0] },
        userId: "u1",
        // extractionMode defaults to "semantic"
      });

      const conversation = "User: My name is Alice.\nAssistant: Hello Alice!";
      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // Should only have extracted facts from LLM (2), no raw turns
      expect(ids).toHaveLength(2);

      // Verify none have stability 0.2 (raw turn marker)
      for (const id of ids) {
        const m = await adapter.getMemory(id);
        expect(m?.stability).not.toBe(0.2);
      }
    });

    test("invalid extractionMode throws", async () => {
      const adapter = new InMemoryAdapter();
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: { embed: async () => [1, 0] },
        userId: "u1",
        config: { extractionMode: "invalid" as any },
      });

      await expect(
        memory.extractAndStore("User: hello", "s1", fakeLlm),
      ).rejects.toThrow(/Invalid extractionMode/);
    });
  });
});
