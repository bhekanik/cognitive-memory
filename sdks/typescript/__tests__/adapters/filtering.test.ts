import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MemoryAdapter } from "../../src/adapters/base";
import { InMemoryAdapter } from "../../src/adapters/memory";
import { JsonlFileAdapter } from "../../src/adapters/jsonl";

async function seedTieredMemories(adapter: MemoryAdapter) {
  const hotId = await adapter.createMemory({
    userId: "u1",
    content: "hot",
    embedding: [1, 0],
    category: "semantic",
    importance: 0.5,
    stability: 0.5,
    accessCount: 0,
    lastAccessed: 1,
    retention: 1,
    metadata: {},
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: "fact",
    validFrom: null,
    validUntil: null,
    ttlSeconds: null,
    sourceTurnIds: [],
  });
  const coldId = await adapter.createMemory({
    userId: "u1",
    content: "cold",
    embedding: [1, 0],
    category: "semantic",
    importance: 0.5,
    stability: 0.5,
    accessCount: 0,
    lastAccessed: 1,
    retention: 1,
    metadata: {},
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: "fact",
    validFrom: null,
    validUntil: null,
    ttlSeconds: null,
    sourceTurnIds: [],
  });
  const supersededId = await adapter.createMemory({
    userId: "u1",
    content: "superseded",
    embedding: [1, 0],
    category: "semantic",
    importance: 0.5,
    stability: 0.5,
    accessCount: 0,
    lastAccessed: 1,
    retention: 1,
    metadata: {},
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: "fact",
    validFrom: null,
    validUntil: null,
    ttlSeconds: null,
    sourceTurnIds: [],
  });
  const stubId = await adapter.createMemory({
    userId: "u1",
    content: "stub",
    embedding: [1, 0],
    category: "semantic",
    importance: 0.5,
    stability: 0.5,
    accessCount: 0,
    lastAccessed: 1,
    retention: 1,
    metadata: {},
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: "fact",
    validFrom: null,
    validUntil: null,
    ttlSeconds: null,
    sourceTurnIds: [],
  });

  await adapter.migrateToCold(coldId, 2);
  await adapter.markSuperseded([supersededId], "summary");
  await adapter.convertToStub(stubId, "stub summary");

  return { hotId, coldId, supersededId, stubId };
}

describe.each([
  ["InMemoryAdapter", async () => new InMemoryAdapter() as MemoryAdapter],
  [
    "JsonlFileAdapter",
    async () => {
      const dir = await mkdtemp(join(tmpdir(), "cm-jsonl-filtering-"));
      return new JsonlFileAdapter({ path: join(dir, "memories.jsonl") }) as MemoryAdapter;
    },
  ],
])("%s filtering", (_name, buildAdapter) => {
  test("normal search excludes cold, superseded, and stub memories", async () => {
    const adapter = await buildAdapter();
    await seedTieredMemories(adapter);

    const results = await adapter.vectorSearch([1, 0], {
      userId: "u1",
      limit: 10,
    });

    expect(results.map((m) => m.content)).toEqual(["hot"]);
  });

  test("cold memories are searchable only when explicitly included", async () => {
    const adapter = await buildAdapter();
    await seedTieredMemories(adapter);

    const results = await adapter.vectorSearch([1, 0], {
      userId: "u1",
      includeCold: true,
      limit: 10,
    });

    expect(results.map((m) => m.content).sort()).toEqual(["cold", "hot"]);
  });

  test("deep recall can include superseded memories without returning stubs", async () => {
    const adapter = await buildAdapter();
    await seedTieredMemories(adapter);

    const results = await adapter.vectorSearch([1, 0], {
      userId: "u1",
      includeCold: true,
      includeSuperseded: true,
      limit: 10,
    });

    expect(results.map((m) => m.content).sort()).toEqual([
      "cold",
      "hot",
      "superseded",
    ]);
  });
});
