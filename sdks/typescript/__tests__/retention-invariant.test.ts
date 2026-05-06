import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import { CognitiveEngine } from "../src/core/engine";
import type { EmbeddingProvider, MemoryCategory } from "../src/core/types";
import { createDefaultMemory, resolveConfig } from "../src/core/types";

function providerFromMap(map: Map<string, number[]>): EmbeddingProvider {
  return {
    async embed(text: string) {
      const v = map.get(text);
      if (!v) throw new Error(`missing embedding for: ${text}`);
      return v;
    },
  };
}

describe("retention materialisation invariant", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  test("memory.retention equals engine.computeRetention after a search() pass", async () => {
    // Materialised retention (TS-only field) must stay in sync with the
    // engine's on-the-fly compute. If a future change writes one and not the
    // other, this test fails.
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["q", [1, 0]],
      ["target", [1, 0]],
    ]);
    const config = resolveConfig({ userId: "u1" });
    const engine = new CognitiveEngine(adapter, config);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });

    const now = Date.now();
    await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "target",
        embedding: [1, 0],
      }),
      category: "semantic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 5 * 24 * 60 * 60 * 1000,
      retention: 1.0, // intentionally stale; search() should refresh it
    });

    await memory.search({ query: "q", limit: 1 });

    // Advance time so retention drifts from 1.0; without a refresh path the
    // materialised field would still report 1.0 and the test would fail.
    vi.setSystemTime(new Date(Date.now() + 60 * 24 * 60 * 60 * 1000));
    await memory.refreshRetentionScores();

    const all = await adapter.allActive();
    for (const m of all) {
      const expected = engine.computeRetention(m, Date.now());
      expect(m.retention).toBeCloseTo(expected, 6);
      expect(m.retention).toBeLessThan(1.0); // sanity: actually drifted
    }
  });
});
