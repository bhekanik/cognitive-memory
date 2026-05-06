import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import type { EmbeddingProvider } from "../src/core/types";
import type { LLMProvider } from "../src/core/extraction";

/**
 * Build an embedding provider that returns vector A for the first call and
 * vector B for the second. Used to drive ingestion-time cosine similarity
 * to a known value.
 */
function pairProvider(a: number[], b: number[]): EmbeddingProvider {
  let i = 0;
  return {
    async embed() {
      const v = i === 0 ? a : b;
      i += 1;
      return v;
    },
  };
}

function twoMemoryLLM(): LLMProvider {
  return {
    async complete() {
      return JSON.stringify([
        { content: "alpha", category: "semantic", importance: 0.5 },
        { content: "beta", category: "semantic", importance: 0.5 },
      ]);
    },
  };
}

async function ingestPair(
  a: number[],
  b: number[],
): Promise<{ adapter: InMemoryAdapter; ids: string[] }> {
  const adapter = new InMemoryAdapter();
  const memory = new CognitiveMemory({
    adapter,
    embeddingProvider: pairProvider(a, b),
    userId: "u1",
    config: { runMaintenanceDuringIngestion: false },
  });
  const ids = await memory.extractAndStore(
    "User: alpha. User: beta.",
    "session-1",
    twoMemoryLLM(),
  );
  return { adapter, ids };
}

describe("ingestion synaptic tagging weight (parity with Python)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  test("sim ≈ 0.5 produces a link with weight ≈ 0.25", async () => {
    // Two unit vectors with cosine = 0.5 (60° apart).
    // Spec formula: min(0.5, 0.2 + (sim - 0.4) * 0.5) = 0.25 at sim = 0.5.
    const { adapter, ids } = await ingestPair(
      [1, 0],
      [0.5, Math.sqrt(0.75)],
    );
    const links = await adapter.getLinkedMemories(ids[0], 0);
    const link = links.find((l) => l.id === ids[1]);
    expect(link).toBeDefined();
    expect(link?.linkStrength).toBeCloseTo(0.25, 4);
  });

  test("sim = 1.0 produces a link with weight 0.5 (cap)", async () => {
    // Identical embeddings → cosine = 1.0.
    // Formula caps at 0.5 even if the linear term would exceed it.
    const { adapter, ids } = await ingestPair([1, 0], [1, 0]);
    const links = await adapter.getLinkedMemories(ids[0], 0);
    const link = links.find((l) => l.id === ids[1]);
    expect(link?.linkStrength).toBeCloseTo(0.5, 4);
  });

  test("sim below 0.4 threshold produces no link", async () => {
    // cosine = 0.39: under both Python's `>=` and any-sane gate, no link.
    const { adapter, ids } = await ingestPair(
      [1, 0],
      [0.39, Math.sqrt(1 - 0.39 * 0.39)],
    );
    const links = await adapter.getLinkedMemories(ids[0], 0);
    expect(links.find((l) => l.id === ids[1])).toBeUndefined();
  });
});
