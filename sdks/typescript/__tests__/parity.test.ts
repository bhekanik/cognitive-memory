/**
 * Cross-SDK parity tests.
 *
 * Both SDKs (Python and TypeScript) read the same scripted scenario JSON,
 * run it through the public API with deterministic embedders, and assert
 * on the observable result. The expected block in each scenario is the
 * shared oracle — if a behaviour diverges between SDKs, this test fails
 * on at least one side.
 *
 * Scenarios live in cognitive-memory-sdk/tests/parity-fixtures/.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import { HashEmbeddingProvider } from "../src/core/embeddings";
import type { MemoryCategory } from "../src/core/types";

const FIXTURES = path.resolve(__dirname, "..", "..", "..", "tests", "parity-fixtures");

const T0 = new Date("2026-01-01T00:00:00.000Z").getTime();

interface ScenarioEvent {
  t_seconds: number;
  op: "add" | "search";
  content?: string;
  category?: string;
  importance?: number;
  query?: string;
  limit?: number;
}

interface Scenario {
  description: string;
  user_id: string;
  embedder: "hash";
  events: ScenarioEvent[];
  expected: {
    memory_count: number;
    categories: Record<string, number>;
    search_top_content: string | null;
    search_top_category: string | null;
  };
}

interface Snapshot {
  memory_count: number;
  categories: Record<string, number>;
  search_top_content: string | null;
  search_top_category: string | null;
}

async function runScenario(scenario: Scenario): Promise<Snapshot> {
  const adapter = new InMemoryAdapter();
  const memory = new CognitiveMemory({
    adapter,
    embeddingProvider: new HashEmbeddingProvider({ dimensions: 64 }),
    userId: scenario.user_id,
    config: { runMaintenanceDuringIngestion: false },
  });

  let topContent: string | null = null;
  let topCategory: string | null = null;

  for (const event of scenario.events) {
    const tsMs = T0 + event.t_seconds * 1000;
    vi.setSystemTime(new Date(tsMs));

    if (event.op === "add") {
      await memory.store({
        content: event.content!,
        category: event.category as MemoryCategory,
        importance: event.importance,
      });
    } else if (event.op === "search") {
      const response = await memory.search({
        query: event.query!,
        limit: event.limit ?? 5,
      });
      const top = response.results[0]?.memory;
      topContent = top?.content ?? null;
      topCategory = top?.category ?? null;
    }
  }

  const all = await adapter.allActive();
  const filtered = all.filter((m) => m.userId === scenario.user_id);
  const categories: Record<string, number> = {};
  for (const m of filtered) {
    categories[m.category] = (categories[m.category] ?? 0) + 1;
  }

  return {
    memory_count: filtered.length,
    categories,
    search_top_content: topContent,
    search_top_category: topCategory,
  };
}

describe("cross-SDK parity", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  test("scenario A: three manual ingests + one search produce the expected snapshot", async () => {
    const scenario: Scenario = JSON.parse(
      readFileSync(path.join(FIXTURES, "scenario-a.json"), "utf-8"),
    );
    const snapshot = await runScenario(scenario);
    const expected = scenario.expected;

    expect(snapshot.memory_count).toBe(expected.memory_count);
    expect(snapshot.categories).toEqual(expected.categories);
    expect(snapshot.search_top_content).toBe(expected.search_top_content);
    expect(snapshot.search_top_category).toBe(expected.search_top_category);
  });
});
