import { describe, expect, it } from "vitest";
import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import type { EmbeddingProvider } from "../src/core/types";

const embeddingProvider: EmbeddingProvider = {
  async embed() {
    return [1, 0];
  },
};

describe("temporal reconstruction", () => {
  it("returns chronological temporal evidence when auto routing is enabled", async () => {
    const memory = new CognitiveMemory({
      adapter: new InMemoryAdapter(),
      embeddingProvider,
      userId: "u1",
      config: { temporalQueryMode: "auto" },
    });

    await memory.store({
      content: "Alex resumed light running.",
      category: "episodic",
      temporal: {
        mentionedAt: { sessionId: "s3", timestamp: "2024-01-20T00:00:00.000Z" },
        eventTime: { start: "2024-01-20T00:00:00.000Z", confidence: 0.9 },
        validTime: { status: "completed" },
        rawTimeExpressions: ["January 20"],
      },
    });
    await memory.store({
      content: "Alex injured her ankle.",
      category: "episodic",
      temporal: {
        mentionedAt: { sessionId: "s1", timestamp: "2024-01-05T00:00:00.000Z" },
        eventTime: { start: "2024-01-05T00:00:00.000Z", confidence: 0.9 },
        validTime: { status: "completed" },
        rawTimeExpressions: ["January 5"],
      },
    });

    const result = await memory.search({
      query: "What happened after Alex injured her ankle?",
      limit: 2,
    });

    expect(result.temporalEvidence?.map((item) => item.content)).toEqual([
      "Alex injured her ankle.",
      "Alex resumed light running.",
    ]);
  });

  it("keeps current-state evidence in score order for now queries", async () => {
    const memory = new CognitiveMemory({
      adapter: new InMemoryAdapter(),
      embeddingProvider,
      userId: "u1",
      config: { temporalQueryMode: "auto" },
    });

    await memory.store({
      content: "Jamie lives in Manchester.",
      category: "semantic",
      temporal: {
        eventTime: { start: "2024-01-01T00:00:00.000Z", confidence: 0.9 },
        validTime: { status: "superseded", validTo: "2024-03-01T00:00:00.000Z" },
      },
    });
    await memory.store({
      content: "Jamie lives in London.",
      category: "semantic",
      temporal: {
        eventTime: { start: "2024-03-01T00:00:00.000Z", confidence: 0.9 },
        validTime: { status: "current", validFrom: "2024-03-01T00:00:00.000Z" },
      },
    });

    const result = await memory.search({
      query: "Where does Jamie live now?",
      limit: 2,
    });

    expect(result.results[0].memory.content).toBe("Jamie lives in London.");
    expect(result.temporalEvidence?.[0].content).toBe("Jamie lives in London.");
  });

  it("does not fire temporal routing on a non-temporal question that merely mentions time", async () => {
    const memory = new CognitiveMemory({
      adapter: new InMemoryAdapter(),
      embeddingProvider,
      userId: "u1",
      config: { temporalQueryMode: "auto" },
    });

    await memory.store({
      content: "Joanna posted a sunset picture taken on a hike.",
      category: "episodic",
      temporal: { eventTime: { start: "2024-07-01T00:00:00.000Z", confidence: 0.9 } },
    });

    // Asks WHAT she photographed; "last summer" is a modifier, not the intent.
    const result = await memory.search({
      query: "What did Joanna take a picture of near Fort Wayne last summer?",
      limit: 5,
    });

    expect(result.temporalEvidence ?? []).toHaveLength(0);
  });
});
