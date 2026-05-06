/**
 * Adapters must throw typed errors on contract violations
 * (spec/adapter-interface.md, Implementation Note 1).
 */

import { InMemoryAdapter } from "../src/adapters/memory";
import { AdapterError, MemoryNotFoundError } from "../src/adapters/errors";

describe("typed adapter errors", () => {
  test("updateMemory of an unknown id throws MemoryNotFoundError", async () => {
    const adapter = new InMemoryAdapter();
    await expect(
      adapter.updateMemory("not-a-real-id", { stability: 0.5 }),
    ).rejects.toBeInstanceOf(MemoryNotFoundError);
  });

  test("MemoryNotFoundError carries the missing id for debuggability", async () => {
    const adapter = new InMemoryAdapter();
    try {
      await adapter.updateMemory("ghost-id", { stability: 0.5 });
      throw new Error("expected throw");
    } catch (err) {
      expect(err).toBeInstanceOf(MemoryNotFoundError);
      expect((err as MemoryNotFoundError).memoryId).toBe("ghost-id");
      expect((err as Error).message).toContain("ghost-id");
    }
  });

  test("MemoryNotFoundError extends AdapterError so a single catch covers all", () => {
    const err = new MemoryNotFoundError("x");
    expect(err).toBeInstanceOf(AdapterError);
    expect(err).toBeInstanceOf(Error);
  });

  test("createMemory rejects an id that already exists in any tier", async () => {
    // Force a deterministic id collision. With a normal random idFactory,
    // collisions are vanishingly rare; with composition/wrapping/replay,
    // they can happen. The defensive guard catches all of those.
    let calls = 0;
    const idFactory = () => {
      calls += 1;
      return calls <= 2 ? "fixed-id" : `id-${calls}`;
    };
    const adapter = new InMemoryAdapter({ idFactory });

    const baseInput = {
      userId: "u1",
      content: "original",
      embedding: [1, 0],
      category: "semantic" as const,
      importance: 0.5,
      stability: 0.3,
      accessCount: 0,
      lastAccessed: Date.now(),
      retention: 1.0,
      associations: {},
      sessionIds: [],
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
      isSuperseded: false,
      supersededBy: null,
      isStub: false,
      contradictedBy: null,
    };

    await adapter.createMemory(baseInput);

    await expect(
      adapter.createMemory({ ...baseInput, content: "impostor" }),
    ).rejects.toBeInstanceOf(AdapterError);

    // Original is intact (the impostor didn't slip through):
    const restored = await adapter.getMemory("fixed-id");
    expect(restored?.content).toBe("original");
  });
});
