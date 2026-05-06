/**
 * batchUpdate(memories) — spec/adapter-interface.md lines 277-287.
 *
 * Update multiple memories in one operation. All-or-nothing where the
 * backend supports it; the in-memory adapter is trivially atomic.
 */

import { InMemoryAdapter } from "../src/adapters/memory";
import type { Memory } from "../src/core/types";
import { createDefaultMemory } from "../src/core/types";

describe("batchUpdate", () => {
  test("applies updates to every supplied memory", async () => {
    const adapter = new InMemoryAdapter();

    const ids: string[] = [];
    for (let i = 0; i < 3; i++) {
      const id = await adapter.createMemory({
        ...createDefaultMemory({
          id: "tmp",
          userId: "u1",
          content: `m${i}`,
          embedding: [1, 0],
        }),
        stability: 0.3,
        importance: 0.5,
      });
      ids.push(id);
    }

    const updated: Memory[] = [];
    for (const id of ids) {
      const m = (await adapter.getMemory(id))!;
      updated.push({ ...m, stability: 0.9, accessCount: m.accessCount + 1 });
    }

    await adapter.batchUpdate(updated);

    for (const id of ids) {
      const m = await adapter.getMemory(id);
      expect(m?.stability).toBe(0.9);
      expect(m?.accessCount).toBe(1);
    }
  });

  test("preserves memories not in the batch", async () => {
    const adapter = new InMemoryAdapter();
    const idA = await adapter.createMemory({
      ...createDefaultMemory({ id: "tmp", userId: "u1", content: "A", embedding: [1, 0] }),
      stability: 0.2,
    });
    const idB = await adapter.createMemory({
      ...createDefaultMemory({ id: "tmp", userId: "u1", content: "B", embedding: [1, 0] }),
      stability: 0.2,
    });

    const a = (await adapter.getMemory(idA))!;
    await adapter.batchUpdate([{ ...a, stability: 0.8 }]);

    const updatedA = await adapter.getMemory(idA);
    const untouchedB = await adapter.getMemory(idB);
    expect(updatedA?.stability).toBe(0.8);
    expect(untouchedB?.stability).toBe(0.2);
  });
});
