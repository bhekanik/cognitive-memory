/**
 * In-Memory Adapter for Cognitive Memory
 *
 * Dict-based tiered storage with brute-force cosine similarity.
 * Zero dependencies, great for testing and single-process use.
 */

import { randomUUID } from "node:crypto";
import type { Memory, ScoredMemory } from "../core/types";
import { cosineSimilarity } from "../utils/embeddings";
import { MemoryAdapter, type MemoryFilters } from "./base";

export class InMemoryAdapter extends MemoryAdapter {
  private memories = new Map<string, Memory>();
  private links = new Map<string, { strength: number; createdAt: number; updatedAt: number }>();
  private now: () => number;
  private idFactory: () => string;

  constructor(options?: { now?: () => number; idFactory?: () => string }) {
    super();
    this.now = options?.now ?? Date.now;
    this.idFactory = options?.idFactory ?? randomUUID;
  }

  async transaction<T>(
    callback: (adapter: MemoryAdapter) => Promise<T>,
  ): Promise<T> {
    return callback(this);
  }

  async createMemory(
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">,
  ): Promise<string> {
    const id = this.idFactory();
    const now = this.now();
    const m: Memory = { ...memory, id, createdAt: now, updatedAt: now };
    this.memories.set(id, m);
    return id;
  }

  async getMemory(id: string): Promise<Memory | null> {
    return this.memories.get(id) ?? null;
  }

  async getMemories(ids: string[]): Promise<Memory[]> {
    return ids
      .map((id) => this.memories.get(id))
      .filter((m): m is Memory => m !== undefined);
  }

  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    let items = Array.from(this.memories.values());

    if (filters.userId)
      items = items.filter((m) => m.userId === filters.userId);
    if (filters.memoryTypes)
      items = items.filter((m) => filters.memoryTypes!.includes(m.memoryType));
    if (filters.minRetention !== undefined)
      items = items.filter((m) => m.retention >= filters.minRetention!);
    if (filters.minImportance !== undefined)
      items = items.filter((m) => m.importance >= filters.minImportance!);
    if (filters.createdAfter !== undefined)
      items = items.filter((m) => m.createdAt >= filters.createdAfter!);
    if (filters.createdBefore !== undefined)
      items = items.filter((m) => m.createdAt <= filters.createdBefore!);
    if (filters.offset) items = items.slice(filters.offset);
    if (filters.limit) items = items.slice(0, filters.limit);
    return items;
  }

  async updateMemory(id: string, updates: Partial<Memory>): Promise<void> {
    const existing = this.memories.get(id);
    if (!existing) return;
    this.memories.set(id, {
      ...existing,
      ...updates,
      id,
      createdAt: existing.createdAt,
    });
  }

  async deleteMemory(id: string): Promise<void> {
    this.memories.delete(id);
  }

  async deleteMemories(ids: string[]): Promise<void> {
    for (const id of ids) this.memories.delete(id);
  }

  async vectorSearch(
    embedding: number[],
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    const items = await this.queryMemories(filters ?? {});
    return items
      .map((m) => {
        const relevanceScore = cosineSimilarity(embedding, m.embedding);
        return {
          ...m,
          relevanceScore,
          finalScore: relevanceScore * m.retention,
        };
      })
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, filters?.limit ?? 5);
  }

  async updateRetentionScores(updates: Map<string, number>): Promise<void> {
    for (const [id, retention] of updates.entries()) {
      const m = this.memories.get(id);
      if (m) this.memories.set(id, { ...m, retention });
    }
  }

  async createOrStrengthenLink(
    sourceId: string,
    targetId: string,
    strength: number,
  ): Promise<void> {
    const key = this.linkKey(sourceId, targetId);
    const existing = this.links.get(key);
    const now = this.now();
    this.links.set(key, {
      strength: Math.min(1, (existing?.strength ?? 0) + strength),
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    });
  }

  async getLinkedMemories(
    memoryId: string,
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    return this.getLinkedMemoriesMultiple([memoryId], minStrength);
  }

  async getLinkedMemoriesMultiple(
    memoryIds: string[],
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    const out = new Map<string, Memory & { linkStrength: number }>();
    for (const id of memoryIds) {
      for (const [key, row] of this.links.entries()) {
        if (row.strength < minStrength) continue;
        const [a, b] = key.split("|");
        const other = a === id ? b : b === id ? a : null;
        if (!other) continue;
        const m = this.memories.get(other);
        if (!m) continue;
        const prev = out.get(other);
        out.set(
          other,
          prev
            ? { ...m, linkStrength: Math.max(prev.linkStrength, row.strength) }
            : { ...m, linkStrength: row.strength },
        );
      }
    }
    return Array.from(out.values());
  }

  async deleteLink(sourceId: string, targetId: string): Promise<void> {
    this.links.delete(this.linkKey(sourceId, targetId));
  }

  async findFadingMemories(
    userId: string,
    maxRetention: number,
  ): Promise<Memory[]> {
    return Array.from(this.memories.values()).filter(
      (m) => m.userId === userId && m.retention < maxRetention,
    );
  }

  async findStableMemories(
    userId: string,
    minStability: number,
    minAccessCount: number,
  ): Promise<Memory[]> {
    return Array.from(this.memories.values()).filter(
      (m) =>
        m.userId === userId &&
        m.stability >= minStability &&
        m.accessCount >= minAccessCount,
    );
  }

  async markSuperseded(memoryIds: string[], summaryId: string): Promise<void> {
    for (const id of memoryIds) {
      const m = this.memories.get(id);
      if (!m) continue;
      this.memories.set(id, {
        ...m,
        metadata: { ...(m.metadata ?? {}), supersededBy: summaryId },
      });
    }
  }

  private linkKey(a: string, b: string): string {
    return a < b ? `${a}|${b}` : `${b}|${a}`;
  }
}
