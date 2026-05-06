/**
 * In-Memory Adapter for Cognitive Memory
 *
 * Three-tier dict-based storage (hot/cold/stub) with brute-force cosine similarity.
 * Zero dependencies, great for testing and single-process use.
 */

import { randomUUID } from "node:crypto";
import type { Memory, ScoredMemory } from "../core/types";
import { createDefaultMemory } from "../core/types";
import { cosineSimilarity } from "../utils/embeddings";
import { MemoryAdapter, type MemoryFilters } from "./base";
import { DuplicateMemoryError, MemoryNotFoundError } from "./errors";

export class InMemoryAdapter extends MemoryAdapter {
  /** Hot store — active memories */
  hot = new Map<string, Memory>();
  /** Cold store — inactive but retained */
  cold = new Map<string, Memory>();
  /** Stub store — archived summaries */
  stubs = new Map<string, Memory>();

  private links = new Map<
    string,
    { strength: number; createdAt: number; updatedAt: number }
  >();
  private now: () => number;
  private idFactory: () => string;

  constructor(options?: { now?: () => number; idFactory?: () => string }) {
    super();
    this.now = options?.now ?? Date.now;
    this.idFactory = options?.idFactory ?? randomUUID;
  }

  // ------------------------------------------------------------------
  // CRUD
  // ------------------------------------------------------------------

  async transaction<T>(
    callback: (adapter: MemoryAdapter) => Promise<T>,
  ): Promise<T> {
    return callback(this);
  }

  async createMemory(
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">,
  ): Promise<string> {
    const id = this.idFactory();
    if (this.hot.has(id) || this.cold.has(id) || this.stubs.has(id)) {
      throw new DuplicateMemoryError(id);
    }
    const now = this.now();
    const m = createDefaultMemory({
      ...memory,
      id,
      createdAt: now,
      updatedAt: now,
    });
    this.hot.set(id, m);
    return id;
  }

  async getMemory(id: string): Promise<Memory | null> {
    return this.hot.get(id) ?? this.cold.get(id) ?? this.stubs.get(id) ?? null;
  }

  async getMemories(ids: string[]): Promise<Memory[]> {
    return ids
      .map((id) => this.hot.get(id) ?? this.cold.get(id) ?? this.stubs.get(id))
      .filter((m): m is Memory => m !== undefined);
  }

  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    let items = Array.from(this.hot.values());

    if (filters.includeCold) {
      items = [...items, ...Array.from(this.cold.values())];
    }
    if (filters.includeStubs) {
      items = [...items, ...Array.from(this.stubs.values())];
    }

    // Exclude superseded by default
    if (!filters.includeSuperseded) {
      items = items.filter((m) => !m.isSuperseded);
    }

    if (!filters.includeStubs) items = items.filter((m) => !m.isStub);

    if (filters.userId)
      items = items.filter((m) => m.userId === filters.userId);
    if (filters.categories)
      items = items.filter((m) => filters.categories!.includes(m.category));
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
    const tier = this.getTier(id);
    if (!tier) {
      throw new MemoryNotFoundError(id);
    }
    const existing = tier.get(id)!;
    tier.set(id, {
      ...existing,
      ...updates,
      id,
      createdAt: existing.createdAt,
    });
  }

  async deleteMemory(id: string): Promise<void> {
    this.hot.delete(id);
    this.cold.delete(id);
    this.stubs.delete(id);
  }

  async deleteMemories(ids: string[]): Promise<void> {
    for (const id of ids) {
      this.hot.delete(id);
      this.cold.delete(id);
      this.stubs.delete(id);
    }
  }

  // ------------------------------------------------------------------
  // Vector search
  // ------------------------------------------------------------------

  async vectorSearch(
    embedding: number[],
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    let items = Array.from(this.hot.values());

    if (filters?.includeCold) {
      items = [...items, ...Array.from(this.cold.values())];
    }
    if (filters?.includeStubs) {
      items = [...items, ...Array.from(this.stubs.values())];
    }

    // Apply filters
    if (filters?.userId)
      items = items.filter((m) => m.userId === filters.userId);
    if (filters?.categories)
      items = items.filter((m) => filters.categories!.includes(m.category));
    if (filters?.minRetention !== undefined)
      items = items.filter((m) => m.retention >= filters.minRetention!);
    if (!filters?.includeSuperseded)
      items = items.filter((m) => !m.isSuperseded);
    if (!filters?.includeStubs) items = items.filter((m) => !m.isStub);

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

  // ------------------------------------------------------------------
  // Lexical search (BM25)
  // ------------------------------------------------------------------

  override async searchLexical(
    query: string,
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    const queryTokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (queryTokens.length === 0) return [];

    let items = Array.from(this.hot.values());
    if (filters?.includeCold)
      items = [...items, ...Array.from(this.cold.values())];
    if (filters?.includeStubs)
      items = [...items, ...Array.from(this.stubs.values())];
    if (!filters?.includeSuperseded)
      items = items.filter((m) => !m.isSuperseded);
    if (!filters?.includeStubs) items = items.filter((m) => !m.isStub);
    if (filters?.userId)
      items = items.filter((m) => m.userId === filters.userId);

    const N = items.length;
    if (N === 0) return [];

    // Build doc tokens
    const docs = items.map((m) => ({
      mem: m,
      tokens: m.content.toLowerCase().split(/\s+/).filter(Boolean),
    }));

    const avgdl = docs.reduce((sum, d) => sum + d.tokens.length, 0) / N;
    const k1 = 1.2;
    const b = 0.75;

    // IDF per query token
    const df = new Map<string, number>();
    for (const t of queryTokens) {
      let count = 0;
      for (const d of docs) {
        if (d.tokens.includes(t)) count++;
      }
      df.set(t, count);
    }

    const idf = new Map<string, number>();
    for (const t of queryTokens) {
      const n = df.get(t) ?? 0;
      idf.set(t, Math.log((N - n + 0.5) / (n + 0.5) + 1.0));
    }

    // Score
    const results: ScoredMemory[] = [];
    for (const { mem, tokens } of docs) {
      const dl = tokens.length;
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

      let score = 0;
      for (const t of queryTokens) {
        const f = tf.get(t) ?? 0;
        if (f === 0) continue;
        const num = f * (k1 + 1);
        const denom = f + k1 * (1 - b + b * dl / avgdl);
        score += (idf.get(t) ?? 0) * num / denom;
      }

      if (score > 0) {
        results.push({
          ...mem,
          relevanceScore: score,
          finalScore: score,
        });
      }
    }

    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return results.slice(0, filters?.limit ?? 30);
  }

  // ------------------------------------------------------------------
  // Retention
  // ------------------------------------------------------------------

  async updateRetentionScores(updates: Map<string, number>): Promise<void> {
    for (const [id, retention] of updates.entries()) {
      const tier = this.getTier(id);
      if (!tier) continue;
      const m = tier.get(id)!;
      tier.set(id, { ...m, retention });
    }
  }

  // ------------------------------------------------------------------
  // Links
  // ------------------------------------------------------------------

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
        const m = await this.getMemory(other);
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

  // ------------------------------------------------------------------
  // Consolidation helpers
  // ------------------------------------------------------------------

  async findFadingMemories(
    userId: string,
    maxRetention: number,
  ): Promise<Memory[]> {
    return [
      ...Array.from(this.hot.values()),
      ...Array.from(this.cold.values()),
    ].filter(
      (m) =>
        m.userId === userId &&
        m.retention < maxRetention &&
        !m.isSuperseded &&
        !m.isStub,
    );
  }

  async findStableMemories(
    userId: string,
    minStability: number,
    minAccessCount: number,
  ): Promise<Memory[]> {
    return Array.from(this.hot.values()).filter(
      (m) =>
        m.userId === userId &&
        m.stability >= minStability &&
        m.accessCount >= minAccessCount,
    );
  }

  async markSuperseded(memoryIds: string[], summaryId: string): Promise<void> {
    for (const id of memoryIds) {
      const tier = this.getTier(id);
      if (!tier) continue;
      const m = tier.get(id)!;
      tier.set(id, {
        ...m,
        isSuperseded: true,
        supersededBy: summaryId,
        metadata: { ...(m.metadata ?? {}), supersededBy: summaryId },
      });
    }
  }

  // ------------------------------------------------------------------
  // Tiered storage
  // ------------------------------------------------------------------

  async migrateToCold(memoryId: string, coldSince: number): Promise<void> {
    const m = this.hot.get(memoryId);
    if (!m) return;
    this.hot.delete(memoryId);
    this.cold.set(memoryId, {
      ...m,
      isCold: true,
      coldSince,
    });
  }

  async migrateToHot(memoryId: string): Promise<void> {
    const m = this.cold.get(memoryId);
    if (!m) return;
    this.cold.delete(memoryId);
    this.hot.set(memoryId, {
      ...m,
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
    });
  }

  async convertToStub(memoryId: string, stubContent: string): Promise<void> {
    const m = this.cold.get(memoryId) ?? this.hot.get(memoryId);
    if (!m) return;
    this.hot.delete(memoryId);
    this.cold.delete(memoryId);
    this.stubs.set(memoryId, {
      ...m,
      content: stubContent,
      isStub: true,
      isCold: false,
      coldSince: null,
      embedding: [],
    });
  }

  // ------------------------------------------------------------------
  // Traversal
  // ------------------------------------------------------------------

  async allActive(): Promise<Memory[]> {
    return [
      ...Array.from(this.hot.values()),
      ...Array.from(this.cold.values()),
    ];
  }

  async allHot(): Promise<Memory[]> {
    return Array.from(this.hot.values());
  }

  async allCold(): Promise<Memory[]> {
    return Array.from(this.cold.values());
  }

  // ------------------------------------------------------------------
  // Counts
  // ------------------------------------------------------------------

  async hotCount(): Promise<number> {
    return this.hot.size;
  }

  async coldCount(): Promise<number> {
    return this.cold.size;
  }

  async stubCount(): Promise<number> {
    return this.stubs.size;
  }

  async totalCount(): Promise<number> {
    return this.hot.size + this.cold.size + this.stubs.size;
  }

  // ------------------------------------------------------------------
  // Reset
  // ------------------------------------------------------------------

  async clear(): Promise<void> {
    this.hot.clear();
    this.cold.clear();
    this.stubs.clear();
    this.links.clear();
  }

  // ------------------------------------------------------------------
  // Private helpers
  // ------------------------------------------------------------------

  private linkKey(a: string, b: string): string {
    return a < b ? `${a}|${b}` : `${b}|${a}`;
  }

  private getTier(id: string): Map<string, Memory> | null {
    if (this.hot.has(id)) return this.hot;
    if (this.cold.has(id)) return this.cold;
    if (this.stubs.has(id)) return this.stubs;
    return null;
  }
}
