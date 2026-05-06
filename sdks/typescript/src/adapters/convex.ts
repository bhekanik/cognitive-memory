/**
 * Convex Adapter for Cognitive Memory
 *
 * Implements MemoryAdapter interface for Convex database.
 */

import type { ConvexClient } from "convex/browser";
import type { FunctionReference } from "convex/server";
import type { Memory, MemoryCategory, SemanticType, ScoredMemory } from "../core/types";
import { createDefaultMemory } from "../core/types";
import { MemoryAdapter, type MemoryFilters } from "./base";

type PublicQueryRef<A extends Record<string, unknown>, R> = FunctionReference<
  "query",
  "public",
  A,
  R
>;
type PublicMutationRef<
  A extends Record<string, unknown>,
  R,
> = FunctionReference<"mutation", "public", A, R>;
type PublicActionRef<A extends Record<string, unknown>, R> = FunctionReference<
  "action",
  "public",
  A,
  R
>;

export type ConvexAdapterFunctions = {
  createCognitiveMemory: PublicMutationRef<Record<string, unknown>, string>;
  updateCognitiveMemory: PublicMutationRef<Record<string, unknown>, null>;
  deleteCognitiveMemory: PublicMutationRef<Record<string, unknown>, null>;
  deleteCognitiveMemories: PublicMutationRef<Record<string, unknown>, null>;
  getCognitiveMemory: PublicQueryRef<Record<string, unknown>, unknown>;
  getCognitiveMemories: PublicQueryRef<Record<string, unknown>, unknown>;
  queryCognitiveMemories: PublicQueryRef<Record<string, unknown>, unknown>;
  findFadingMemories: PublicQueryRef<Record<string, unknown>, unknown>;
  findStableMemories: PublicQueryRef<Record<string, unknown>, unknown>;
  markSuperseded: PublicMutationRef<Record<string, unknown>, null>;
  batchUpdateRetention: PublicMutationRef<Record<string, unknown>, null>;
  cognitiveVectorSearch: PublicActionRef<Record<string, unknown>, unknown>;

  createOrStrengthenLink: PublicMutationRef<Record<string, unknown>, null>;
  getLinkedMemories: PublicQueryRef<Record<string, unknown>, unknown>;
  getLinkedMemoriesMultiple: PublicQueryRef<Record<string, unknown>, unknown>;
  deleteLink: PublicMutationRef<Record<string, unknown>, null>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

const VALID_CATEGORIES = new Set(["episodic", "semantic", "procedural", "core"]);
function isMemoryCategory(value: unknown): value is MemoryCategory {
  return typeof value === "string" && VALID_CATEGORIES.has(value);
}

export class ConvexAdapter extends MemoryAdapter {
  private client: ConvexClient;
  private fns: ConvexAdapterFunctions;

  constructor(options: {
    client: ConvexClient;
    functions: ConvexAdapterFunctions;
  }) {
    super();
    this.client = options.client;
    this.fns = options.functions;
  }

  async transaction<T>(
    callback: (adapter: MemoryAdapter) => Promise<T>,
  ): Promise<T> {
    return callback(this);
  }

  async createMemory(
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">,
  ): Promise<string> {
    const now = Date.now();

    const category =
      isRecord(memory.metadata) && typeof memory.metadata.category === "string"
        ? memory.metadata.category
        : "user_profile";

    const importance10 = memory.importance * 10;

    const id = await this.client.mutation(this.fns.createCognitiveMemory, {
      userId: memory.userId,
      content: memory.content,
      embedding: memory.embedding,
      category: memory.category,
      stability: memory.stability,
      accessCount: memory.accessCount,
      lastAccessed: memory.lastAccessed,
      retention: memory.retention,
      metadata: {
        ...(memory.metadata ?? {}),
        category,
        importance: importance10,
        extractedAt: now,
        confidence: 1.0,
        verifiedBy: "manual",
      },
      associations: memory.associations,
      sessionIds: memory.sessionIds,
      isCold: memory.isCold,
      coldSince: memory.coldSince,
      daysAtFloor: memory.daysAtFloor,
      isSuperseded: memory.isSuperseded,
      supersededBy: memory.supersededBy,
      isStub: memory.isStub,
      contradictedBy: memory.contradictedBy,
      semanticType: memory.semanticType,
      validFrom: memory.validFrom,
      validUntil: memory.validUntil,
      ttlSeconds: memory.ttlSeconds,
      sourceTurnIds: memory.sourceTurnIds,
    });

    return id;
  }

  async getMemory(id: string): Promise<Memory | null> {
    const raw = await this.client.query(this.fns.getCognitiveMemory, { id });
    if (!raw) return null;
    return this.convexToMemory(raw);
  }

  async getMemories(ids: string[]): Promise<Memory[]> {
    const raw = await this.client.query(this.fns.getCognitiveMemories, { ids });
    if (!Array.isArray(raw)) return [];
    const out: Memory[] = [];
    for (const m of raw) {
      const memory = this.convexToMemory(m);
      if (memory) out.push(memory);
    }
    return out;
  }

  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    const raw = await this.client.query(this.fns.queryCognitiveMemories, {
      ...filters,
    });
    if (!Array.isArray(raw)) return [];
    const out: Memory[] = [];
    for (const m of raw) {
      const memory = this.convexToMemory(m);
      if (memory) out.push(memory);
    }
    return out;
  }

  async updateMemory(id: string, updates: Partial<Memory>): Promise<void> {
    const payload: Record<string, unknown> = { id };

    if (updates.userId !== undefined) payload.userId = updates.userId;
    if (updates.content !== undefined) payload.content = updates.content;
    if (updates.embedding !== undefined) payload.embedding = updates.embedding;
    if (updates.category !== undefined) payload.category = updates.category;
    if (updates.importance !== undefined)
      payload.importance = updates.importance * 10;
    if (updates.stability !== undefined) payload.stability = updates.stability;
    if (updates.accessCount !== undefined)
      payload.accessCount = updates.accessCount;
    if (updates.lastAccessed !== undefined)
      payload.lastAccessed = updates.lastAccessed;
    if (updates.retention !== undefined) payload.retention = updates.retention;
    if (updates.metadata !== undefined) payload.metadata = updates.metadata;
    if (updates.associations !== undefined)
      payload.associations = updates.associations;
    if (updates.sessionIds !== undefined) payload.sessionIds = updates.sessionIds;
    if (updates.isCold !== undefined) payload.isCold = updates.isCold;
    if (updates.coldSince !== undefined) payload.coldSince = updates.coldSince;
    if (updates.daysAtFloor !== undefined)
      payload.daysAtFloor = updates.daysAtFloor;
    if (updates.isSuperseded !== undefined)
      payload.isSuperseded = updates.isSuperseded;
    if (updates.supersededBy !== undefined)
      payload.supersededBy = updates.supersededBy;
    if (updates.isStub !== undefined) payload.isStub = updates.isStub;
    if (updates.contradictedBy !== undefined)
      payload.contradictedBy = updates.contradictedBy;
    if (updates.semanticType !== undefined)
      payload.semanticType = updates.semanticType;
    if (updates.validFrom !== undefined) payload.validFrom = updates.validFrom;
    if (updates.validUntil !== undefined) payload.validUntil = updates.validUntil;
    if (updates.ttlSeconds !== undefined) payload.ttlSeconds = updates.ttlSeconds;
    if (updates.sourceTurnIds !== undefined)
      payload.sourceTurnIds = updates.sourceTurnIds;
    if (updates.createdAt !== undefined) payload.createdAt = updates.createdAt;
    if (updates.updatedAt !== undefined) payload.updatedAt = updates.updatedAt;

    await this.client.mutation(this.fns.updateCognitiveMemory, payload);
  }

  async deleteMemory(id: string): Promise<void> {
    await this.client.mutation(this.fns.deleteCognitiveMemory, { id });
  }

  async deleteMemories(ids: string[]): Promise<void> {
    await this.client.mutation(this.fns.deleteCognitiveMemories, { ids });
  }

  async vectorSearch(
    embedding: number[],
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    const raw = await this.client.action(this.fns.cognitiveVectorSearch, {
      embedding,
      userId: filters?.userId,
      categories: filters?.categories,
      minRetention: filters?.minRetention,
      limit: filters?.limit ?? 5,
      includeSuperseded: filters?.includeSuperseded ?? false,
      includeCold: filters?.includeCold ?? false,
      includeStubs: filters?.includeStubs ?? false,
    });

    if (!Array.isArray(raw)) return [];

    const out: ScoredMemory[] = [];
    for (const item of raw) {
      if (!isRecord(item)) continue;
      const memory = this.convexToMemory(item.memory);
      if (!memory) continue;
      const relevanceScore =
        typeof item.relevanceScore === "number" ? item.relevanceScore : 0;
      out.push({
        ...memory,
        relevanceScore,
        finalScore: relevanceScore * memory.retention,
      });
    }

    return out;
  }

  async updateRetentionScores(updates: Map<string, number>): Promise<void> {
    const entries = Array.from(updates.entries()).map(([id, retention]) => ({
      id,
      retention,
    }));
    await this.client.mutation(this.fns.batchUpdateRetention, {
      updates: entries,
    });
  }

  async createOrStrengthenLink(
    sourceId: string,
    targetId: string,
    strength: number,
  ): Promise<void> {
    await this.client.mutation(this.fns.createOrStrengthenLink, {
      sourceId,
      targetId,
      strength,
    });
  }

  async getLinkedMemories(
    memoryId: string,
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    const raw = await this.client.query(this.fns.getLinkedMemories, {
      memoryId,
      minStrength,
    });
    return this.linkedResultToMemories(raw);
  }

  async getLinkedMemoriesMultiple(
    memoryIds: string[],
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    const raw = await this.client.query(this.fns.getLinkedMemoriesMultiple, {
      memoryIds,
      minStrength,
    });
    return this.linkedResultToMemories(raw);
  }

  async deleteLink(sourceId: string, targetId: string): Promise<void> {
    await this.client.mutation(this.fns.deleteLink, { sourceId, targetId });
  }

  async findFadingMemories(
    userId: string,
    maxRetention: number,
  ): Promise<Memory[]> {
    const raw = await this.client.query(this.fns.findFadingMemories, {
      userId,
      maxRetention,
    });
    if (!Array.isArray(raw)) return [];
    const out: Memory[] = [];
    for (const m of raw) {
      const memory = this.convexToMemory(m);
      if (memory) out.push(memory);
    }
    return out;
  }

  async findStableMemories(
    userId: string,
    minStability: number,
    minAccessCount: number,
  ): Promise<Memory[]> {
    const raw = await this.client.query(this.fns.findStableMemories, {
      userId,
      minStability,
      minAccessCount,
    });
    if (!Array.isArray(raw)) return [];
    const out: Memory[] = [];
    for (const m of raw) {
      const memory = this.convexToMemory(m);
      if (memory) out.push(memory);
    }
    return out;
  }

  async markSuperseded(memoryIds: string[], summaryId: string): Promise<void> {
    await this.client.mutation(this.fns.markSuperseded, {
      memoryIds,
      summaryId,
    });
  }

  // ------------------------------------------------------------------
  // Tiered storage (delegate to Convex backend)
  // ------------------------------------------------------------------

  async migrateToCold(memoryId: string, coldSince: number): Promise<void> {
    await this.updateMemory(memoryId, {
      isCold: true,
      coldSince,
    } as Partial<Memory>);
  }

  async migrateToHot(memoryId: string): Promise<void> {
    await this.updateMemory(memoryId, {
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
    } as Partial<Memory>);
  }

  async convertToStub(memoryId: string, stubContent: string): Promise<void> {
    await this.updateMemory(memoryId, {
      content: stubContent,
      isStub: true,
      isCold: false,
      coldSince: null,
      embedding: [],
    } as Partial<Memory>);
  }

  // ------------------------------------------------------------------
  // Traversal (basic implementation via queryMemories)
  // ------------------------------------------------------------------

  async allActive(): Promise<Memory[]> {
    const all = await this.queryMemories({});
    return all.filter((m) => !m.isStub);
  }

  async allHot(): Promise<Memory[]> {
    const all = await this.queryMemories({});
    return all.filter((m) => !m.isCold && !m.isStub);
  }

  async allCold(): Promise<Memory[]> {
    const all = await this.queryMemories({});
    return all.filter((m) => m.isCold && !m.isStub);
  }

  // ------------------------------------------------------------------
  // Counts
  // ------------------------------------------------------------------

  async hotCount(): Promise<number> {
    return (await this.allHot()).length;
  }

  async coldCount(): Promise<number> {
    return (await this.allCold()).length;
  }

  async stubCount(): Promise<number> {
    const all = await this.queryMemories({ includeSuperseded: true });
    return all.filter((m) => m.isStub).length;
  }

  async totalCount(): Promise<number> {
    const all = await this.queryMemories({ includeSuperseded: true });
    return all.length;
  }

  // ------------------------------------------------------------------
  // Reset
  // ------------------------------------------------------------------

  async clear(): Promise<void> {
    const all = await this.queryMemories({ includeSuperseded: true });
    await this.deleteMemories(all.map((m) => m.id));
  }

  // ------------------------------------------------------------------
  // Private helpers
  // ------------------------------------------------------------------

  private linkedResultToMemories(
    raw: unknown,
  ): Array<Memory & { linkStrength: number }> {
    if (!Array.isArray(raw)) return [];
    const out: Array<Memory & { linkStrength: number }> = [];
    for (const item of raw) {
      if (!isRecord(item)) continue;
      const memory = this.convexToMemory(item.memory);
      if (!memory) continue;
      const linkStrength =
        typeof item.strength === "number" ? item.strength : 0;
      out.push({ ...memory, linkStrength });
    }
    return out;
  }

  private convexToMemory(raw: unknown): Memory | null {
    if (!isRecord(raw)) return null;

    const id = typeof raw._id === "string" ? raw._id : null;
    const userId = typeof raw.userId === "string" ? raw.userId : null;
    const content = typeof raw.content === "string" ? raw.content : null;
    const embedding = Array.isArray(raw.embedding) ? raw.embedding : null;
    const createdAt = typeof raw.createdAt === "number" ? raw.createdAt : null;
    const updatedAt = typeof raw.updatedAt === "number" ? raw.updatedAt : null;

    if (
      !id ||
      !userId ||
      !content ||
      !embedding ||
      createdAt === null ||
      updatedAt === null
    ) {
      return null;
    }

    const metadata = isRecord(raw.metadata) ? raw.metadata : undefined;
    const importance10 =
      metadata && typeof metadata.importance === "number"
        ? metadata.importance
        : 5;

    const category = isMemoryCategory(raw.category) ? raw.category : "semantic";
    const stability = typeof raw.stability === "number" ? raw.stability : 0.3;
    const accessCount =
      typeof raw.accessCount === "number" ? raw.accessCount : 0;
    const lastAccessed =
      typeof raw.lastAccessed === "number" ? raw.lastAccessed : createdAt;
    const retention = typeof raw.retention === "number" ? raw.retention : 1.0;

    return createDefaultMemory({
      id,
      userId,
      content,
      embedding: embedding.reduce<number[]>((acc, n) => {
        if (typeof n === "number") acc.push(n);
        return acc;
      }, []),
      category,
      importance: importance10 / 10,
      stability,
      accessCount,
      lastAccessed,
      retention,
      createdAt,
      updatedAt,
      metadata: metadata ?? undefined,
      isCold: typeof raw.isCold === "boolean" ? raw.isCold : false,
      coldSince: typeof raw.coldSince === "number" ? raw.coldSince : null,
      daysAtFloor: typeof raw.daysAtFloor === "number" ? raw.daysAtFloor : 0,
      isSuperseded: typeof raw.isSuperseded === "boolean" ? raw.isSuperseded : false,
      supersededBy: typeof raw.supersededBy === "string" ? raw.supersededBy : null,
      isStub: typeof raw.isStub === "boolean" ? raw.isStub : false,
      contradictedBy: typeof raw.contradictedBy === "string" ? raw.contradictedBy : null,
      semanticType: (typeof raw.semanticType === "string" ? raw.semanticType : "other") as SemanticType,
      validFrom: typeof raw.validFrom === "number" ? raw.validFrom : null,
      validUntil: typeof raw.validUntil === "number" ? raw.validUntil : null,
      ttlSeconds: typeof raw.ttlSeconds === "number" ? raw.ttlSeconds : null,
      sourceTurnIds: Array.isArray(raw.sourceTurnIds) ? raw.sourceTurnIds.filter((s: unknown) => typeof s === "string") : [],
      sessionIds: Array.isArray(raw.sessionIds) ? raw.sessionIds.filter((s: unknown) => typeof s === "string") : [],
      associations: isRecord(raw.associations) ? raw.associations as Memory["associations"] : {},
    });
  }
}
