/**
 * Cognitive Memory System - Main Class
 *
 * High-level API for cognitive memory with decay, retrieval strengthening,
 * associative linking, tiered storage, and LLM extraction.
 */

import type { MemoryAdapter, MemoryFilters } from "../adapters/base";
import { cosineSimilarity } from "../utils/embeddings";
import { greedyClusterBySimilarity } from "./clustering";
import { updateStability } from "./decay";
import { CognitiveEngine } from "./engine";
import {
  extractFromConversation,
  extractRawTurns,
  detectConflict,
  compressMemories,
} from "./extraction";
import type { LLMProvider } from "./extraction";
import type {
  CognitiveMemoryConfig,
  ConsolidationResult,
  EmbeddingProvider,
  Memory,
  MemoryCategory,
  MemoryInput,
  MemoryStats,
  ResolvedCognitiveMemoryConfig,
  RetrievalQuery,
  ScoredMemory,
  SearchResponse,
  SearchResult,
} from "./types";
import { resolveConfig, getRetentionFloor } from "./types";
import { assertNonEmptyString, assertUnitInterval, assertNonNegativeInt } from "./validation";
import { setTimeout as sleep } from "node:timers/promises";

const SYNAPTIC_TAGGING_THRESHOLD = 0.4;
const STABILITY_REINFORCEMENT_THRESHOLD = 0.75;
const INGESTION_MAINTENANCE_INTERVAL = 5;

/**
 * Main cognitive memory system
 *
 * Provides high-level API for storing, retrieving, and managing memories
 * with human-like characteristics: decay, retrieval strengthening,
 * associative linking, tiered storage, and LLM extraction.
 */
export class CognitiveMemory {
  private adapter: MemoryAdapter;
  private embeddingProvider: EmbeddingProvider;
  private config: ResolvedCognitiveMemoryConfig;
  private engine: CognitiveEngine;
  private conflictQueue: Array<{ newId: string; existingId: string; similarity: number }> = [];
  private ingestionTickCounter = 0;

  constructor(options: {
    adapter: MemoryAdapter;
    embeddingProvider: EmbeddingProvider;
    userId: string;
    config?: Partial<CognitiveMemoryConfig>;
  }) {
    assertNonEmptyString("userId", options.userId);

    this.adapter = options.adapter;
    this.embeddingProvider = options.embeddingProvider;
    this.config = resolveConfig({
      userId: options.userId,
      ...options.config,
    });
    this.engine = new CognitiveEngine(this.adapter, this.config);
  }

  /**
   * Store a new memory
   *
   * Generates embedding and initializes cognitive metadata.
   */
  async store(input: MemoryInput): Promise<string> {
    assertNonEmptyString("content", input.content);
    if (input.importance !== undefined)
      assertUnitInterval("importance", input.importance);
    if (input.stability !== undefined)
      assertUnitInterval("stability", input.stability);

    const embedding = await this.embedWithRetry(input.content);
    const category = input.category ?? "semantic";

    const now = Date.now();
    const memory: Omit<Memory, "id" | "createdAt" | "updatedAt"> = {
      userId: this.config.userId,
      content: input.content,
      embedding,
      category,
      importance: input.importance ?? this.config.defaultImportance,
      stability: input.stability ?? this.config.defaultStability,
      accessCount: 0,
      lastAccessed: now,
      retention: 1.0,
      metadata: input.metadata,
      associations: {},
      sessionIds: [],
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
      isSuperseded: false,
      supersededBy: null,
      isStub: false,
      contradictedBy: null,
      semanticType: input.semanticType ?? "other",
      validFrom: input.validFrom ?? null,
      validUntil: input.validUntil ?? null,
      ttlSeconds: input.ttlSeconds ?? null,
      sourceTurnIds: [],
    };

    return this.adapter.createMemory(memory);
  }

  /**
   * Extract memories from conversation text using LLM, check conflicts, and store.
   *
   * This is the main ingestion method matching Python's extract_and_store.
   */
  async extractAndStore(
    conversationText: string,
    sessionId: string,
    llm: LLMProvider,
  ): Promise<string[]> {
    assertNonEmptyString("conversationText", conversationText);
    assertNonEmptyString("sessionId", sessionId);

    const mode = this.config.extractionMode;
    if (!["raw", "semantic", "hybrid"].includes(mode)) {
      throw new Error(`Invalid extractionMode: "${mode}". Must be "raw", "semantic", or "hybrid".`);
    }

    const storedIds: string[] = [];

    // --- Semantic extraction (modes: semantic, hybrid) ---
    if (mode === "semantic" || mode === "hybrid") {
      const extracted = await extractFromConversation(
        conversationText,
        llm,
        this.config,
        sessionId,
      );

      for (const mem of extracted) {
        const embedding = await this.embedWithRetry(mem.content);

        const similar = await this.adapter.vectorSearch(embedding, {
          userId: this.config.userId,
          limit: 5,
        });

        const id = await this.adapter.createMemory({
          ...mem,
          embedding,
        });
        storedIds.push(id);

        for (const existingMem of similar) {
          if (existingMem.id !== id && existingMem.relevanceScore > 0.85) {
            // Skip if same session root (e.g. dual-perspective of same conversation)
            const newRoots = new Set((mem.sessionIds ?? []).map(s => s.replace(/_perspective_.*$/, "")));
            const existingRoots = new Set((existingMem.sessionIds ?? []).map(s => s.replace(/_perspective_.*$/, "")));
            const hasSharedRoot = [...newRoots].some(r => existingRoots.has(r));
            if (hasSharedRoot) continue;

            this.conflictQueue.push({
              newId: id,
              existingId: existingMem.id,
              similarity: existingMem.relevanceScore,
            });
          }
          if (existingMem.id !== id && existingMem.relevanceScore > STABILITY_REINFORCEMENT_THRESHOLD) {
            await this.adapter.updateMemory(existingMem.id, {
              stability: Math.min(1.0, existingMem.stability + 0.05),
            });
          }
        }
      }
    }

    // --- Raw turn storage (modes: raw, hybrid) ---
    if (mode === "raw" || mode === "hybrid") {
      const rawTurns = extractRawTurns(conversationText, this.config, sessionId);

      for (const mem of rawTurns) {
        const embedding = await this.embedWithRetry(mem.content);
        const id = await this.adapter.createMemory({
          ...mem,
          embedding,
        });
        storedIds.push(id);
      }
    }

    // Synaptic tagging — create associations between memories from same session
    if (storedIds.length > 1) {
      const storedMemories = await this.adapter.getMemories(storedIds);
      for (let i = 0; i < storedMemories.length; i++) {
        for (let j = i + 1; j < storedMemories.length; j++) {
          if (
            storedMemories[i].embedding.length > 0 &&
            storedMemories[j].embedding.length > 0
          ) {
            const sim = cosineSimilarity(
              storedMemories[i].embedding,
              storedMemories[j].embedding,
            );
            if (sim >= SYNAPTIC_TAGGING_THRESHOLD) {
              // Mirror Python core.py:236-238 verbatim. Above threshold the
              // expression is algebraically equivalent to `sim * 0.5`, but the
              // explicit `min(0.5, ...)` cap and `sim - threshold` form keep
              // the SDKs behaviourally identical at the threshold boundary.
              const weight = Math.min(
                0.5,
                0.2 + (sim - SYNAPTIC_TAGGING_THRESHOLD) * 0.5,
              );
              await this.adapter.createOrStrengthenLink(
                storedMemories[i].id,
                storedMemories[j].id,
                weight,
              );
            }
          }
        }
      }
    }

    // Optionally run maintenance
    if (this.config.runMaintenanceDuringIngestion) {
      this.ingestionTickCounter += 1;
      if (this.ingestionTickCounter % INGESTION_MAINTENANCE_INTERVAL === 0) {
        await this.tick();
      }
    }

    return storedIds;
  }

  /**
   * Search using the full 6-step retrieval pipeline from the engine.
   *
   * Returns SearchResponse with results, evidence chains, and optional trace.
   */
  async search(query: RetrievalQuery, llm?: LLMProvider): Promise<SearchResponse> {
    const {
      query: queryText,
      limit = 10,
      minRetention,
      categories,
      includeAssociations = true,
      sessionId,
      deepRecall = false,
      trace = false,
    } = query;

    assertNonEmptyString("query", queryText);

    const queryEmbedding = await this.embedWithRetry(queryText);
    const now = Date.now();

    return this.engine.search(queryEmbedding, now, limit, sessionId, deepRecall, queryText, trace, llm, {
      userId: this.config.userId,
      categories,
      minRetention,
      includeAssociations,
    });
  }

  /**
   * Retrieve memories relevant to a query (backward-compatible simpler API).
   * Uses the same v6 scoring pipeline as search(), then maps results to ScoredMemory[].
   */
  async retrieve(query: RetrievalQuery): Promise<ScoredMemory[]> {
    const response = await this.search({
      ...query,
      limit: query.limit ?? 5,
      minRetention: query.minRetention ?? this.config.minRetention,
    });

    return response.results.map((result) => ({
      ...result.memory,
      retention: result.retentionScore,
      relevanceScore: result.relevanceScore,
      finalScore: result.combinedScore,
    }));
  }

  /**
   * Get a memory by ID
   */
  async get(id: string): Promise<Memory | null> {
    const memory = await this.adapter.getMemory(id);

    if (memory) {
      await this.strengthenMemories([memory]);
    }

    return memory;
  }

  /**
   * Query memories for this user
   */
  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    const memories = await this.adapter.queryMemories({
      ...filters,
      userId: this.config.userId,
    });
    if (memories.length > 0) {
      await this.strengthenMemories(memories);
    }
    return memories;
  }

  /**
   * Update a memory's content
   */
  async update(id: string, content: string): Promise<void> {
    assertNonEmptyString("content", content);

    const memory = await this.adapter.getMemory(id);
    if (!memory) {
      throw new Error(`Memory ${id} not found`);
    }

    const embedding = await this.embedWithRetry(content);

    await this.adapter.updateMemory(id, {
      content,
      embedding,
      updatedAt: Date.now(),
    });
  }

  /**
   * Run consolidation process
   */
  async consolidate(): Promise<ConsolidationResult> {
    const result: ConsolidationResult = {
      decayed: [],
      compressed: [],
      promotionCandidates: [],
      deleted: 0,
    };

    await this.refreshRetentionScores();

    // 1. Find fading memories
    const fading = await this.adapter.findFadingMemories(
      this.config.userId,
      this.config.consolidationRetentionThreshold,
    );

    result.decayed = fading.map((m) => ({
      id: m.id,
      retention: m.retention,
    }));

    // 2. Group by category, cluster by similarity
    const byCategory = new Map<MemoryCategory, Memory[]>();
    for (const memory of fading) {
      const cat: MemoryCategory = memory.category ?? "semantic";
      if (!byCategory.has(cat)) byCategory.set(cat, []);
      byCategory.get(cat)!.push(memory);
    }

    for (const [category, memories] of byCategory) {
      if (memories.length < this.config.consolidationGroupSize) continue;

      const groups = greedyClusterBySimilarity(
        memories,
        this.config.consolidationSimilarityThreshold,
        this.config.consolidationGroupSize,
      );

      for (const group of groups) {
        const summary = this.summarizeMemories(group);

        const summaryId = await this.store({
          content: summary,
          category,
          importance: Math.max(...group.map((m) => m.importance)),
          metadata: {
            compressed: true,
            sourceCount: group.length,
            category,
          },
        });

        await this.adapter.markSuperseded(
          group.map((m) => m.id),
          summaryId,
        );

        result.compressed.push({
          summaryId,
          originalIds: group.map((m) => m.id),
          count: group.length,
        });
      }
    }

    // 3. Find promotion candidates
    const stable = await this.adapter.findStableMemories(
      this.config.userId,
      this.config.coreStabilityThreshold,
      this.config.coreAccessThreshold,
    );

    result.promotionCandidates = stable.map((m) => ({
      id: m.id,
      stability: m.stability,
      accessCount: m.accessCount,
    }));

    // 4. Delete very faded memories
    const veryFaded = await this.adapter.queryMemories({
      userId: this.config.userId,
      minRetention: 0,
    });

    const toDelete = veryFaded.filter((m) => {
      const daysSinceAccess =
        (Date.now() - m.lastAccessed) / (1000 * 60 * 60 * 24);
      return m.retention < 0.05 && daysSinceAccess > 30 && !m.isSuperseded;
    });

    if (toDelete.length > 0) {
      await this.adapter.deleteMemories(toDelete.map((m) => m.id));
      result.deleted = toDelete.length;
    }

    return result;
  }

  /**
   * Run all periodic maintenance: cold migration, TTL expiry, consolidation
   */
  async tick(llm?: LLMProvider): Promise<void> {
    // Process deferred conflict queue
    if (llm) {
      await this.resolveConflictQueue(llm);
    }

    const now = Date.now();
    const compressor = llm
      ? (contents: string[]) => compressMemories(contents, llm, this.config)
      : undefined;
    await this.engine.tick(now, this.embeddingProvider, compressor);
  }

  /**
   * Process pending conflict candidates with LLM verification.
   * Called during tick() to resolve conflicts deferred from ingestion.
   */
  private async resolveConflictQueue(llm: LLMProvider, maxPerTick = 50): Promise<void> {
    if (this.conflictQueue.length === 0) return;

    // Sort by similarity descending — highest similarity = most likely conflict
    this.conflictQueue.sort((a, b) => b.similarity - a.similarity);

    const toProcess = this.conflictQueue.splice(0, maxPerTick);
    let resolved = 0;

    for (const { newId, existingId } of toProcess) {
      const [newMem, existingMem] = await Promise.all([
        this.adapter.getMemory(newId),
        this.adapter.getMemory(existingId),
      ]);

      if (!newMem || !existingMem) continue;
      if (existingMem.isSuperseded || newMem.isSuperseded) continue;

      const conflict = await detectConflict(existingMem, newMem, llm, this.config);

      if (conflict === "CONTRADICTION" || conflict === "UPDATE") {
        // Demote-and-keep: preserve the original memory as an audit trail.
        // Mirror Python's algorithm at sdks/python/src/cognitive_memory/core.py:389-405.
        const wasCore = existingMem.category === "core";
        const liftedImportance = Math.max(
          existingMem.importance,
          newMem.importance,
        );

        await this.adapter.updateMemory(existingId, {
          contradictedBy: newId,
          category: wasCore ? "semantic" : existingMem.category,
        });

        await this.adapter.updateMemory(newId, {
          importance: liftedImportance,
          category:
            conflict === "CONTRADICTION" && wasCore
              ? "core"
              : newMem.category,
        });
        resolved++;
      }
    }
  }

  /**
   * Get memory system stats
   */
  async getStats(): Promise<MemoryStats> {
    const [hotCount, coldCount, stubCount, totalCount] = await Promise.all([
      this.adapter.hotCount(),
      this.adapter.coldCount(),
      this.adapter.stubCount(),
      this.adapter.totalCount(),
    ]);

    const allActive = await this.adapter.allActive();
    let coreCount = 0;
    let faintCount = 0;
    let totalRetention = 0;

    for (const m of allActive) {
      if (m.category === "core") coreCount++;
      const retention = this.engine.computeRetention(m);
      if (retention < this.config.faintThreshold) faintCount++;
      totalRetention += retention;
    }

    return {
      total: totalCount,
      hot: hotCount,
      cold: coldCount,
      stub: stubCount,
      core: coreCount,
      faint: faintCount,
      avgRetention: allActive.length > 0 ? totalRetention / allActive.length : 0,
    };
  }

  /**
   * Clear all memories for this user
   */
  async clear(): Promise<void> {
    await this.adapter.clear();
  }

  /**
   * Recompute + persist retention for all memories for this user.
   */
  async refreshRetentionScores(): Promise<void> {
    const memories = await this.adapter.queryMemories({
      userId: this.config.userId,
      minRetention: 0,
      includeSuperseded: true,
    });

    const updates = new Map<string, number>();
    for (const m of memories) {
      const retention = this.engine.computeRetention(m);
      updates.set(m.id, retention);
    }
    if (updates.size > 0) await this.adapter.updateRetentionScores(updates);
  }

  /**
   * Create a link between two memories
   */
  async link(
    sourceId: string,
    targetId: string,
    strength: number = 0.5,
  ): Promise<void> {
    assertUnitInterval("strength", strength);
    await this.adapter.createOrStrengthenLink(sourceId, targetId, strength);
  }

  // ------------------------------------------------------------------
  // Private methods
  // ------------------------------------------------------------------

  private async strengthenMemories(memories: Memory[]): Promise<void> {
    const now = Date.now();

    const updates: Array<{ id: string; updates: Partial<Memory> }> = [];

    for (const memory of memories) {
      assertUnitInterval("stability", memory.stability);
      assertUnitInterval("importance", memory.importance);
      assertNonNegativeInt("accessCount", memory.accessCount);
      if (!Number.isFinite(memory.lastAccessed)) {
        throw new Error(
          `Invalid lastAccessed: ${memory.lastAccessed} (must be valid timestamp)`,
        );
      }

      const daysSinceAccess =
        (now - memory.lastAccessed) / (1000 * 60 * 60 * 24);

      const newStability = updateStability(
        memory.stability,
        daysSinceAccess,
        this.config.directBoost,
        this.config.maxSpacedRepMultiplier,
        this.config.spacedRepIntervalDays,
      );

      const newRetention = this.engine.computeRetention({
        ...memory,
        stability: newStability,
        accessCount: memory.accessCount + 1,
        lastAccessed: now,
      });

      updates.push({
        id: memory.id,
        updates: {
          stability: newStability,
          accessCount: memory.accessCount + 1,
          lastAccessed: now,
          retention: newRetention,
        },
      });
    }

    await this.adapter.transaction(async (adapter) => {
      await Promise.all(
        updates.map(({ id, updates: memoryUpdates }) =>
          adapter.updateMemory(id, memoryUpdates),
        ),
      );
    });
  }

  private async strengthenLinks(memoryIds: string[]): Promise<void> {
    for (let i = 0; i < memoryIds.length; i++) {
      for (let j = i + 1; j < memoryIds.length; j++) {
        await this.adapter.createOrStrengthenLink(
          memoryIds[i],
          memoryIds[j],
          0.1,
        );
      }
    }
  }

  private summarizeMemories(memories: Memory[]): string {
    const combined = memories.map((m) => m.content).join(". ");
    return combined.length > 500 ? `${combined.slice(0, 497)}...` : combined;
  }

  private async embedWithRetry(text: string): Promise<number[]> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await this.embeddingProvider.embed(text);
      } catch (err) {
        lastError = err;
        if (attempt < 2) {
          const delayMs = Math.min(2000, 250 * 2 ** attempt);
          await sleep(delayMs);
        }
      }
    }
    throw new Error(`Embedding failed: ${String(lastError)}`);
  }
}
