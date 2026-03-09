/**
 * Cognitive Memory System - Engine
 *
 * Implements all mechanisms from the paper:
 * - Decay model with floors (Section 3.2, 3.3)
 * - Two-tier retrieval boosting (Section 3.5)
 * - Associative memory graph (Section 3.6)
 * - Consolidation (Section 3.7)
 * - Tiered storage with cold TTL (Section 3.8)
 * - Core memory promotion (Section 3.4)
 */

import type { MemoryAdapter } from "../adapters/base";
import { cosineSimilarity } from "../utils/embeddings";
import { calculateRetention, updateStability } from "./decay";
import type { LLMProvider } from "./extraction";
import { rerankCandidates } from "./extraction";
import type {
  Association,
  EmbeddingProvider,
  Memory,
  MemoryCategory,
  ResolvedCognitiveMemoryConfig,
  SearchResponse,
  SearchResult,
  SearchTrace,
  SemanticType,
  StageTrace,
} from "./types";
import { getRetentionFloor } from "./types";

const MS_PER_DAY = 1000 * 60 * 60 * 24;
const EXPIRABLE_TYPES: Set<string> = new Set(["plan", "transient_state"]);

/**
 * The computational core. Operates on a MemoryAdapter and applies
 * all the temporal dynamics described in the paper.
 */
export class CognitiveEngine {
  constructor(
    private adapter: MemoryAdapter,
    private config: ResolvedCognitiveMemoryConfig,
  ) {}

  // ------------------------------------------------------------------
  // Decay model - Equation 1
  // ------------------------------------------------------------------

  computeRetention(memory: Memory, now: number = Date.now()): number {
    if (memory.isStub) return 0.0;

    const category = memory.category;
    const rates = this.config.decayRates;
    const baseDecay = rates[category];

    if (category === "procedural" || baseDecay === Number.POSITIVE_INFINITY) {
      return 1.0;
    }

    const floor = getRetentionFloor(category, this.config);

    const lastAccessed = memory.lastAccessed || memory.createdAt;
    const daysSinceAccess = Math.max(0, (now - lastAccessed) / MS_PER_DAY);

    const S = Math.max(memory.stability, 0.01);
    const B = Math.min(3.0, 1.0 + memory.importance * 2.0);
    const effectiveRate = S * B * baseDecay;

    const raw = this.config.decayModel === "power"
      ? (1 + daysSinceAccess / effectiveRate) ** (-this.config.powerDecayGamma)
      : Math.exp(-daysSinceAccess / effectiveRate);
    return Math.max(floor, Math.min(1, raw));
  }

  // ------------------------------------------------------------------
  // Retrieval scoring - Equation 3
  // ------------------------------------------------------------------

  scoreMemory(memory: Memory, relevance: number, now: number = Date.now()): number {
    const retention = this.computeRetention(memory, now);
    const alpha = this.config.retrievalScoreExponent;
    return relevance * retention ** alpha;
  }

  // ------------------------------------------------------------------
  // Retrieval boosting - Section 3.5
  // ------------------------------------------------------------------

  private spacedRepFactor(memory: Memory, now: number): number {
    const lastAccessed = memory.lastAccessed || memory.createdAt;
    const dtDays = Math.max(0, (now - lastAccessed) / MS_PER_DAY);
    return Math.min(
      this.config.maxSpacedRepMultiplier,
      dtDays / this.config.spacedRepIntervalDays,
    );
  }

  applyDirectBoost(memory: Memory, now: number, sessionId?: string): void {
    const factor = this.spacedRepFactor(memory, now);
    memory.stability = Math.min(
      1.0,
      memory.stability + this.config.directBoost * factor,
    );
    memory.accessCount += 1;
    memory.lastAccessed = now;
    if (sessionId && !memory.sessionIds.includes(sessionId)) {
      memory.sessionIds.push(sessionId);
    }
  }

  applyAssociativeBoost(memory: Memory, now: number, sessionId?: string): void {
    const factor = this.spacedRepFactor(memory, now);
    memory.stability = Math.min(
      1.0,
      memory.stability + this.config.associativeBoost * factor,
    );
    memory.accessCount += 1;
    memory.lastAccessed = now;
    if (sessionId && !memory.sessionIds.includes(sessionId)) {
      memory.sessionIds.push(sessionId);
    }
  }

  // ------------------------------------------------------------------
  // Core memory promotion - Section 3.4
  // ------------------------------------------------------------------

  checkCorePromotion(memory: Memory): boolean {
    if (memory.category === "core") return false;

    if (
      memory.accessCount >= this.config.coreAccessThreshold &&
      memory.stability >= this.config.coreStabilityThreshold &&
      memory.sessionIds.length >= this.config.coreSessionThreshold
    ) {
      memory.category = "core";
      memory.memoryType = "semantic"; // core maps to semantic for backward compat
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Associative graph - Section 3.6
  // ------------------------------------------------------------------

  strengthenAssociation(
    memA: Memory,
    memB: Memory,
    now: number,
  ): void {
    const amount = this.config.associationStrengthenAmount;

    // A -> B
    if (!memA.associations[memB.id]) {
      memA.associations[memB.id] = {
        targetId: memB.id,
        weight: 0,
        lastCoRetrieval: null,
        createdAt: now,
      };
    }
    const assocAB = memA.associations[memB.id];
    assocAB.weight = Math.min(1.0, assocAB.weight + amount);
    assocAB.lastCoRetrieval = now;

    // B -> A
    if (!memB.associations[memA.id]) {
      memB.associations[memA.id] = {
        targetId: memA.id,
        weight: 0,
        lastCoRetrieval: null,
        createdAt: now,
      };
    }
    const assocBA = memB.associations[memA.id];
    assocBA.weight = Math.min(1.0, assocBA.weight + amount);
    assocBA.lastCoRetrieval = now;
  }

  decayAssociation(assoc: Association, now: number): number {
    if (assoc.lastCoRetrieval === null) return assoc.weight;
    const dtDays = Math.max(0, (now - assoc.lastCoRetrieval) / MS_PER_DAY);
    const tau = this.config.associationDecayConstantDays;
    const decayed = assoc.weight * Math.exp(-dtDays / tau);
    assoc.weight = decayed;
    return decayed;
  }

  getAssociatedMemories(
    memory: Memory,
    now: number,
  ): Array<{ memory: Memory; weight: number }> {
    const results: Array<{ memory: Memory; weight: number }> = [];
    const threshold = this.config.associationRetrievalThreshold;

    for (const assoc of Object.values(memory.associations)) {
      const weight = this.decayAssociation(assoc, now);
      if (weight < threshold) continue;

      // Sync get from adapter (InMemoryAdapter has hot/cold/stubs Maps)
      const target = this.syncGet(assoc.targetId);
      if (!target || target.isStub) continue;

      results.push({ memory: target, weight });
    }

    return results;
  }

  private syncGet(memoryId: string): Memory | null {
    const adapter = this.adapter as any;
    // InMemoryAdapter exposes hot/cold/stubs Maps
    if (adapter.hot) {
      return adapter.hot.get(memoryId) ??
        adapter.cold?.get(memoryId) ??
        adapter.stubs?.get(memoryId) ??
        null;
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Graph expansion / bridge discovery (v6)
  // ------------------------------------------------------------------

  private expandGraph(
    anchors: Memory[],
    now: number,
    seenIds: Set<string>,
    maxHops: number,
  ): Array<{ memory: Memory; weight: number }> {
    let frontier = anchors.slice();
    const results: Array<{ memory: Memory; weight: number }> = [];

    for (let hop = 0; hop < maxHops; hop++) {
      const nextFrontier: Memory[] = [];
      for (const mem of frontier) {
        for (const assoc of Object.values(mem.associations)) {
          const weight = this.decayAssociation(assoc, now);
          if (weight < this.config.minBridgeEdgeWeight) continue;
          if (seenIds.has(assoc.targetId)) continue;

          const target = this.syncGet(assoc.targetId);
          if (!target || target.isStub) continue;

          seenIds.add(target.id);
          results.push({ memory: target, weight });
          nextFrontier.push(target);
        }
      }
      frontier = nextFrontier;
    }

    return results;
  }

  private findBridgePaths(anchors: Memory[], now: number): string[][] {
    if (anchors.length < 2) return [];

    const chains: string[][] = [];
    const anchorIds = anchors.slice(0, 3).map((m) => m.id);

    for (let i = 0; i < anchorIds.length; i++) {
      for (let j = i + 1; j < anchorIds.length; j++) {
        const path = this.bfsPath(anchorIds[i], anchorIds[j], now, 3);
        if (path && path.length > 2) {
          chains.push(path);
          if (chains.length >= this.config.maxBridgePaths) return chains;
        }
      }
    }
    return chains;
  }

  private bfsPath(
    startId: string,
    endId: string,
    now: number,
    maxDepth: number,
  ): string[] | null {
    const queue: string[][] = [[startId]];
    const visited = new Set([startId]);

    while (queue.length > 0) {
      const path = queue.shift()!;
      if (path.length > maxDepth + 1) break;

      const currentId = path[path.length - 1];
      const current = this.syncGet(currentId);
      if (!current) continue;

      for (const assoc of Object.values(current.associations)) {
        const weight = this.decayAssociation(assoc, now);
        if (weight < this.config.minBridgeEdgeWeight) continue;
        if (visited.has(assoc.targetId)) continue;

        const newPath = [...path, assoc.targetId];
        if (assoc.targetId === endId) return newPath;

        visited.add(assoc.targetId);
        queue.push(newPath);
      }
    }

    return null;
  }

  // ------------------------------------------------------------------
  // Validity filtering (v6)
  // ------------------------------------------------------------------

  private isExpired(memory: Memory, now: number): boolean {
    if (!memory.semanticType || !EXPIRABLE_TYPES.has(memory.semanticType)) {
      return false;
    }
    if (memory.validUntil != null && now > memory.validUntil) {
      return true;
    }
    if (memory.ttlSeconds != null && memory.createdAt) {
      const expiry = memory.createdAt + memory.ttlSeconds * 1000;
      if (now > expiry) return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Full retrieval pipeline
  // ------------------------------------------------------------------

  async search(
    queryEmbedding: number[],
    now: number,
    topK: number = 10,
    sessionId?: string,
    deepRecall: boolean = false,
    queryText?: string,
    trace: boolean = false,
    llm?: LLMProvider,
  ): Promise<SearchResponse> {
    const alpha = this.config.retrievalScoreExponent;
    const searchTrace: SearchTrace | undefined = trace
      ? { totalWallMs: 0, totalTokens: 0, stages: {} }
      : undefined;
    const tTotal = trace ? performance.now() : 0;

    // Step 1: Similarity search in hot store
    let t0 = trace ? performance.now() : 0;
    const candidates = await this.adapter.vectorSearch(queryEmbedding, {
      limit: topK * 3,
      includeSuperseded: deepRecall,
    });
    if (trace && searchTrace) {
      searchTrace.stages.vector_search = {
        name: "vector_search",
        wallMs: performance.now() - t0,
        candidateCount: candidates.length,
      };
    }

    // Step 1b: Lexical search (if hybrid enabled)
    if (this.config.hybridSearch && queryText) {
      const lexicalResults = await this.adapter.searchLexical(queryText, {
        limit: this.config.kSparse,
        includeSuperseded: deepRecall,
      });
      const seenIds = new Set(candidates.map((c) => (c as Memory).id));
      for (const lexMem of lexicalResults) {
        if (!seenIds.has(lexMem.id)) {
          // Compute dense similarity for lexical-only candidates
          const sim = lexMem.embedding.length > 0
            ? cosineSimilarity(queryEmbedding, lexMem.embedding)
            : 0.1;
          candidates.push({
            ...lexMem,
            relevanceScore: sim,
            finalScore: sim,
          });
          seenIds.add(lexMem.id);
        }
      }
    }

    // Step 2: Score candidates
    t0 = trace ? performance.now() : 0;
    let scored: SearchResult[] = [];
    for (const candidate of candidates) {
      const mem = candidate as Memory;
      const relevance = candidate.relevanceScore;
      const retention = this.computeRetention(mem, now);
      let combined = relevance * retention ** alpha;

      // Deep recall penalty for superseded memories
      if (mem.isSuperseded && deepRecall) {
        combined *= this.config.deepRecallPenalty;
      }

      scored.push({
        memory: mem,
        relevanceScore: relevance,
        retentionScore: retention,
        combinedScore: combined,
        isAssociative: false,
        viaDeepRecall: mem.isSuperseded && deepRecall,
      });
    }

    // Step 2b: Validity filtering
    if (this.config.filterExpiredTransients) {
      const filtered: SearchResult[] = [];
      for (const r of scored) {
        if (this.isExpired(r.memory, now)) {
          if (deepRecall && this.config.includeExpiredInDeepRecall) {
            r.combinedScore *= this.config.deepRecallPenalty;
            r.viaDeepRecall = true;
            filtered.push(r);
          }
          // else: exclude
        } else {
          filtered.push(r);
        }
      }
      scored = filtered;
    }

    if (trace && searchTrace) {
      searchTrace.stages.scoring = {
        name: "scoring",
        wallMs: performance.now() - t0,
        candidateCount: scored.length,
      };
    }

    // Sort by combined score
    scored.sort((a, b) => b.combinedScore - a.combinedScore);

    // Step 2c: LLM rerank (if enabled and LLM provided)
    if (this.config.rerankEnabled && llm && queryText && scored.length > 1) {
      t0 = trace ? performance.now() : 0;
      const kRerank = Math.min(this.config.kRerank, scored.length);
      const toRerank = scored.slice(0, kRerank);

      const { rerankedIndices, usage } = await rerankCandidates(
        queryText,
        toRerank.map((r) => ({ content: r.memory.content })),
        llm,
        this.config,
      );

      // Rebuild scored array: reranked items first (in LLM order), then remainder
      const reranked: SearchResult[] = [];
      const usedIndices = new Set<number>();
      for (const idx of rerankedIndices) {
        if (idx < toRerank.length) {
          reranked.push(toRerank[idx]);
          usedIndices.add(idx);
        }
      }
      // Append any that LLM omitted (considered irrelevant but keep them at lower priority)
      for (let i = 0; i < toRerank.length; i++) {
        if (!usedIndices.has(i)) reranked.push(toRerank[i]);
      }
      // Append remainder beyond kRerank
      scored = [...reranked, ...scored.slice(kRerank)];

      if (trace && searchTrace) {
        const promptTokens = usage.promptTokens ?? 0;
        const completionTokens = usage.completionTokens ?? 0;
        searchTrace.stages.rerank = {
          name: "rerank",
          wallMs: performance.now() - t0,
          candidateCount: kRerank,
          promptTokens,
          completionTokens,
        };
        searchTrace.totalTokens += promptTokens + completionTokens;
      }
    }

    // Take top-k direct results
    const directResults = scored.slice(0, topK);

    // Step 3: Collect associated memories
    const seenIds = new Set(directResults.map((r) => r.memory.id));
    const associativeResults: SearchResult[] = [];

    for (const result of directResults) {
      const associated = this.getAssociatedMemories(result.memory, now);
      for (const { memory: assocMem, weight: assocWeight } of associated) {
        if (seenIds.has(assocMem.id)) continue;
        seenIds.add(assocMem.id);

        const relevance =
          assocMem.embedding.length > 0
            ? cosineSimilarity(queryEmbedding, assocMem.embedding)
            : 0.1;
        const retention = this.computeRetention(assocMem, now);
        const combined = relevance * retention ** alpha * assocWeight;

        associativeResults.push({
          memory: assocMem,
          relevanceScore: relevance,
          retentionScore: retention,
          combinedScore: combined,
          isAssociative: true,
          viaDeepRecall: false,
        });
      }
    }

    // Step 3b: Graph expansion (if configured)
    if (this.config.graphExpansionHops > 0) {
      const expanded = this.expandGraph(
        directResults.map((r) => r.memory),
        now, seenIds, this.config.graphExpansionHops,
      );
      for (const { memory: expMem, weight: expWeight } of expanded) {
        const relevance = expMem.embedding.length > 0
          ? cosineSimilarity(queryEmbedding, expMem.embedding)
          : 0.1;
        const retention = this.computeRetention(expMem, now);
        const combined = relevance * retention ** alpha * expWeight;

        associativeResults.push({
          memory: expMem,
          relevanceScore: relevance,
          retentionScore: retention,
          combinedScore: combined,
          isAssociative: true,
          viaDeepRecall: false,
        });
      }
    }

    // Bridge discovery
    let evidenceChains: string[][] = [];
    if (this.config.bridgeDiscovery) {
      evidenceChains = this.findBridgePaths(
        directResults.map((r) => r.memory), now,
      );
    }

    // Step 4: Apply boosts
    for (const result of directResults) {
      this.applyDirectBoost(result.memory, now, sessionId);
      if (result.memory.isCold) {
        await this.adapter.migrateToHot(result.memory.id);
      }
    }

    for (const result of associativeResults) {
      this.applyAssociativeBoost(result.memory, now, sessionId);
      if (result.memory.isCold) {
        await this.adapter.migrateToHot(result.memory.id);
      }
    }

    // Step 5: Check core promotions
    for (const result of [...directResults, ...associativeResults]) {
      this.checkCorePromotion(result.memory);
    }

    // Step 6: Strengthen associations between co-retrieved memories
    const directMems = directResults.map((r) => r.memory);
    for (let i = 0; i < directMems.length; i++) {
      for (let j = i + 1; j < directMems.length; j++) {
        this.strengthenAssociation(directMems[i], directMems[j], now);
      }
    }

    // Combine and sort
    const allResults = [...directResults, ...associativeResults];
    allResults.sort((a, b) => b.combinedScore - a.combinedScore);

    const final = allResults.slice(0, topK);
    if (evidenceChains.length > 0 && final.length > 0) {
      final[0].evidenceChains = evidenceChains;
    }

    if (trace && searchTrace) {
      searchTrace.totalWallMs = performance.now() - tTotal;
    }

    return {
      results: final,
      evidenceChains,
      trace: searchTrace,
    };
  }

  // ------------------------------------------------------------------
  // Cold storage management - Section 3.8
  // ------------------------------------------------------------------

  async runColdMigration(now: number): Promise<void> {
    const thresholdDays = this.config.coldMigrationDays;

    for (const mem of await this.adapter.allHot()) {
      if (mem.category === "core") continue;

      if (mem.isSuperseded) {
        await this.adapter.migrateToCold(mem.id, now);
        continue;
      }

      const retention = this.computeRetention(mem, now);
      const floor = getRetentionFloor(mem.category, this.config);
      const atFloor = Math.abs(retention - floor) < 0.001;

      if (atFloor) {
        mem.daysAtFloor += 1;
      } else {
        mem.daysAtFloor = 0;
      }

      if (mem.daysAtFloor >= thresholdDays) {
        await this.adapter.migrateToCold(mem.id, now);
      }
    }
  }

  async runColdTtlExpiry(now: number): Promise<void> {
    const ttlDays = this.config.coldStorageTtlDays;

    for (const mem of await this.adapter.allCold()) {
      if (mem.coldSince === null) continue;
      if (mem.category === "core") continue;

      const daysCold = (now - mem.coldSince) / MS_PER_DAY;
      if (daysCold >= ttlDays) {
        const stubContent = `[archived] ${mem.content.slice(0, 200)}`;
        await this.adapter.convertToStub(mem.id, stubContent);
      }
    }
  }

  // ------------------------------------------------------------------
  // Consolidation - Section 3.7
  // ------------------------------------------------------------------

  async runConsolidation(
    now: number,
    embedder: EmbeddingProvider,
    llmCompress?: (contents: string[]) => string | Promise<string>,
  ): Promise<void> {
    const threshold = this.config.consolidationRetentionThreshold;
    const groupSize = this.config.consolidationGroupSize;
    const simThreshold = this.config.consolidationSimilarityThreshold;

    // Find fading non-core, non-superseded memories in hot store
    const fading: Memory[] = [];
    for (const mem of await this.adapter.allHot()) {
      if (mem.isSuperseded || mem.category === "core") continue;
      const retention = this.computeRetention(mem, now);
      if (retention < threshold) {
        fading.push(mem);
      }
    }

    if (fading.length < groupSize) return;

    // Group by category
    const byCategory = new Map<MemoryCategory, Memory[]>();
    for (const mem of fading) {
      const group = byCategory.get(mem.category) ?? [];
      group.push(mem);
      byCategory.set(mem.category, group);
    }

    for (const [category, mems] of byCategory) {
      if (mems.length < groupSize) continue;

      // Simple greedy clustering by embedding similarity
      const used = new Set<string>();
      const groups: Memory[][] = [];

      for (let i = 0; i < mems.length; i++) {
        if (used.has(mems[i].id)) continue;
        const group = [mems[i]];

        for (let j = i + 1; j < mems.length; j++) {
          if (used.has(mems[j].id)) continue;
          if (mems[i].embedding.length > 0 && mems[j].embedding.length > 0) {
            const sim = cosineSimilarity(mems[i].embedding, mems[j].embedding);
            if (sim >= simThreshold) {
              group.push(mems[j]);
              if (group.length >= groupSize) break;
            }
          }
        }

        if (group.length >= groupSize) {
          const finalGroup = group.slice(0, groupSize);
          groups.push(finalGroup);
          for (const m of finalGroup) used.add(m.id);
        }
      }

      // Create summaries for each group
      for (const group of groups) {
        const contents = group.map((m) => m.content);

        let summaryText: string;
        if (llmCompress) {
          summaryText = await llmCompress(contents);
        } else {
          summaryText = "Summary: " + contents.join(" | ");
        }

        // Create summary embedding
        const summaryEmbedding = await embedder.embed(summaryText);

        // Create summary memory
        const summaryId = await this.adapter.createMemory({
          userId: this.config.userId,
          content: summaryText,
          embedding: summaryEmbedding,
          category,
          memoryType: category === "core" ? "semantic" : (category as any),
          importance: Math.max(...group.map((m) => m.importance)),
          stability:
            group.reduce((sum, m) => sum + m.stability, 0) / group.length,
          accessCount: Math.max(...group.map((m) => m.accessCount)),
          lastAccessed: now,
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
        });

        // Supersede originals and move to cold
        await this.adapter.markSuperseded(
          group.map((m) => m.id),
          summaryId,
        );

        for (const m of group) {
          m.isSuperseded = true;
          m.supersededBy = summaryId;
          await this.adapter.migrateToCold(m.id, now);
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // Maintenance tick
  // ------------------------------------------------------------------

  async tick(
    now: number,
    embedder: EmbeddingProvider,
    llmCompress?: (contents: string[]) => string | Promise<string>,
  ): Promise<void> {
    await this.runColdMigration(now);
    await this.runColdTtlExpiry(now);
    await this.runConsolidation(now, embedder, llmCompress);
  }
}
