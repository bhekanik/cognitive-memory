/**
 * Cognitive Memory System - Base Adapter Interface
 *
 * Abstract adapter that concrete implementations (Convex, Postgres, etc.) must implement.
 * Provides database-agnostic interface for memory storage and retrieval.
 */

import type { Memory, MemoryCategory, ScoredMemory } from "../core/types";

/**
 * Filters for querying memories
 */
export interface MemoryFilters {
  /** Filter by user ID */
  userId?: string;

  /** Filter by memory categories */
  categories?: MemoryCategory[];

  /** Minimum retention threshold */
  minRetention?: number;

  /** Minimum importance */
  minImportance?: number;

  /** Created after timestamp */
  createdAfter?: number;

  /** Created before timestamp */
  createdBefore?: number;

  /** Limit number of results */
  limit?: number;

  /** Offset for pagination */
  offset?: number;

  /** Include superseded memories in results */
  includeSuperseded?: boolean;

  /** Include cold memories in retrieval/query results */
  includeCold?: boolean;

  /** Include archived stubs in results */
  includeStubs?: boolean;
}

/**
 * Abstract adapter interface for memory persistence
 *
 * Implementations must handle:
 * - CRUD operations on memories
 * - Vector search for semantic retrieval
 * - Link management for associative memory
 * - Tiered storage (hot/cold/stub)
 * - Batch operations for consolidation
 */
export abstract class MemoryAdapter {
  // ------------------------------------------------------------------
  // CRUD
  // ------------------------------------------------------------------

  abstract createMemory(
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">,
  ): Promise<string>;

  abstract getMemory(id: string): Promise<Memory | null>;

  async getMemories(ids: string[]): Promise<Memory[]> {
    const memories = await Promise.all(ids.map((id) => this.getMemory(id)));
    return memories.filter((memory): memory is Memory => memory !== null);
  }

  abstract queryMemories(filters: MemoryFilters): Promise<Memory[]>;

  abstract updateMemory(id: string, updates: Partial<Memory>): Promise<void>;

  abstract deleteMemory(id: string): Promise<void>;

  abstract deleteMemories(ids: string[]): Promise<void>;

  // ------------------------------------------------------------------
  // Vector search
  // ------------------------------------------------------------------

  abstract vectorSearch(
    embedding: number[],
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]>;

  // ------------------------------------------------------------------
  // Lexical search (optional, for hybrid retrieval)
  // ------------------------------------------------------------------

  /**
   * BM25/lexical search. Override in adapters that support it.
   * Default: returns empty array (dense-only fallback).
   */
  async searchLexical(
    _query: string,
    _filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    return [];
  }

  // ------------------------------------------------------------------
  // Batch update
  // ------------------------------------------------------------------

  /**
   * Update multiple memories in a single operation. Atomic where the
   * backend supports it. Default implementation falls back to N
   * serial `updateMemory` calls; override for true atomicity.
   *
   * Spec: spec/adapter-interface.md lines 277-287.
   */
  async batchUpdate(memories: Memory[]): Promise<void> {
    for (const m of memories) {
      const { id, createdAt: _createdAt, ...updates } = m;
      await this.updateMemory(id, updates);
    }
  }

  // ------------------------------------------------------------------
  // Retention
  // ------------------------------------------------------------------

  abstract updateRetentionScores(updates: Map<string, number>): Promise<void>;

  // ------------------------------------------------------------------
  // Links
  // ------------------------------------------------------------------

  abstract createOrStrengthenLink(
    sourceId: string,
    targetId: string,
    strength: number,
  ): Promise<void>;

  abstract getLinkedMemories(
    memoryId: string,
    minStrength?: number,
  ): Promise<Array<Memory & { linkStrength: number }>>;

  abstract getLinkedMemoriesMultiple(
    memoryIds: string[],
    minStrength?: number,
  ): Promise<Array<Memory & { linkStrength: number }>>;

  abstract deleteLink(sourceId: string, targetId: string): Promise<void>;

  // ------------------------------------------------------------------
  // Consolidation helpers
  // ------------------------------------------------------------------

  abstract findFadingMemories(
    userId: string,
    maxRetention: number,
  ): Promise<Memory[]>;

  abstract findStableMemories(
    userId: string,
    minStability: number,
    minAccessCount: number,
  ): Promise<Memory[]>;

  abstract markSuperseded(
    memoryIds: string[],
    summaryId: string,
  ): Promise<void>;

  // ------------------------------------------------------------------
  // Tiered storage
  // ------------------------------------------------------------------

  abstract migrateToCold(memoryId: string, coldSince: number): Promise<void>;

  abstract migrateToHot(memoryId: string): Promise<void>;

  abstract convertToStub(memoryId: string, stubContent: string): Promise<void>;

  // ------------------------------------------------------------------
  // Traversal
  // ------------------------------------------------------------------

  abstract allActive(): Promise<Memory[]>;

  abstract allHot(): Promise<Memory[]>;

  abstract allCold(): Promise<Memory[]>;

  // ------------------------------------------------------------------
  // Counts
  // ------------------------------------------------------------------

  abstract hotCount(): Promise<number>;

  abstract coldCount(): Promise<number>;

  abstract stubCount(): Promise<number>;

  abstract totalCount(): Promise<number>;

  // ------------------------------------------------------------------
  // Reset
  // ------------------------------------------------------------------

  abstract clear(): Promise<void>;

  // ------------------------------------------------------------------
  // Transaction
  // ------------------------------------------------------------------

  abstract transaction<T>(
    callback: (adapter: MemoryAdapter) => Promise<T>,
  ): Promise<T>;
}
