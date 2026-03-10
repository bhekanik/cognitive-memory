/**
 * Cognitive Memory System - Core Types
 *
 * TypeScript interfaces for human-like memory with Ebbinghaus decay,
 * spaced repetition, and associative linking.
 */

/**
 * Memory categories with different decay characteristics
 *
 * - Episodic: Events with time/place context (45-day base decay)
 * - Semantic: Facts without temporal context (120-day base decay)
 * - Procedural: Skills and how-to knowledge (no decay, updated by correction)
 * - Core: Identity-level facts with high retention floor (120-day base decay)
 */
export type MemoryCategory = "episodic" | "semantic" | "procedural" | "core";

/**
 * @deprecated Use MemoryCategory instead
 */
export type MemoryType = "episodic" | "semantic" | "procedural";

/**
 * Semantic type classification (orthogonal to MemoryCategory)
 */
export type SemanticType = "fact" | "preference" | "plan" | "transient_state" | "other";

/**
 * Bidirectional association between two memories
 */
export interface Association {
  targetId: string;
  weight: number;
  lastCoRetrieval: number | null;
  createdAt: number;
}

/**
 * Base memory interface with cognitive metadata
 */
export interface Memory {
  /** Unique identifier */
  id: string;

  /** User/agent this memory belongs to */
  userId: string;

  /** Memory content (text) */
  content: string;

  /** Vector embedding for semantic search */
  embedding: number[];

  /** Category of memory (affects decay rate and floor) */
  category: MemoryCategory;

  /**
   * @deprecated Use `category` instead. Kept for backward compatibility.
   */
  memoryType: MemoryType;

  /** Importance score (0.0-1.0, affects decay resistance) */
  importance: number;

  /** Stability (0.0-1.0, grows with retrievals) */
  stability: number;

  /** Number of times this memory has been accessed */
  accessCount: number;

  /** Timestamp of last access */
  lastAccessed: number;

  /** Current retention score (0.0-1.0, cached for performance) */
  retention: number;

  /** When this memory was created */
  createdAt: number;

  /** When this memory was last updated */
  updatedAt: number;

  /** Optional metadata */
  metadata?: Record<string, unknown>;

  /** Associations to other memories (keyed by target memory ID) */
  associations: Record<string, Association>;

  /** Session IDs that accessed this memory (for core promotion) */
  sessionIds: string[];

  /** Whether this memory is in cold storage */
  isCold: boolean;

  /** Timestamp when moved to cold storage */
  coldSince: number | null;

  /** Consecutive days at retention floor */
  daysAtFloor: number;

  /** Whether this memory was superseded by a consolidation summary */
  isSuperseded: boolean;

  /** ID of the summary memory that superseded this one */
  supersededBy: string | null;

  /** Whether this memory is an archived stub */
  isStub: boolean;

  /** ID of memory that contradicts this one */
  contradictedBy: string | null;

  /** Semantic type classification (orthogonal to category) */
  semanticType?: SemanticType;

  /** Timestamp when this memory becomes valid */
  validFrom?: number | null;

  /** Timestamp when this memory expires */
  validUntil?: number | null;

  /** Time-to-live in seconds (alternative to validUntil) */
  ttlSeconds?: number | null;

  /** Turn IDs this memory was extracted from (for retrieval diagnostics) */
  sourceTurnIds?: string[];
}

/**
 * Input for storing a new memory
 */
export interface MemoryInput {
  /** Memory content */
  content: string;

  /** Category of memory */
  category?: MemoryCategory;

  /**
   * @deprecated Use `category` instead
   */
  memoryType?: MemoryType;

  /** Importance (0.0-1.0), auto-scored if not provided */
  importance?: number;

  /** Initial stability (default: 0.3) */
  stability?: number;

  /** Optional metadata */
  metadata?: Record<string, unknown>;

  /** Semantic type classification */
  semanticType?: SemanticType;

  /** Timestamp when this memory becomes valid */
  validFrom?: number | null;

  /** Timestamp when this memory expires */
  validUntil?: number | null;

  /** Time-to-live in seconds */
  ttlSeconds?: number | null;
}

/**
 * Query for retrieving memories
 */
export interface RetrievalQuery {
  /** Search query text */
  query: string;

  /** Maximum number of results */
  limit?: number;

  /** Minimum retention threshold (0.0-1.0) */
  minRetention?: number;

  /** Filter by memory categories */
  categories?: MemoryCategory[];

  /**
   * @deprecated Use `categories` instead
   */
  memoryTypes?: MemoryType[];

  /** Include associatively linked memories */
  includeAssociations?: boolean;

  /** Session ID for tracking */
  sessionId?: string;

  /** Enable deep recall (include superseded memories) */
  deepRecall?: boolean;

  /** Include per-stage instrumentation trace in response */
  trace?: boolean;
}

/**
 * A scored memory from retrieval
 */
export interface SearchResult {
  memory: Memory;
  relevanceScore: number;
  retentionScore: number;
  combinedScore: number;
  isAssociative: boolean;
  viaDeepRecall: boolean;
  evidenceChains?: string[][];
}

/**
 * Timing and stats for a single pipeline stage
 */
export interface StageTrace {
  name: string;
  wallMs: number;
  candidateCount: number;
  promptTokens?: number;
  completionTokens?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Per-query instrumentation trace
 */
export interface SearchTrace {
  totalWallMs: number;
  totalTokens: number;
  stages: Record<string, StageTrace>;
}

/**
 * Full search response with results, evidence chains, and optional trace
 */
export interface SearchResponse {
  results: SearchResult[];
  evidenceChains: string[][];
  trace?: SearchTrace;
}

/**
 * Memory with retrieval score (backward compat)
 */
export interface ScoredMemory extends Memory {
  /** Semantic similarity score */
  relevanceScore: number;

  /** Final score (relevance x retention) */
  finalScore: number;
}

/**
 * Link between two memories
 */
export interface MemoryLink {
  /** Source memory ID */
  sourceId: string;

  /** Target memory ID */
  targetId: string;

  /** Link strength (0.0-1.0) */
  strength: number;

  /** When this link was created */
  createdAt: number;

  /** When this link was last strengthened */
  updatedAt: number;
}

/**
 * Result of consolidation process
 */
export interface ConsolidationResult {
  /** Memories that decayed significantly */
  decayed: Array<{ id: string; retention: number }>;

  /** Compressed memory groups */
  compressed: Array<{
    summaryId: string;
    originalIds: string[];
    count: number;
  }>;

  /** Memories eligible for promotion to long-term storage */
  promotionCandidates: Array<{
    id: string;
    stability: number;
    accessCount: number;
  }>;

  /** Number of memories soft-deleted */
  deleted: number;
}

/**
 * Decay calculation parameters
 */
export interface DecayParameters {
  /** Memory stability (0.0-1.0) */
  stability: number;

  /** Importance score (0.0-1.0) */
  importance: number;

  /** Number of times accessed (frequency signal) */
  accessCount?: number;

  /** Timestamp of last access */
  lastAccessed: number;

  /** Memory category */
  category: MemoryCategory;

  /**
   * @deprecated Use `category` instead
   */
  memoryType?: MemoryType;
}

/**
 * Memory system stats
 */
export interface MemoryStats {
  total: number;
  hot: number;
  cold: number;
  stub: number;
  core: number;
  faint: number;
  avgRetention: number;
}

/**
 * Configuration for cognitive memory system
 */
export interface CognitiveMemoryConfig {
  /** User ID this memory system belongs to */
  userId: string;

  // -- Decay --

  /** Decay model: "exponential" (default) or "power" (power-law) */
  decayModel?: "exponential" | "power";

  /** Gamma exponent for power-law decay (default: 1/ln(2) ~= 1.4427) */
  powerDecayGamma?: number;

  /** Default importance for new memories (0.0-1.0) */
  defaultImportance?: number;

  /** Default stability for new memories (0.0-1.0) */
  defaultStability?: number;

  /** Minimum retention for retrieval */
  minRetention?: number;

  /** Threshold below which memories are considered "faint" */
  faintThreshold?: number;

  /** Base decay days by memory category */
  decayRates?: Partial<Record<MemoryCategory, number>>;

  // -- Retention floors --

  /** Retention floor for core memories (default: 0.60) */
  coreRetentionFloor?: number;

  /** Retention floor for regular memories (default: 0.02) */
  regularRetentionFloor?: number;

  // -- Retrieval scoring --

  /** Alpha exponent in R^alpha scoring (default: 0.3) */
  retrievalScoreExponent?: number;

  // -- Retrieval boosting --

  /** Direct retrieval stability boost (default: 0.1) */
  directBoost?: number;

  /** Associative retrieval stability boost (default: 0.03) */
  associativeBoost?: number;

  /** Max spaced repetition multiplier (default: 2.0) */
  maxSpacedRepMultiplier?: number;

  /** Spaced repetition interval in days (default: 7.0) */
  spacedRepIntervalDays?: number;

  // -- Core promotion --

  /** Access count threshold for core promotion (default: 10) */
  coreAccessThreshold?: number;

  /** Stability threshold for core promotion (default: 0.85) */
  coreStabilityThreshold?: number;

  /** Session count threshold for core promotion (default: 3) */
  coreSessionThreshold?: number;

  // -- Associations --

  /** Amount to strengthen associations on co-retrieval (default: 0.1) */
  associationStrengthenAmount?: number;

  /** Minimum association weight for retrieval (default: 0.3) */
  associationRetrievalThreshold?: number;

  /** Association decay time constant in days (default: 90) */
  associationDecayConstantDays?: number;

  // -- Consolidation --

  /** Retention threshold below which memories are candidates (default: 0.20) */
  consolidationRetentionThreshold?: number;

  /** Minimum group size for consolidation (default: 5) */
  consolidationGroupSize?: number;

  /** Cosine similarity threshold for grouping (default: 0.70) */
  consolidationSimilarityThreshold?: number;

  // -- Tiered storage --

  /** Days at floor before cold migration (default: 7) */
  coldMigrationDays?: number;

  /** Days in cold storage before TTL expiry (default: 180) */
  coldStorageTtlDays?: number;

  // -- Deep recall --

  /** Score penalty multiplier for superseded memories (default: 0.5) */
  deepRecallPenalty?: number;

  // -- Hybrid retrieval (v6) --

  /** Enable hybrid dense + BM25 lexical search (default: false) */
  hybridSearch?: boolean;

  /** Top-k for BM25 lexical search (default: 30) */
  kSparse?: number;

  // -- Validity filtering (v6) --

  /** Filter expired plan/transient_state memories from results (default: true) */
  filterExpiredTransients?: boolean;

  /** Include expired memories when deep_recall is enabled (default: true) */
  includeExpiredInDeepRecall?: boolean;

  // -- LLM rerank (v6) --

  /** Enable LLM reranking of top candidates (default: false) */
  rerankEnabled?: boolean;

  /** Number of top candidates to send to LLM for reranking (default: 10) */
  kRerank?: number;

  /** Model for reranking (defaults to extractionModel) */
  rerankModel?: string;

  // -- Graph expansion / bridge discovery (v6) --

  /** Number of hops for graph expansion (0=disabled, 1 or 2) */
  graphExpansionHops?: number;

  /** Enable bridge path discovery between top results */
  bridgeDiscovery?: boolean;

  /** Maximum number of bridge paths to return */
  maxBridgePaths?: number;

  /** Minimum edge weight for graph traversal */
  minBridgeEdgeWeight?: number;

  // -- Ingestion --

  /** Run maintenance (tick) during ingestion (default: true) */
  runMaintenanceDuringIngestion?: boolean;

  // -- Models --

  /** LLM model for extraction (default: "gpt-4o-mini") */
  extractionModel?: string;

  /** Embedding model (default: "text-embedding-3-small") */
  embeddingModel?: string;

  /** Embedding dimensions (default: 1536) */
  embeddingDimensions?: number;

  /** Custom instructions prepended to extraction prompt */
  customExtractionInstructions?: string | null;

  /** Extraction mode: "raw" stores turns verbatim, "semantic" uses LLM extraction, "hybrid" does both */
  extractionMode?: "raw" | "semantic" | "hybrid";
}

/**
 * Resolved config with all defaults applied
 */
export interface ResolvedCognitiveMemoryConfig {
  userId: string;

  decayModel: "exponential" | "power";
  powerDecayGamma: number;

  defaultImportance: number;
  defaultStability: number;
  minRetention: number;
  faintThreshold: number;
  decayRates: Record<MemoryCategory, number>;

  coreRetentionFloor: number;
  regularRetentionFloor: number;

  retrievalScoreExponent: number;

  directBoost: number;
  associativeBoost: number;
  maxSpacedRepMultiplier: number;
  spacedRepIntervalDays: number;

  coreAccessThreshold: number;
  coreStabilityThreshold: number;
  coreSessionThreshold: number;

  associationStrengthenAmount: number;
  associationRetrievalThreshold: number;
  associationDecayConstantDays: number;

  consolidationRetentionThreshold: number;
  consolidationGroupSize: number;
  consolidationSimilarityThreshold: number;

  coldMigrationDays: number;
  coldStorageTtlDays: number;

  deepRecallPenalty: number;

  hybridSearch: boolean;
  kSparse: number;

  filterExpiredTransients: boolean;
  includeExpiredInDeepRecall: boolean;

  rerankEnabled: boolean;
  kRerank: number;
  rerankModel: string | null;

  graphExpansionHops: number;
  bridgeDiscovery: boolean;
  maxBridgePaths: number;
  minBridgeEdgeWeight: number;

  runMaintenanceDuringIngestion: boolean;

  extractionModel: string;
  embeddingModel: string;
  embeddingDimensions: number;
  customExtractionInstructions: string | null;
  extractionMode: "raw" | "semantic" | "hybrid";
}

/**
 * Default config values matching Python SDK
 */
export const DEFAULT_CONFIG: Omit<ResolvedCognitiveMemoryConfig, "userId"> = {
  decayModel: "exponential",
  powerDecayGamma: 1.4427,

  defaultImportance: 0.5,
  defaultStability: 0.3,
  minRetention: 0.2,
  faintThreshold: 0.15,
  decayRates: {
    episodic: 45,
    semantic: 120,
    procedural: Number.POSITIVE_INFINITY,
    core: 120,
  },

  coreRetentionFloor: 0.60,
  regularRetentionFloor: 0.02,

  retrievalScoreExponent: 0.3,

  directBoost: 0.1,
  associativeBoost: 0.03,
  maxSpacedRepMultiplier: 2.0,
  spacedRepIntervalDays: 7.0,

  coreAccessThreshold: 10,
  coreStabilityThreshold: 0.85,
  coreSessionThreshold: 3,

  associationStrengthenAmount: 0.1,
  associationRetrievalThreshold: 0.3,
  associationDecayConstantDays: 90,

  consolidationRetentionThreshold: 0.20,
  consolidationGroupSize: 5,
  consolidationSimilarityThreshold: 0.70,

  coldMigrationDays: 7,
  coldStorageTtlDays: 180,

  deepRecallPenalty: 0.5,

  hybridSearch: false,
  kSparse: 30,

  filterExpiredTransients: true,
  includeExpiredInDeepRecall: true,

  rerankEnabled: false,
  kRerank: 10,
  rerankModel: null,

  graphExpansionHops: 1,
  bridgeDiscovery: false,
  maxBridgePaths: 3,
  minBridgeEdgeWeight: 0.3,

  runMaintenanceDuringIngestion: true,

  extractionModel: "gpt-4o-mini",
  embeddingModel: "text-embedding-3-small",
  embeddingDimensions: 1536,
  customExtractionInstructions: null,
  extractionMode: "semantic",
};

/**
 * Resolve partial config into full config with defaults
 */
export function resolveConfig(
  config: CognitiveMemoryConfig,
): ResolvedCognitiveMemoryConfig {
  return {
    userId: config.userId,
    decayModel: config.decayModel ?? DEFAULT_CONFIG.decayModel,
    powerDecayGamma: config.powerDecayGamma ?? DEFAULT_CONFIG.powerDecayGamma,
    defaultImportance: config.defaultImportance ?? DEFAULT_CONFIG.defaultImportance,
    defaultStability: config.defaultStability ?? DEFAULT_CONFIG.defaultStability,
    minRetention: config.minRetention ?? DEFAULT_CONFIG.minRetention,
    faintThreshold: config.faintThreshold ?? DEFAULT_CONFIG.faintThreshold,
    decayRates: {
      ...DEFAULT_CONFIG.decayRates,
      ...config.decayRates,
    },
    coreRetentionFloor: config.coreRetentionFloor ?? DEFAULT_CONFIG.coreRetentionFloor,
    regularRetentionFloor: config.regularRetentionFloor ?? DEFAULT_CONFIG.regularRetentionFloor,
    retrievalScoreExponent: config.retrievalScoreExponent ?? DEFAULT_CONFIG.retrievalScoreExponent,
    directBoost: config.directBoost ?? DEFAULT_CONFIG.directBoost,
    associativeBoost: config.associativeBoost ?? DEFAULT_CONFIG.associativeBoost,
    maxSpacedRepMultiplier: config.maxSpacedRepMultiplier ?? DEFAULT_CONFIG.maxSpacedRepMultiplier,
    spacedRepIntervalDays: config.spacedRepIntervalDays ?? DEFAULT_CONFIG.spacedRepIntervalDays,
    coreAccessThreshold: config.coreAccessThreshold ?? DEFAULT_CONFIG.coreAccessThreshold,
    coreStabilityThreshold: config.coreStabilityThreshold ?? DEFAULT_CONFIG.coreStabilityThreshold,
    coreSessionThreshold: config.coreSessionThreshold ?? DEFAULT_CONFIG.coreSessionThreshold,
    associationStrengthenAmount: config.associationStrengthenAmount ?? DEFAULT_CONFIG.associationStrengthenAmount,
    associationRetrievalThreshold: config.associationRetrievalThreshold ?? DEFAULT_CONFIG.associationRetrievalThreshold,
    associationDecayConstantDays: config.associationDecayConstantDays ?? DEFAULT_CONFIG.associationDecayConstantDays,
    consolidationRetentionThreshold: config.consolidationRetentionThreshold ?? DEFAULT_CONFIG.consolidationRetentionThreshold,
    consolidationGroupSize: config.consolidationGroupSize ?? DEFAULT_CONFIG.consolidationGroupSize,
    consolidationSimilarityThreshold: config.consolidationSimilarityThreshold ?? DEFAULT_CONFIG.consolidationSimilarityThreshold,
    coldMigrationDays: config.coldMigrationDays ?? DEFAULT_CONFIG.coldMigrationDays,
    coldStorageTtlDays: config.coldStorageTtlDays ?? DEFAULT_CONFIG.coldStorageTtlDays,
    deepRecallPenalty: config.deepRecallPenalty ?? DEFAULT_CONFIG.deepRecallPenalty,
    hybridSearch: config.hybridSearch ?? DEFAULT_CONFIG.hybridSearch,
    kSparse: config.kSparse ?? DEFAULT_CONFIG.kSparse,
    filterExpiredTransients: config.filterExpiredTransients ?? DEFAULT_CONFIG.filterExpiredTransients,
    includeExpiredInDeepRecall: config.includeExpiredInDeepRecall ?? DEFAULT_CONFIG.includeExpiredInDeepRecall,
    rerankEnabled: config.rerankEnabled ?? DEFAULT_CONFIG.rerankEnabled,
    kRerank: config.kRerank ?? DEFAULT_CONFIG.kRerank,
    rerankModel: config.rerankModel ?? DEFAULT_CONFIG.rerankModel,
    graphExpansionHops: config.graphExpansionHops ?? DEFAULT_CONFIG.graphExpansionHops,
    bridgeDiscovery: config.bridgeDiscovery ?? DEFAULT_CONFIG.bridgeDiscovery,
    maxBridgePaths: config.maxBridgePaths ?? DEFAULT_CONFIG.maxBridgePaths,
    minBridgeEdgeWeight: config.minBridgeEdgeWeight ?? DEFAULT_CONFIG.minBridgeEdgeWeight,
    runMaintenanceDuringIngestion: config.runMaintenanceDuringIngestion ?? DEFAULT_CONFIG.runMaintenanceDuringIngestion,
    extractionModel: config.extractionModel ?? DEFAULT_CONFIG.extractionModel,
    embeddingModel: config.embeddingModel ?? DEFAULT_CONFIG.embeddingModel,
    embeddingDimensions: config.embeddingDimensions ?? DEFAULT_CONFIG.embeddingDimensions,
    customExtractionInstructions: config.customExtractionInstructions ?? DEFAULT_CONFIG.customExtractionInstructions,
    extractionMode: config.extractionMode ?? DEFAULT_CONFIG.extractionMode,
  };
}

/**
 * Embedding provider interface
 */
export interface EmbeddingProvider {
  /** Generate embedding vector for text */
  embed(text: string): Promise<number[]>;

  /** Generate embeddings for multiple texts */
  embedBatch?(texts: string[]): Promise<number[][]>;

  /** Embedding dimensions */
  readonly dimensions?: number;
}

/**
 * Map a MemoryCategory to the deprecated MemoryType field.
 * "core" has no MemoryType equivalent, so it maps to "semantic".
 */
export function categoryToMemoryType(category: MemoryCategory): MemoryType {
  return category === "core" ? "semantic" : category;
}

/**
 * Create a default Memory object with all fields initialized
 */
export function createDefaultMemory(
  partial: Partial<Memory> & { id: string; userId: string; content: string; embedding: number[] },
): Memory {
  const now = Date.now();
  const category = partial.category ?? "semantic";
  return {
    category,
    memoryType: categoryToMemoryType(category),
    importance: 0.5,
    stability: 0.3,
    accessCount: 0,
    lastAccessed: now,
    retention: 1.0,
    createdAt: now,
    updatedAt: now,
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: "other",
    validFrom: null,
    validUntil: null,
    ttlSeconds: null,
    sourceTurnIds: [],
    ...partial,
  };
}

/**
 * Get the retention floor for a memory category
 */
export function getRetentionFloor(
  category: MemoryCategory,
  config: { coreRetentionFloor: number; regularRetentionFloor: number },
): number {
  return category === "core" ? config.coreRetentionFloor : config.regularRetentionFloor;
}

/**
 * Get the base decay rate for a memory category
 */
export function getBaseDecayRate(
  category: MemoryCategory,
  rates: Record<MemoryCategory, number>,
): number {
  return rates[category];
}
