// Adapters
export type { ConvexAdapterFunctions, JsonlFileAdapterOptions, MemoryFilters, PostgresAdapterOptions } from "./adapters";
export { ConvexAdapter, InMemoryAdapter, JsonlFileAdapter, MemoryAdapter, PostgresAdapter, postgresSchemaSql } from "./adapters";

// Core types
export type {
  Association,
  CognitiveMemoryConfig,
  ConsolidationResult,
  DecayParameters,
  EmbeddingProvider,
  Memory,
  MemoryCategory,
  MemoryInput,
  MemoryLink,
  MemoryStats,
  MemoryType,
  ResolvedCognitiveMemoryConfig,
  RetrievalQuery,
  ScoredMemory,
  SearchResponse,
  SearchResult,
  SearchTrace,
  SemanticType,
  StageTrace,
} from "./core";

// Core classes and functions
export {
  BASE_DECAY_RATES,
  DECAY_FLOORS,
  CognitiveEngine,
  CognitiveMemory,
  DEFAULT_CONFIG,
  HashEmbeddingProvider,
  OpenAIEmbeddingProvider,
  calculateRetention,
  categoryToMemoryType,
  createDefaultMemory,
  getBaseDecayRate,
  getRetentionFloor,
  resolveConfig,
  updateStability,
} from "./core";

// Extraction
export {
  CONFLICT_PROMPT,
  CONSOLIDATION_PROMPT,
  EXTRACTION_PROMPT,
  RERANK_PROMPT,
  compressMemories,
  detectConflict,
  extractFromConversation,
  rerankCandidates,
} from "./core";
export type { ConflictType, LLMProvider, LLMUsage, RerankResult } from "./core";

// Embedding providers
export type {
  HashEmbeddingProviderOptions,
  OpenAIEmbeddingProviderOptions,
} from "./core";

// Utils
export {
  categorizeMemoryType,
  cosineSimilarity,
  euclideanDistance,
  extractTopics,
  normalizeVector,
  scoreImportance,
} from "./utils";
