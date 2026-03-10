export { CognitiveMemory } from "./CognitiveMemory";
export { BASE_DECAY_RATES, DECAY_FLOORS, calculateRetention, updateStability } from "./decay";
export { CognitiveEngine } from "./engine";
export {
  extractFromConversation,
  detectConflict,
  compressMemories,
  rerankCandidates,
  EXTRACTION_PROMPT,
  CONFLICT_PROMPT,
  CONSOLIDATION_PROMPT,
  RERANK_PROMPT,
} from "./extraction";
export type { LLMProvider, LLMUsage, ConflictType, RerankResult } from "./extraction";
export {
  OpenAIEmbeddingProvider,
  HashEmbeddingProvider,
} from "./embeddings";
export type {
  OpenAIEmbeddingProviderOptions,
  HashEmbeddingProviderOptions,
} from "./embeddings";
export {
  DEFAULT_CONFIG,
  resolveConfig,
  createDefaultMemory,
  categoryToMemoryType,
  getRetentionFloor,
  getBaseDecayRate,
} from "./types";
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
} from "./types";
