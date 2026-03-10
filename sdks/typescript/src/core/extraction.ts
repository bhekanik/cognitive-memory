/**
 * Cognitive Memory System - LLM Extraction & Conflict Detection
 *
 * Uses an LLM to:
 * 1. Extract discrete memories from conversation turns
 * 2. Classify memory category (episodic/semantic/procedural/core)
 * 3. Assign importance scores
 * 4. Detect conflicts with existing memories
 * 5. Compress memory groups during consolidation
 */

import type { Memory, MemoryCategory, ResolvedCognitiveMemoryConfig, SemanticType } from "./types";
import { createDefaultMemory, categoryToMemoryType } from "./types";

// ---------------------------------------------------------------------------
// LLM Provider interface
// ---------------------------------------------------------------------------

export interface LLMUsage {
  promptTokens?: number;
  completionTokens?: number;
}

export interface LLMProvider {
  complete(
    prompt: string,
    options?: {
      model?: string;
      maxTokens?: number;
      temperature?: number;
    },
  ): Promise<string>;

  /**
   * Optional: complete with token usage tracking.
   * If not implemented, falls back to complete() with no usage info.
   */
  completeWithUsage?(
    prompt: string,
    options?: {
      model?: string;
      maxTokens?: number;
      temperature?: number;
    },
  ): Promise<{ text: string; usage: LLMUsage }>;
}

// ---------------------------------------------------------------------------
// Prompts (ported from Python SDK)
// ---------------------------------------------------------------------------

export const EXTRACTION_PROMPT = `Extract ALL facts and events from this conversation. Be thorough — extract every distinct piece of information, no matter how brief or incidental.

You are a NARRATOR, not a summarizer. Record what happened and what was said, not your interpretation of it.

For each memory, provide:
- content: one specific fact or event in a clear sentence. INCLUDE specific names, dates, numbers, and places.
- category:
  - "core": identity info (name, age, gender, relationship status, nationality, medical, family members, profession, where they live/moved from)
  - "semantic": lasting facts, preferences, plans, relationships, opinions, hobbies. DEFAULT if unsure.
  - "episodic": specific one-time events with dates/times
  - "procedural": routines, habits, skills
- importance: 0.0 to 1.0
- memory_type: "fact" | "preference" | "plan" | "transient_state" | "other"
  - "fact": verifiable statement about world or user (e.g. "Alex is 32 years old")
  - "preference": user likes/dislikes (e.g. "Alex prefers dark roast coffee")
  - "plan": future intention or scheduled event (e.g. "Alex has a meeting at 3pm tomorrow")
  - "transient_state": temporary mood, location, current activity (e.g. "Alex is currently at the airport")
  - "other": default if none of the above apply
- valid_from: (optional) ISO date string when this becomes valid. Only for time-bounded memories.
- valid_until: (optional) ISO date string when this expires. Use for plans and transient states.
- source_turn_ids: (optional) array of turn numbers this was extracted from (e.g. [1, 3])

CRITICAL RULES:
1. NARRATE, don't interpret. Store WHAT HAPPENED, not what it means.
   BAD: "Alex enjoys outdoor activities" (interpretation)
   GOOD: "Alex went hiking at Mount Rainier on March 12, 2024" (what happened)
   BAD: "Sam is artistic" (interpretation)
   GOOD: "Sam painted a landscape of the lake in 2023" (what happened)
2. Extract EVERY specific event, activity, and experience mentioned — even brief ones. A picnic, a book read, a race run, a song listened to — ALL get their own memory.
3. RESOLVE relative dates using the conversation date at the top (e.g., conversation on "8 May 2023" + "yesterday" = May 7, 2023). Include resolved dates in the content.
4. For lasting facts (preferences, traits, relationships), extract those too as semantic memories.
5. Extract each distinct fact as a SEPARATE memory. One fact per memory.
6. If messages are labeled User and Assistant, PRIORITIZE extracting memories from User messages. User messages contain personal information we need to remember. Assistant messages are less important unless they contain facts the user confirmed.
7. Don't skip brief or passing mentions. If someone mentions a fact once in a single sentence, it's still a memory worth storing. A passing reference to a hometown, a book title, or a pet's name is just as important as a detailed story.

Conversation:
{conversation}

Respond with a JSON array only. No markdown, no preamble.
Example: [{"content": "Alex is a 32-year-old software engineer", "category": "core", "importance": 0.9, "memory_type": "fact"}, {"content": "Alex prefers window seats on flights", "category": "semantic", "importance": 0.5, "memory_type": "preference"}, {"content": "Alex has a dentist appointment on March 15, 2024", "category": "episodic", "importance": 0.6, "memory_type": "plan", "valid_until": "2024-03-15T23:59:59"}, {"content": "Alex is currently feeling stressed about the deadline", "category": "episodic", "importance": 0.4, "memory_type": "transient_state"}, {"content": "Sam ran a 5K for charity the weekend before March 10, 2024", "category": "episodic", "importance": 0.5, "memory_type": "fact"}]`;

export const CONFLICT_PROMPT = `Does the new memory contradict or update an existing memory?

Existing memory: "{existing}"
New memory: "{new}"

Respond with exactly one word: CONTRADICTION, UPDATE, OVERLAP, or NONE.
- CONTRADICTION: the new memory directly negates the existing one
- UPDATE: the new memory is a newer version of the same fact
- OVERLAP: they cover similar ground but don't conflict
- NONE: they are unrelated`;

export const RERANK_PROMPT = `Given the query and a list of candidate memories, rerank them by relevance. Return a JSON array of indices (0-based) from most to least relevant. Only include indices of memories that are relevant to the query.

Query: "{query}"

Candidates:
{candidates}

Respond with a JSON array of indices only, e.g. [2, 0, 4, 1]. No explanation.`;

export const CONSOLIDATION_PROMPT = `Compress these related memories into a single concise summary that preserves all key facts.

Memories:
{memories}

Write one clear paragraph. Preserve specific names, dates, numbers, and preferences. Do not add information that isn't in the originals.`;

// ---------------------------------------------------------------------------
// Conflict types
// ---------------------------------------------------------------------------

export type ConflictType = "CONTRADICTION" | "UPDATE" | "OVERLAP" | "NONE";

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

type MemoryWithoutIds = Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">;

const VALID_CATEGORIES = new Set<MemoryCategory>(["episodic", "semantic", "procedural", "core"]);
const VALID_SEMANTIC_TYPES = new Set<SemanticType>(["fact", "preference", "plan", "transient_state", "other"]);

function baseMemoryFields(
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
  now: number,
): Omit<MemoryWithoutIds, "content" | "category" | "memoryType" | "importance" | "stability" | "semanticType" | "validFrom" | "validUntil" | "ttlSeconds" | "sourceTurnIds"> {
  return {
    userId: config.userId,
    accessCount: 0,
    lastAccessed: now,
    retention: 1.0,
    associations: {},
    sessionIds: [sessionId],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
  };
}

function parseOptionalTimestamp(value: unknown): number | null {
  if (value == null) return null;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

export async function extractFromConversation(
  conversationText: string,
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): Promise<MemoryWithoutIds[]> {
  let prompt = EXTRACTION_PROMPT.replace("{conversation}", conversationText);

  if (config.customExtractionInstructions) {
    prompt = `IMPORTANT INSTRUCTIONS FOR MEMORY EXTRACTION:\n${config.customExtractionInstructions}\n\n${prompt}`;
  }

  const raw = await llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 2000,
    temperature: 0,
  });

  const items = parseExtractionResponse(raw, conversationText);
  return buildMemories(items, config, sessionId);
}

function parseExtractionResponse(
  raw: string,
  fallbackText: string,
): Array<Record<string, unknown>> {
  try {
    let cleaned = raw.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
    }
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) return parsed;
    return [{ content: fallbackText.slice(0, 500), category: "episodic", importance: 0.5 }];
  } catch {
    return [{ content: fallbackText.slice(0, 500), category: "episodic", importance: 0.5 }];
  }
}

function buildMemories(
  items: Array<Record<string, unknown>>,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): MemoryWithoutIds[] {
  const now = Date.now();
  const memories: MemoryWithoutIds[] = [];

  for (const item of items) {
    if (typeof item !== "object" || item === null) continue;

    const content = typeof item.content === "string" ? item.content.trim() : "";
    if (!content) continue;

    const catStr = (typeof item.category === "string" ? item.category.toLowerCase() : "semantic") as MemoryCategory;
    const category: MemoryCategory = VALID_CATEGORIES.has(catStr) ? catStr : "semantic";

    let importance = typeof item.importance === "number" ? item.importance : 0.5;
    importance = Math.max(0.0, Math.min(1.0, importance));

    const rawSemanticType = typeof item.memory_type === "string" ? item.memory_type : "other";
    const semanticType: SemanticType = VALID_SEMANTIC_TYPES.has(rawSemanticType as SemanticType)
      ? (rawSemanticType as SemanticType)
      : "other";

    let sourceTurnIds: string[] = [];
    if (Array.isArray(item.source_turn_ids)) {
      sourceTurnIds = item.source_turn_ids.map((t: unknown) => String(t));
    }

    memories.push({
      ...baseMemoryFields(config, sessionId, now),
      content,
      category,
      memoryType: categoryToMemoryType(category),
      importance,
      stability: 0.1 + importance * 0.3,
      semanticType,
      validFrom: parseOptionalTimestamp(item.valid_from),
      validUntil: parseOptionalTimestamp(item.valid_until),
      ttlSeconds: typeof item.ttl_seconds === "number" ? Math.floor(item.ttl_seconds) : null,
      sourceTurnIds,
    });
  }

  return memories;
}

// ---------------------------------------------------------------------------
// Raw turn extraction (no LLM)
// ---------------------------------------------------------------------------

export function extractRawTurns(
  conversationText: string,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): MemoryWithoutIds[] {
  const now = Date.now();
  const lines = conversationText.trim().split("\n");
  const memories: MemoryWithoutIds[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("[") && line.endsWith("]")) continue;

    memories.push({
      ...baseMemoryFields(config, sessionId, now),
      content: line,
      category: "episodic",
      memoryType: "episodic",
      importance: 0.5,
      stability: 0.2,
      semanticType: "other",
      validFrom: null,
      validUntil: null,
      ttlSeconds: null,
      sourceTurnIds: [],
    });
  }

  return memories;
}

// ---------------------------------------------------------------------------
// Conflict detection
// ---------------------------------------------------------------------------

export async function detectConflict(
  existingMemory: Memory,
  newMemory: { content: string },
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
): Promise<ConflictType> {
  const prompt = CONFLICT_PROMPT
    .replace("{existing}", existingMemory.content)
    .replace("{new}", newMemory.content);

  const raw = await llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 20,
    temperature: 0,
  });

  const upper = raw.trim().toUpperCase();
  for (const label of ["CONTRADICTION", "UPDATE", "OVERLAP", "NONE"] as ConflictType[]) {
    if (upper.includes(label)) return label;
  }
  return "NONE";
}

// ---------------------------------------------------------------------------
// Memory compression (for consolidation)
// ---------------------------------------------------------------------------

export async function compressMemories(
  contents: string[],
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
): Promise<string> {
  const numbered = contents.map((c, i) => `${i + 1}. ${c}`).join("\n");
  const prompt = CONSOLIDATION_PROMPT.replace("{memories}", numbered);

  return llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 500,
    temperature: 0,
  });
}

// ---------------------------------------------------------------------------
// LLM reranking (v6)
// ---------------------------------------------------------------------------

export interface RerankResult {
  rerankedIndices: number[];
  usage: LLMUsage;
}

export async function rerankCandidates(
  query: string,
  candidates: Array<{ content: string }>,
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
): Promise<RerankResult> {
  const numbered = candidates
    .map((c, i) => `[${i}] ${c.content}`)
    .join("\n");
  const prompt = RERANK_PROMPT
    .replace("{query}", query)
    .replace("{candidates}", numbered);

  const model = config.rerankModel ?? config.extractionModel;
  let text: string;
  let usage: LLMUsage = {};

  if (llm.completeWithUsage) {
    const result = await llm.completeWithUsage(prompt, {
      model,
      maxTokens: 200,
      temperature: 0,
    });
    text = result.text;
    usage = result.usage;
  } else {
    text = await llm.complete(prompt, {
      model,
      maxTokens: 200,
      temperature: 0,
    });
  }

  try {
    let cleaned = text.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
    }
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) {
      const indices = [...new Set(
        parsed.filter(
          (n): n is number =>
            Number.isInteger(n) && n >= 0 && n < candidates.length,
        ),
      )];
      return { rerankedIndices: indices, usage };
    }
  } catch {
    // Fallback: return original order
  }

  return {
    rerankedIndices: candidates.map((_, i) => i),
    usage,
  };
}
