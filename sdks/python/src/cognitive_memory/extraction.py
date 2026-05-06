"""
LLM-powered memory extraction and conflict detection.

Uses an LLM to:
1. Extract discrete memories from conversation turns
2. Classify memory type (episodic/semantic/procedural)
3. Assign importance scores
4. Detect core memory candidates
5. Detect conflicts with existing memories
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from .llm import LLMProvider, OpenAILLMProvider
from .types import Memory, MemoryCategory, CognitiveMemoryConfig

VALID_MEMORY_TYPES = {"fact", "preference", "plan", "transient_state", "other"}


def _parse_optional_datetime(value) -> Optional[datetime]:
    """Parse an ISO datetime string, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract ALL facts and events from this conversation. Be thorough — extract every distinct piece of information, no matter how brief or incidental.

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
Example: [{{"content": "Alex is a 32-year-old software engineer", "category": "core", "importance": 0.9, "memory_type": "fact"}}, {{"content": "Alex prefers window seats on flights", "category": "semantic", "importance": 0.5, "memory_type": "preference"}}, {{"content": "Alex has a dentist appointment on March 15, 2024", "category": "episodic", "importance": 0.6, "memory_type": "plan", "valid_until": "2024-03-15T23:59:59"}}, {{"content": "Alex is currently feeling stressed about the deadline", "category": "episodic", "importance": 0.4, "memory_type": "transient_state"}}, {{"content": "Sam ran a 5K for charity the weekend before March 10, 2024", "category": "episodic", "importance": 0.5, "memory_type": "fact"}}]"""


CONFLICT_PROMPT = """Does the new memory contradict or update an existing memory?

Existing memory: "{existing}"
New memory: "{new}"

Respond with exactly one word: CONTRADICTION, UPDATE, OVERLAP, or NONE.
- CONTRADICTION: the new memory directly negates the existing one
- UPDATE: the new memory is a newer version of the same fact
- OVERLAP: they cover similar ground but don't conflict
- NONE: they are unrelated"""


RERANK_PROMPT = """Given the query and a list of candidate memories, rerank them by relevance. Return a JSON array of indices (0-based) from most to least relevant. Only include indices of memories that are relevant to the query.

Query: "{query}"

Candidates:
{candidates}

Respond with a JSON array of indices only, e.g. [2, 0, 4, 1]. No explanation."""


CONSOLIDATION_PROMPT = """Compress these related memories into a single concise summary that preserves all key facts.

Memories:
{memories}

Write one clear paragraph. Preserve specific names, dates, numbers, and preferences. Do not add information that isn't in the originals."""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """Extracts structured memories from conversation text."""

    def __init__(
        self,
        config: CognitiveMemoryConfig,
        llm: Optional[LLMProvider] = None,
    ):
        self.config = config
        self._llm: LLMProvider = llm or OpenAILLMProvider(
            model=config.extraction_model
        )

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        text, _ = self._call_llm_with_usage(prompt, max_tokens=max_tokens)
        return text

    def extract_from_conversation(
        self,
        conversation_text: str,
        session_id: str,
        timestamp: datetime,
    ) -> list[Memory]:
        """
        Extract memories from a conversation using an LLM.

        Returns a list of Memory objects with content, category,
        importance, but without embeddings (caller must embed).
        """
        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)
        if self.config.custom_extraction_instructions:
            prompt = (
                f"IMPORTANT INSTRUCTIONS FOR MEMORY EXTRACTION:\n"
                f"{self.config.custom_extraction_instructions}\n\n"
                f"{prompt}"
            )
        raw = self._call_llm(prompt, max_tokens=2000)
        items = self._parse_extraction_response(raw, conversation_text)
        return self._build_memories(items, session_id, timestamp)

    def _parse_extraction_response(self, raw: str, fallback_text: str) -> list[dict]:
        """Parse LLM extraction response into list of dicts."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return [{
                "content": fallback_text[:500],
                "category": "episodic",
                "importance": 0.5,
            }]

    def _build_memories(self, items: list[dict], session_id: str, timestamp: datetime) -> list[Memory]:
        """Convert parsed dicts into Memory objects."""
        memories = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "").strip()
            if not content:
                continue

            cat_str = item.get("category", "episodic").lower()
            try:
                category = MemoryCategory(cat_str)
            except ValueError:
                category = MemoryCategory.EPISODIC

            importance = float(item.get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))

            # v6: Parse semantic type and validity
            memory_type = item.get("memory_type", "other")
            if memory_type not in VALID_MEMORY_TYPES:
                memory_type = "other"

            valid_from = _parse_optional_datetime(item.get("valid_from"))
            valid_until = _parse_optional_datetime(item.get("valid_until"))
            ttl_seconds = item.get("ttl_seconds")
            if ttl_seconds is not None:
                try:
                    ttl_seconds = int(ttl_seconds)
                except (ValueError, TypeError):
                    ttl_seconds = None

            source_turn_ids = item.get("source_turn_ids", [])
            if not isinstance(source_turn_ids, list):
                source_turn_ids = []
            source_turn_ids = [str(t) for t in source_turn_ids]

            mem = Memory(
                content=content,
                category=category,
                importance=importance,
                stability=0.1 + (importance * 0.3),
                created_at=timestamp,
                last_accessed_at=timestamp,
                memory_type=memory_type,
                valid_from=valid_from,
                valid_until=valid_until,
                ttl_seconds=ttl_seconds,
                source_turn_ids=source_turn_ids,
            )
            mem.session_ids.add(session_id)
            memories.append(mem)

        return memories

    def detect_conflict(
        self,
        new_memory: Memory,
        existing_memory: Memory,
    ) -> str:
        """
        Detect if a new memory conflicts with an existing one.
        Returns: "CONTRADICTION", "UPDATE", "OVERLAP", or "NONE"
        """
        prompt = CONFLICT_PROMPT.format(
            existing=existing_memory.content,
            new=new_memory.content,
        )
        raw = self._call_llm(prompt, max_tokens=20)
        raw_upper = raw.strip().upper()

        for label in ["CONTRADICTION", "UPDATE", "OVERLAP", "NONE"]:
            if label in raw_upper:
                return label
        return "NONE"

    def extract_raw_turns(
        self,
        conversation_text: str,
        session_id: str,
        timestamp: datetime,
    ) -> list[Memory]:
        """
        Parse conversation into individual turns and store each verbatim.
        No LLM extraction — preserves exact dialog for granular retrieval.
        """
        lines = conversation_text.strip().split("\n")
        memories = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip header lines like "[This conversation took place on ...]"
            if line.startswith("[") and line.endswith("]"):
                continue
            mem = Memory(
                content=line,
                category=MemoryCategory.EPISODIC,
                importance=0.5,
                stability=0.2,
                created_at=timestamp,
                last_accessed_at=timestamp,
            )
            mem.session_ids.add(session_id)
            memories.append(mem)
        return memories

    def compress_memories(self, contents: list[str]) -> str:
        """Compress a group of memories into a summary."""
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contents))
        prompt = CONSOLIDATION_PROMPT.format(memories=numbered)
        return self._call_llm(prompt, max_tokens=500)

    def _call_llm_with_usage(
        self, prompt: str, max_tokens: int = 200, model: Optional[str] = None,
    ) -> tuple[str, dict]:
        """Call the injected :class:`LLMProvider` and return (text, usage)."""
        return self._llm.complete_with_usage(
            prompt,
            max_tokens=max_tokens,
            model=model or self.config.extraction_model,
        )

    def rerank_candidates(
        self, query: str, candidates: list[str],
    ) -> tuple[list[int], dict]:
        """
        Rerank candidates using LLM. Returns (reranked_indices, usage_dict).
        usage_dict has prompt_tokens and completion_tokens.
        """
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        prompt = RERANK_PROMPT.format(query=query, candidates=numbered)
        model = self.config.rerank_model or self.config.extraction_model
        text, usage = self._call_llm_with_usage(prompt, max_tokens=200, model=model)

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                # Validate indices
                seen = set()
                indices = []
                for n in parsed:
                    if isinstance(n, int) and 0 <= n < len(candidates) and n not in seen:
                        indices.append(n)
                        seen.add(n)
                return indices, usage
        except (json.JSONDecodeError, ValueError):
            pass

        return list(range(len(candidates))), usage
