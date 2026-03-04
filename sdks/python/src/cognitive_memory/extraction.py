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

from .types import Memory, MemoryCategory, CognitiveMemoryConfig


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
Example: [{{"content": "Alex is a 32-year-old software engineer", "category": "core", "importance": 0.9}}, {{"content": "Alex is single", "category": "core", "importance": 0.7}}, {{"content": "Alex moved from Germany three years ago", "category": "core", "importance": 0.8}}, {{"content": "Sam has two dogs named Biscuit and Maple", "category": "core", "importance": 0.8}}, {{"content": "Alex finished reading The Great Gatsby in January 2024", "category": "episodic", "importance": 0.5}}, {{"content": "Sam ran a 5K for charity the weekend before March 10, 2024", "category": "episodic", "importance": 0.5}}]"""


CONFLICT_PROMPT = """Does the new memory contradict or update an existing memory?

Existing memory: "{existing}"
New memory: "{new}"

Respond with exactly one word: CONTRADICTION, UPDATE, OVERLAP, or NONE.
- CONTRADICTION: the new memory directly negates the existing one
- UPDATE: the new memory is a newer version of the same fact
- OVERLAP: they cover similar ground but don't conflict
- NONE: they are unrelated"""


CONSOLIDATION_PROMPT = """Compress these related memories into a single concise summary that preserves all key facts.

Memories:
{memories}

Write one clear paragraph. Preserve specific names, dates, numbers, and preferences. Do not add information that isn't in the originals."""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """Extracts structured memories from conversation text."""

    def __init__(self, config: CognitiveMemoryConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        import time as _time
        client = self._get_client()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.config.extraction_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2 and ("500" in str(e) or "server_error" in str(e)):
                    _time.sleep(2 ** attempt)
                    continue
                raise

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

            mem = Memory(
                content=content,
                category=category,
                importance=importance,
                stability=0.1 + (importance * 0.3),
                created_at=timestamp,
                last_accessed_at=timestamp,
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

    def compress_memories(self, contents: list[str]) -> str:
        """Compress a group of memories into a summary."""
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contents))
        prompt = CONSOLIDATION_PROMPT.format(memories=numbered)
        return self._call_llm(prompt, max_tokens=500)
