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
import calendar
from datetime import datetime, timedelta
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


def _resolve_relative_datetime(raw: str, reference: datetime) -> Optional[datetime]:
    """Resolve a natural-language date phrase against a session timestamp."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    lowered = text.lower()

    # Deterministic fallbacks keep core behavior testable when optional
    # dateparser is not installed in a local SDK checkout.
    if lowered in {"yesterday", "the day before"}:
        return reference - timedelta(days=1)
    if lowered in {"today", "now"}:
        return reference
    if lowered == "tomorrow":
        return reference + timedelta(days=1)
    if lowered in {"next month", "the next month"}:
        month = reference.month + 1
        year = reference.year + (1 if month > 12 else 0)
        month = 1 if month > 12 else month
        day = min(reference.day, calendar.monthrange(year, month)[1])
        return reference.replace(year=year, month=month, day=day)
    if lowered in {"last month", "previous month"}:
        month = reference.month - 1
        year = reference.year - (1 if month < 1 else 0)
        month = 12 if month < 1 else month
        day = min(reference.day, calendar.monthrange(year, month)[1])
        return reference.replace(year=year, month=month, day=day)

    try:
        import dateparser

        parsed = dateparser.parse(
            text,
            settings={
                "RELATIVE_BASE": reference,
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": reference.tzinfo is not None,
            },
        )
        if parsed is not None:
            return parsed
    except Exception:
        pass

    return _parse_optional_datetime(text)


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _normalise_event_time(value, reference: datetime) -> dict:
    if isinstance(value, dict):
        raw = value.get("raw_expression") or value.get("raw")
        start = _parse_optional_datetime(value.get("start"))
        end = _parse_optional_datetime(value.get("end"))
        if start is None and isinstance(raw, str):
            start = _resolve_relative_datetime(raw, reference)
        return {
            "start": _iso_or_none(start),
            "end": _iso_or_none(end),
            "granularity": value.get("granularity", "unknown"),
            "raw_expression": raw,
            "confidence": float(value.get("confidence", 0.6)),
        }
    if isinstance(value, str):
        resolved = _resolve_relative_datetime(value, reference)
        return {
            "start": _iso_or_none(resolved),
            "end": None,
            "granularity": "unknown",
            "raw_expression": value,
            "confidence": 0.5 if resolved is not None else 0.0,
        }
    return {}


def _normalise_temporal_metadata(
    item: dict,
    session_id: str,
    timestamp: datetime,
    valid_from: Optional[datetime],
    valid_until: Optional[datetime],
) -> dict:
    valid_time = item.get("valid_time") if isinstance(item.get("valid_time"), dict) else {}
    status = item.get("status") or valid_time.get("status") or "unknown"
    raw_expressions = item.get("raw_time_expressions", [])
    if not isinstance(raw_expressions, list):
        raw_expressions = [str(raw_expressions)]

    return {
        "mentioned_at": {
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
        },
        "event_time": _normalise_event_time(item.get("event_time"), timestamp),
        "valid_time": {
            "valid_from": _iso_or_none(valid_from) or valid_time.get("valid_from"),
            "valid_to": _iso_or_none(valid_until) or valid_time.get("valid_to"),
            "status": str(status),
        },
        "raw_time_expressions": [str(expr) for expr in raw_expressions],
        "relations": item.get("temporal_relations", []) if isinstance(item.get("temporal_relations"), list) else [],
    }


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
- event_time: (optional) object with start/end ISO strings, raw_expression, granularity, confidence. Set when a specific event time is mentioned or resolvable.
- valid_time: REQUIRED object with at least a status. Pick the best fit — DO NOT default to "unknown":
  - "current": semantic facts, preferences, identity, ongoing states. The default for lasting memories.
  - "planned": future intentions, scheduled events.
  - "in_progress": ongoing activities.
  - "completed": past one-time events that have finished.
  - "cancelled": abandoned plans.
  - "superseded": replaced by a later fact (rarely set at extraction).
  - "hypothetical": conditional or counterfactual.
- event_frame: (optional) object with event_type, subjects, action, objects, location.
- raw_time_expressions: REQUIRED array of every original time phrase tied to this memory. Use [] only when the memory is genuinely timeless. Include all of: explicit dates ("January 21, 2022"), relative phrases ("yesterday", "last Friday", "3 years ago", "recently"), durations ("for 3 years"), and current-state markers ("currently", "still", "now"). For memories anchored to "around the conversation date" without an explicit phrase, include the conversation date.

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
8. EVERY memory MUST have valid_time.status and raw_time_expressions populated. Lasting facts and preferences get status="current" and raw_time_expressions including any duration/recency phrases or the conversation date. Past events get status="completed" and the resolved date(s).

Conversation:
{conversation}

Respond with a JSON array only. No markdown, no preamble.
Example (every memory has valid_time.status AND raw_time_expressions): [{{"content": "Alex is a 32-year-old software engineer", "category": "core", "importance": 0.9, "memory_type": "fact", "valid_time": {{"status": "current"}}, "raw_time_expressions": ["currently"]}}, {{"content": "Alex prefers window seats on flights", "category": "semantic", "importance": 0.5, "memory_type": "preference", "valid_time": {{"status": "current"}}, "raw_time_expressions": ["currently"]}}, {{"content": "Alex has a dentist appointment on March 15, 2024", "category": "episodic", "importance": 0.6, "memory_type": "plan", "event_time": {{"start": "2024-03-15T00:00:00", "raw_expression": "March 15, 2024", "granularity": "day", "confidence": 0.9}}, "valid_until": "2024-03-15T23:59:59", "valid_time": {{"status": "planned"}}, "raw_time_expressions": ["March 15, 2024"], "event_frame": {{"event_type": "plan", "subjects": ["Alex"], "action": "attend", "objects": ["dentist appointment"]}}}}, {{"content": "Alex is feeling stressed about the deadline", "category": "episodic", "importance": 0.4, "memory_type": "transient_state", "valid_time": {{"status": "in_progress"}}, "raw_time_expressions": ["currently"]}}, {{"content": "Sam ran a 5K for charity the weekend before March 10, 2024", "category": "episodic", "importance": 0.5, "memory_type": "fact", "event_time": {{"start": "2024-03-03T00:00:00", "raw_expression": "the weekend before March 10, 2024", "granularity": "week", "confidence": 0.7}}, "valid_time": {{"status": "completed"}}, "raw_time_expressions": ["the weekend before March 10, 2024", "March 3, 2024"]}}]"""


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
            temporal = _normalise_temporal_metadata(
                item, session_id, timestamp, valid_from, valid_until,
            )
            event_frame = item.get("event_frame", {})
            if not isinstance(event_frame, dict):
                event_frame = {}

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
                temporal=temporal,
                event_frame=event_frame,
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
                temporal={
                    "mentioned_at": {
                        "session_id": session_id,
                        "timestamp": timestamp.isoformat(),
                    },
                    "event_time": {},
                    "valid_time": {"status": "unknown"},
                    "raw_time_expressions": [],
                    "relations": [],
                },
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
