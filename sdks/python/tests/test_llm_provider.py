"""
Tests for the LLMProvider abstraction.

Verifies that an injected LLM provider replaces the default OpenAI client
end-to-end, so the SDK can run without OPENAI_API_KEY when the caller wires
their own provider.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import pytest

from cognitive_memory import CognitiveMemoryConfig, MemoryExtractor
from cognitive_memory.llm import LLMProvider, LLMUsage


class StubLLM(LLMProvider):
    """Records the prompts it received and returns a canned response."""

    def __init__(self, response: str, usage: Optional[LLMUsage] = None):
        self._response = response
        self._usage = usage or {"prompt_tokens": 1, "completion_tokens": 1}
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self._response

    def complete_with_usage(
        self, prompt: str, **kwargs
    ) -> tuple[str, LLMUsage]:
        self.calls.append(prompt)
        return self._response, self._usage


@pytest.fixture
def no_openai_key(monkeypatch):
    """Guarantee no OPENAI_API_KEY is set so the stub path is the only path."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_extractor_uses_injected_llm_provider(no_openai_key):
    """An injected LLMProvider becomes the only LLM seam, so extraction works
    without OPENAI_API_KEY and yields the LLM-returned content verbatim."""
    canned = json.dumps(
        [
            {
                "content": "Alex is 32 years old",
                "category": "core",
                "importance": 0.9,
                "memory_type": "fact",
            }
        ]
    )
    stub = StubLLM(response=canned)
    extractor = MemoryExtractor(CognitiveMemoryConfig(), llm=stub)

    memories = extractor.extract_from_conversation(
        conversation_text="User: I'm 32.",
        session_id="s1",
        timestamp=datetime(2026, 1, 1),
    )

    assert len(memories) == 1
    assert memories[0].content == "Alex is 32 years old"
    assert "OPENAI_API_KEY" not in os.environ


def test_default_extractor_uses_openai_provider():
    """When no llm is passed, MemoryExtractor falls back to OpenAILLMProvider.
    Construction must not require OPENAI_API_KEY (lazy client)."""
    from cognitive_memory.llm import OpenAILLMProvider

    extractor = MemoryExtractor(CognitiveMemoryConfig())
    assert isinstance(extractor._llm, OpenAILLMProvider)


def test_conflict_detection_uses_injected_provider(no_openai_key):
    """detect_conflict() routes through the same LLM seam as extraction."""
    from cognitive_memory.types import Memory, MemoryCategory

    stub = StubLLM(response="CONTRADICTION")
    extractor = MemoryExtractor(CognitiveMemoryConfig(), llm=stub)

    existing = Memory(content="prefers tea", category=MemoryCategory.SEMANTIC)
    new = Memory(content="prefers coffee now", category=MemoryCategory.SEMANTIC)

    label = extractor.detect_conflict(new, existing)
    assert label == "CONTRADICTION"


def test_malformed_json_falls_back_to_safe_default(no_openai_key):
    """When the LLM returns junk, extraction yields a single fallback
    memory containing the source text (existing safe-default behaviour)."""
    stub = StubLLM(response="not json at all <<<")
    extractor = MemoryExtractor(CognitiveMemoryConfig(), llm=stub)

    memories = extractor.extract_from_conversation(
        conversation_text="User: hello world",
        session_id="s1",
        timestamp=datetime(2026, 1, 1),
    )

    assert len(memories) == 1
    assert "hello world" in memories[0].content


def test_complete_with_usage_returns_token_counts(no_openai_key):
    """Provider's complete_with_usage returns (text, usage) where usage
    carries prompt_tokens and completion_tokens. This is the contract that
    the rerank stage and tracing depend on."""
    stub = StubLLM(
        response='[{"content": "x", "category": "semantic", "importance": 0.5}]',
        usage={"prompt_tokens": 42, "completion_tokens": 7},
    )
    text, usage = stub.complete_with_usage("any prompt")

    assert text.startswith("[")
    assert usage["prompt_tokens"] == 42
    assert usage["completion_tokens"] == 7
