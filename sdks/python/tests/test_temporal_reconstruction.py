from __future__ import annotations

from datetime import datetime

from cognitive_memory import CognitiveMemoryConfig, Memory, MemoryCategory, SyncCognitiveMemory
from cognitive_memory.embeddings import EmbeddingProvider
from cognitive_memory.extraction import MemoryExtractor


class StaticLLM:
    def __init__(self, text: str):
        self.text = text

    def complete_with_usage(self, prompt: str, max_tokens: int = 1000, model: str | None = None):
        return self.text, {}


class FixedEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [1, 0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_extraction_resolves_relative_event_time():
    llm = StaticLLM(
        '[{"content":"Alex sent the Oxford application yesterday",'
        '"category":"episodic","importance":0.7,"memory_type":"fact",'
        '"event_time":{"raw_expression":"yesterday","granularity":"day","confidence":0.9},'
        '"status":"completed"}]'
    )
    extractor = MemoryExtractor(CognitiveMemoryConfig(), llm=llm)

    memories = extractor.extract_from_conversation(
        "User: I sent the Oxford application yesterday.",
        "session_1",
        datetime(2024, 3, 10, 12, 0, 0),
    )

    assert memories[0].temporal["event_time"]["start"].startswith("2024-03-09")
    assert memories[0].temporal["valid_time"]["status"] == "completed"


def test_extraction_clamps_month_relative_event_time():
    llm = StaticLLM(
        '[{"content":"Alex visited the gallery last month",'
        '"category":"episodic","importance":0.7,"memory_type":"fact",'
        '"event_time":{"raw_expression":"last month","granularity":"month","confidence":0.9},'
        '"status":"completed"}]'
    )
    extractor = MemoryExtractor(CognitiveMemoryConfig(), llm=llm)

    memories = extractor.extract_from_conversation(
        "User: I visited the gallery last month.",
        "session_1",
        datetime(2023, 10, 31, 12, 0, 0),
    )

    assert memories[0].temporal["event_time"]["start"].startswith("2023-09-30")


def test_temporal_query_returns_chronological_evidence():
    config = CognitiveMemoryConfig(temporal_query_mode="auto")
    memory = SyncCognitiveMemory(config=config, embedder="hash")

    memory.add_memory_object(
        Memory(
            content="Alex resumed light running.",
            category=MemoryCategory.EPISODIC,
            created_at=datetime(2024, 1, 20),
            last_accessed_at=datetime(2024, 1, 20),
            temporal={
                "mentioned_at": {"session_id": "s3", "timestamp": "2024-01-20T00:00:00"},
                "event_time": {"start": "2024-01-20T00:00:00", "confidence": 0.9},
                "valid_time": {"status": "completed"},
                "raw_time_expressions": ["January 20"],
            },
        )
    )
    memory.add_memory_object(
        Memory(
            content="Alex injured her ankle.",
            category=MemoryCategory.EPISODIC,
            created_at=datetime(2024, 1, 5),
            last_accessed_at=datetime(2024, 1, 5),
            temporal={
                "mentioned_at": {"session_id": "s1", "timestamp": "2024-01-05T00:00:00"},
                "event_time": {"start": "2024-01-05T00:00:00", "confidence": 0.9},
                "valid_time": {"status": "completed"},
                "raw_time_expressions": ["January 5"],
            },
        )
    )

    result = memory.search("What happened after Alex injured her ankle?", top_k=2)

    assert [item["content"] for item in result.temporal_evidence] == [
        "Alex injured her ankle.",
        "Alex resumed light running.",
    ]


def test_current_temporal_query_keeps_current_state_first():
    config = CognitiveMemoryConfig(temporal_query_mode="auto")
    memory = SyncCognitiveMemory(config=config, embedder=FixedEmbeddingProvider())

    memory.add_memory_object(
        Memory(
            content="Jamie lives in Manchester.",
            category=MemoryCategory.SEMANTIC,
            created_at=datetime(2024, 1, 1),
            last_accessed_at=datetime(2024, 1, 1),
            temporal={
                "event_time": {"start": "2024-01-01T00:00:00", "confidence": 0.9},
                "valid_time": {"status": "superseded", "valid_to": "2024-03-01T00:00:00"},
            },
        )
    )
    memory.add_memory_object(
        Memory(
            content="Jamie lives in London.",
            category=MemoryCategory.SEMANTIC,
            created_at=datetime(2024, 3, 1),
            last_accessed_at=datetime(2024, 3, 1),
            temporal={
                "event_time": {"start": "2024-03-01T00:00:00", "confidence": 0.9},
                "valid_time": {"status": "current", "valid_from": "2024-03-01T00:00:00"},
            },
        )
    )

    result = memory.search("Where does Jamie live now?", top_k=2)

    assert result.results[0].memory.content == "Jamie lives in London."
    assert result.temporal_evidence[0]["content"] == "Jamie lives in London."


def test_is_temporal_query_precision():
    from cognitive_memory.engine import _is_temporal_query

    # Asks for a time / duration / current state -> temporal.
    temporal = [
        "When did Nate win his first video game tournament?",
        "When is Joanna going to make ice cream?",
        "How long has Nate had his turtles?",
        "For how long has Nate had his turtles?",
        "How often does Joanna write?",
        "What year did Joanna graduate?",
        "Since when has Maria volunteered?",
        "What happened after Alex injured her ankle?",
        "Where does Jamie live now?",
        "What is Maria's current job?",
        "Does Maria still go to the gym?",
    ]
    # Mentions a time word in a subordinate clause but asks what/how/who -> not.
    non_temporal = [
        "How did Joanna feel when someone wrote her a letter after reading her blog post?",
        "What did Joanna take a picture of near Fort Wayne last summer?",
        "What genre is Joanna's first screenplay?",
        "Was the first half of September 2022 a good month career-wise for Nate?",
        "What does Maria know about gardening?",  # 'know' must not match 'now'
        "What did Joanna just finish last Friday?",
    ]
    for q in temporal:
        assert _is_temporal_query(q), f"should be temporal: {q!r}"
    for q in non_temporal:
        assert not _is_temporal_query(q), f"should NOT be temporal: {q!r}"
