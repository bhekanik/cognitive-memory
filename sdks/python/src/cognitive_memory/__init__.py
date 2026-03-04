"""
cognitive-memory: A Python SDK for biologically-inspired agent memory.

Implements decay floors, emergent core memory detection, two-tier
retrieval boosting, associative linking, tiered hot/cold storage
with TTL, and reversible consolidation.

Quick start:
    from cognitive_memory import SyncCognitiveMemory

    mem = SyncCognitiveMemory(embedder="hash")
    mem.add("User is allergic to shellfish", category="core", importance=0.95)
    results = mem.search("what allergies does the user have?")
"""

from .core import CognitiveMemory
from ._sync import SyncCognitiveMemory
from .types import (
    Memory,
    MemoryCategory,
    CognitiveMemoryConfig,
    SearchResult,
    Association,
)
from .engine import CognitiveEngine
from .extraction import MemoryExtractor
from .embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HashEmbeddings,
    cosine_similarity,
)
from .adapters import MemoryAdapter, InMemoryAdapter

# Backward compat
from .store import MemoryStore

__version__ = "0.1.0"

__all__ = [
    # Main API
    "CognitiveMemory",
    "SyncCognitiveMemory",
    # Types
    "CognitiveMemoryConfig",
    "Memory",
    "MemoryCategory",
    "SearchResult",
    "Association",
    # Engine
    "CognitiveEngine",
    # Extraction
    "MemoryExtractor",
    # Embeddings
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "HashEmbeddings",
    "cosine_similarity",
    # Adapters
    "MemoryAdapter",
    "InMemoryAdapter",
    # Backward compat
    "MemoryStore",
]
