"""
Embedding providers for cognitive-memory.

Supports OpenAI embeddings and a local numpy fallback for testing.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for embedding text."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI text-embedding-3-small (or any model)."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self._model = model
        self._dims = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = client.embeddings.create(input=texts, model=self._model)
        # sort by index to preserve order
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class HashEmbeddings(EmbeddingProvider):
    """
    Deterministic hash-based embeddings for testing without API calls.

    Not semantically meaningful, but consistent (same text = same vector)
    and fast. Useful for unit tests and dry runs.
    """

    def __init__(self, dimensions: int = 384):
        self._dims = dimensions

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        # hash the text to get a deterministic seed
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dims).astype(np.float32)
        # normalize to unit length
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
