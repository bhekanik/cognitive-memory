"""Adapter implementations for cognitive-memory storage backends."""

from .base import MemoryAdapter
from .memory import InMemoryAdapter

__all__ = [
    "MemoryAdapter",
    "InMemoryAdapter",
]
