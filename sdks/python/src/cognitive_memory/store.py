"""
Backward compatibility shim.

The MemoryStore class has been replaced by the adapter pattern.
Import InMemoryAdapter from .adapters.memory instead.
"""

from .adapters.memory import InMemoryAdapter as MemoryStore  # noqa: F401

__all__ = ["MemoryStore"]
