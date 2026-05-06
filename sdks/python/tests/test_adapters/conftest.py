"""Adapter test fixtures."""

import pytest
from cognitive_memory.adapters.memory import InMemoryAdapter
from cognitive_memory.adapters.jsonl import JsonlFileAdapter


@pytest.fixture
def in_memory_adapter():
    """Create a fresh InMemoryAdapter for each test."""
    return InMemoryAdapter()


@pytest.fixture(params=["in_memory", "jsonl"])
def adapter(request, tmp_path):
    """Parametrized fixture for adapter conformance tests.

    Every concrete adapter implementation must pass the same suite — one
    contract, many backends.
    """
    if request.param == "in_memory":
        return InMemoryAdapter()
    if request.param == "jsonl":
        return JsonlFileAdapter(str(tmp_path / "store.jsonl"))
    raise ValueError(f"Unknown adapter: {request.param}")
