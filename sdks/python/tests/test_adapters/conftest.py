"""Adapter test fixtures."""

import pytest
from cognitive_memory.adapters.memory import InMemoryAdapter


@pytest.fixture
def in_memory_adapter():
    """Create a fresh InMemoryAdapter for each test."""
    return InMemoryAdapter()


@pytest.fixture(params=["in_memory"])
def adapter(request):
    """Parametrized fixture for adapter conformance tests."""
    if request.param == "in_memory":
        return InMemoryAdapter()
    raise ValueError(f"Unknown adapter: {request.param}")
