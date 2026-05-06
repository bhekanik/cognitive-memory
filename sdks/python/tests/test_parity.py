"""
Cross-SDK parity tests.

Both SDKs (Python and TypeScript) read the same scripted scenario JSON,
run it through the public API with deterministic embedders, and assert
on the observable result. The expected block in each scenario is the
shared oracle — if a behaviour diverges between SDKs, this test fails
on at least one side.

Scenarios live in cognitive-memory-sdk/tests/parity-fixtures/.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cognitive_memory import (
    CognitiveMemory,
    CognitiveMemoryConfig,
    HashEmbeddings,
    InMemoryAdapter,
    MemoryCategory,
)


FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "parity-fixtures"

T0 = datetime(2026, 1, 1)


def _category(value: str) -> MemoryCategory:
    return MemoryCategory(value)


async def _run_scenario(scenario: dict) -> dict:
    """Execute the scenario through the Python SDK's public API and return
    a normalised observable snapshot."""
    adapter = InMemoryAdapter()
    config = CognitiveMemoryConfig(run_maintenance_during_ingestion=False)
    mem = CognitiveMemory(
        config=config,
        adapter=adapter,
        embedder=HashEmbeddings(dimensions=64),
        user_id=scenario["user_id"],
    )

    search_results: list[dict] = []
    for event in scenario["events"]:
        ts = T0 + timedelta(seconds=event["t_seconds"])
        op = event["op"]

        if op == "add":
            await mem.add(
                content=event["content"],
                category=_category(event["category"]),
                importance=event["importance"],
                timestamp=ts,
            )
        elif op == "search":
            response = await mem.search(
                query=event["query"],
                top_k=event.get("limit", 5),
                timestamp=ts,
            )
            search_results.append(
                {
                    "query": event["query"],
                    "top_content": response.results[0].memory.content if response.results else None,
                    "top_category": response.results[0].memory.category.value if response.results else None,
                }
            )
        else:
            raise ValueError(f"Unknown op: {op}")

    all_mems = await adapter.all_active(user_id=scenario["user_id"])
    cat_counts = Counter(m.category.value for m in all_mems)

    return {
        "memory_count": len(all_mems),
        "categories": dict(cat_counts),
        "search_top_content": search_results[0]["top_content"] if search_results else None,
        "search_top_category": search_results[0]["top_category"] if search_results else None,
    }


@pytest.mark.asyncio
async def test_parity_scenario_a():
    """Scenario A: three manual ingests + one search. Both SDKs must
    converge on the same observable end state."""
    with open(FIXTURES / "scenario-a.json") as f:
        scenario = json.load(f)

    snapshot = await _run_scenario(scenario)
    expected = scenario["expected"]

    assert snapshot["memory_count"] == expected["memory_count"]
    assert snapshot["categories"] == expected["categories"]
    assert snapshot["search_top_content"] == expected["search_top_content"]
    assert snapshot["search_top_category"] == expected["search_top_category"]
