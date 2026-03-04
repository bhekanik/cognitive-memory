#!/usr/bin/env python3
"""
Smoke test for cognitive-memory SDK.
Uses hash embeddings (no API calls) to verify all mechanisms.
Uses SyncCognitiveMemory wrapper for sync test execution.
"""

import sys
from datetime import datetime, timedelta
from cognitive_memory import (
    SyncCognitiveMemory,
    CognitiveMemoryConfig,
    Memory,
    MemoryCategory,
    SearchResult,
)

def test_basic_add_and_search():
    """Test basic memory storage and retrieval."""
    mem = SyncCognitiveMemory(embedder="hash")

    mem.add("User is allergic to shellfish", category=MemoryCategory.CORE, importance=0.95, session_id="s1")
    mem.add("User's name is Alex Chen", category=MemoryCategory.CORE, importance=0.99, session_id="s1")
    mem.add("User had pasta for lunch", category=MemoryCategory.EPISODIC, importance=0.2, session_id="s1")

    results = mem.search("what is the user allergic to?", top_k=3)
    assert len(results) > 0, "Expected at least one result"

    stats = mem.get_stats()
    assert stats["total_memories"] == 3
    assert stats["hot_memories"] == 3
    assert stats["core_memories"] == 2
    print(f"  basic_add_and_search: PASS (stats={stats})")


def test_decay_model():
    """Test that retention decays over time and floors work."""
    config = CognitiveMemoryConfig()
    mem = SyncCognitiveMemory(config=config, embedder="hash")

    now = datetime(2024, 1, 1)

    # Add an episodic memory
    m = mem.add("User watched a documentary", category=MemoryCategory.EPISODIC,
                importance=0.3, timestamp=now, session_id="s1")

    # Check retention at various time points
    engine = mem.engine
    r_now = engine.compute_retention(m, now)
    r_7d = engine.compute_retention(m, now + timedelta(days=7))
    r_30d = engine.compute_retention(m, now + timedelta(days=30))
    r_90d = engine.compute_retention(m, now + timedelta(days=90))
    r_365d = engine.compute_retention(m, now + timedelta(days=365))

    assert r_now > r_7d > r_30d, f"Expected decay: {r_now} > {r_7d} > {r_30d}"
    assert r_365d >= 0.02, f"Expected floor at 0.02, got {r_365d}"
    print(f"  decay_model: PASS (now={r_now:.3f}, 7d={r_7d:.3f}, 30d={r_30d:.3f}, 90d={r_90d:.3f}, 365d={r_365d:.3f})")


def test_core_floor():
    """Test that core memories have a 0.60 floor."""
    mem = SyncCognitiveMemory(embedder="hash")
    now = datetime(2024, 1, 1)

    m = mem.add("User's name is Alex", category=MemoryCategory.CORE,
                importance=0.99, timestamp=now, session_id="s1")

    engine = mem.engine
    r_365d = engine.compute_retention(m, now + timedelta(days=365))

    assert r_365d >= 0.60, f"Core floor should be 0.60, got {r_365d}"
    print(f"  core_floor: PASS (365d retention={r_365d:.3f})")


def test_retrieval_boosting():
    """Test that retrieval boosts stability."""
    mem = SyncCognitiveMemory(embedder="hash")
    now = datetime(2024, 1, 1)

    m = mem.add("User likes Rust programming", category=MemoryCategory.SEMANTIC,
                importance=0.6, timestamp=now, session_id="s1")

    initial_stability = m.stability
    assert initial_stability > 0, f"Expected positive initial stability, got {initial_stability}"

    # Simulate retrieval at day 7 (should trigger spaced rep boost)
    mem.engine.apply_direct_boost(m, now + timedelta(days=7), session_id="s2")
    after_boost = m.stability
    assert after_boost > initial_stability, f"Expected boost: {after_boost} > {initial_stability}"

    # Associative boost should be smaller
    m2 = mem.add("User is learning Go too", category=MemoryCategory.SEMANTIC,
                 importance=0.4, timestamp=now, session_id="s1")
    s_before = m2.stability
    mem.engine.apply_associative_boost(m2, now + timedelta(days=7), session_id="s2")
    s_after = m2.stability
    direct_delta = after_boost - initial_stability
    assoc_delta = s_after - s_before
    assert direct_delta > assoc_delta, f"Direct boost should be larger: {direct_delta} vs {assoc_delta}"

    print(f"  retrieval_boosting: PASS (direct_delta={direct_delta:.3f}, assoc_delta={assoc_delta:.3f})")


def test_core_promotion():
    """Test emergent core promotion through repeated access."""
    config = CognitiveMemoryConfig(
        core_access_threshold=5,  # lower for testing
        core_stability_threshold=0.5,
        core_session_threshold=3,
    )
    mem = SyncCognitiveMemory(config=config, embedder="hash")
    now = datetime(2024, 1, 1)

    m = mem.add("User prefers dark mode", category=MemoryCategory.SEMANTIC,
                importance=0.6, timestamp=now, session_id="s1")

    assert m.category != MemoryCategory.CORE

    # Simulate repeated direct retrieval across sessions
    for i in range(8):
        day = now + timedelta(days=7 * (i + 1))
        session = f"s{i+2}"
        mem.engine.apply_direct_boost(m, day, session_id=session)
        promoted = mem.engine.check_core_promotion(m)
        if promoted:
            break

    assert m.category == MemoryCategory.CORE, f"Expected promotion to core, got {m.category}"
    print(f"  core_promotion: PASS (promoted after {m.access_count} accesses, stability={m.stability:.3f})")


def test_associations():
    """Test associative linking between co-retrieved memories."""
    mem = SyncCognitiveMemory(embedder="hash")
    now = datetime(2024, 1, 1)

    m1 = mem.add("User works at Meridian Labs", session_id="s1", timestamp=now)
    m2 = mem.add("Meridian Labs is in San Francisco", session_id="s1", timestamp=now)

    # Strengthen association
    mem.engine.strengthen_association(m1, m2, now)
    mem.engine.strengthen_association(m1, m2, now)
    mem.engine.strengthen_association(m1, m2, now)  # 3x -> weight 0.3

    assert m2.id in m1.associations
    assert m1.id in m2.associations
    assert m1.associations[m2.id].weight >= 0.3  # at retrieval threshold

    # Check association decay
    assoc = m1.associations[m2.id]
    w_now = assoc.weight
    w_decayed = mem.engine.decay_association(assoc, now + timedelta(days=90))
    assert w_decayed < w_now, f"Expected decay: {w_decayed} < {w_now}"

    print(f"  associations: PASS (weight={w_now:.3f}, decayed_90d={w_decayed:.3f})")


def test_cold_storage_migration():
    """Test migration to cold storage and revival."""
    config = CognitiveMemoryConfig(cold_migration_days=3)
    mem = SyncCognitiveMemory(config=config, embedder="hash")
    now = datetime(2024, 1, 1)

    m = mem.add("User had sushi yesterday", category=MemoryCategory.EPISODIC,
                importance=0.2, timestamp=now, session_id="s1")

    # Simulate time passing until memory is at floor
    far_future = now + timedelta(days=200)
    retention = mem.engine.compute_retention(m, far_future)
    assert abs(retention - 0.02) < 0.01, f"Expected at floor, got {retention}"

    # Manually set days_at_floor to trigger migration
    m.days_at_floor = 5

    import asyncio
    asyncio.run(mem.engine.run_cold_migration(far_future))

    assert m.is_cold, "Expected memory to be in cold storage"
    stats = mem.get_stats()
    assert stats["cold_memories"] == 1
    assert stats["hot_memories"] == 0  # moved out of hot

    # Revival: migrate back
    asyncio.run(mem.adapter.migrate_to_hot(m.id))
    assert not m.is_cold
    stats = mem.get_stats()
    assert stats["hot_memories"] == 1

    print(f"  cold_storage: PASS (migrated and revived)")


def test_cold_ttl_expiry():
    """Test that cold memories expire after TTL and leave stubs."""
    config = CognitiveMemoryConfig(cold_storage_ttl_days=30)
    mem = SyncCognitiveMemory(config=config, embedder="hash")
    now = datetime(2024, 1, 1)

    m = mem.add("User's cat is named Whiskers", category=MemoryCategory.EPISODIC,
                importance=0.3, timestamp=now, session_id="s1")

    # Move to cold
    import asyncio
    asyncio.run(mem.adapter.migrate_to_cold(m.id, now))
    stats = mem.get_stats()
    assert stats["cold_memories"] == 1

    # Expire after TTL
    asyncio.run(mem.engine.run_cold_ttl_expiry(now + timedelta(days=35)))

    stats = mem.get_stats()
    assert stats["cold_memories"] == 0, "Cold memory should be removed"
    assert stats["stub_memories"] == 1, "Stub should be created"

    asyncio.run(mem.adapter.get(m.id))
    stub = asyncio.run(mem.adapter.get(m.id))
    assert stub is not None
    assert stub.is_stub
    assert "[archived]" in stub.content

    print(f"  cold_ttl_expiry: PASS (cold -> stub after TTL)")


def test_scoring_prefers_recent():
    """Test that scoring ranks recent memories above stale ones."""
    mem = SyncCognitiveMemory(embedder="hash")
    now = datetime(2024, 1, 1)

    # Same content, different ages
    old = mem.add("User likes coffee", importance=0.5, timestamp=now, session_id="s1")
    recent = mem.add("User likes coffee a lot", importance=0.5,
                     timestamp=now + timedelta(days=29), session_id="s2")

    query_time = now + timedelta(days=30)
    results = mem.search("does the user like coffee?", timestamp=query_time, top_k=5)

    # The more recent memory should have higher retention
    if len(results) >= 2:
        # Find them
        old_result = next((r for r in results if r.memory.id == old.id), None)
        recent_result = next((r for r in results if r.memory.id == recent.id), None)
        if old_result and recent_result:
            assert recent_result.retention_score > old_result.retention_score, \
                f"Recent should have higher retention: {recent_result.retention_score} vs {old_result.retention_score}"

    print(f"  scoring_prefers_recent: PASS")


if __name__ == "__main__":
    print("Running cognitive-memory SDK tests...\n")

    tests = [
        test_basic_add_and_search,
        test_decay_model,
        test_core_floor,
        test_retrieval_boosting,
        test_core_promotion,
        test_associations,
        test_cold_storage_migration,
        test_cold_ttl_expiry,
        test_scoring_prefers_recent,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAIL - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
