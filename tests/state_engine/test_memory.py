from packages.state_engine import Memory, MemoryImportance, MemoryStatus, MemoryType

def test_memory_defaults():
    memory = Memory(
        namespace="tenant-1",
        content="Customer prefers PostgreSQL.",
        memory_type=MemoryType.LONG_TERM,
    )

    assert memory.memory_type == MemoryType.LONG_TERM
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.importance == MemoryImportance.MEDIUM
    assert 0.0 <= memory.confidence <= 1.0
    assert memory.created_at is not None

def test_memory_can_be_superseded():
    old_memory = Memory(
        namespace="tenant-1",
        content="Customer uses PostgreSQL.",
        memory_type=MemoryType.LONG_TERM,
    )

    new_memory = Memory(
        namespace="tenant-1",
        content="Customer migrated to Supabase.",
        memory_type=MemoryType.LONG_TERM,
    )

    old_memory.status = MemoryStatus.SUPERSEDED
    old_memory.superseded_by = new_memory.id

    assert old_memory.status == MemoryStatus.SUPERSEDED
    assert old_memory.superseded_by == new_memory.id