from enum import Enum

class MemoryType(str, Enum):
    """The five memory classes managed by CAML."""
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryStatus(str, Enum):
    """Lifecycle state of a memory"""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MemoryImportance(str, Enum):
    """Coarse importance classification used by lifecycle policies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"