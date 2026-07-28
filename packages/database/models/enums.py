from enum import Enum

class MemoryState(str, Enum):
    OBSERVED = "observed"
    VERIFIED = "verified"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"