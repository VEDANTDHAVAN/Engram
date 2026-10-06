"""Deterministic TaskFlow project memories and tool data for the demo."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DemoMemory:
    content: str
    memory_type: str
    metadata: dict[str, object] = field(default_factory=dict)


TASKFLOW_MEMORIES: tuple[DemoMemory, ...] = (
    DemoMemory(
        "TaskFlow uses React with TypeScript for its frontend.",
        "semantic", {"topic": "architecture", "state": "active"},
    ),
    DemoMemory(
        "TaskFlow uses FastAPI with Python for its backend.",
        "semantic", {"topic": "architecture", "state": "active"},
    ),
    DemoMemory(
        "TaskFlow uses PostgreSQL as its canonical source of truth.",
        "semantic", {"topic": "architecture", "state": "active"},
    ),
    DemoMemory(
        "Redis is only for caching, background jobs, and transient state; it must not become TaskFlow's canonical database.",
        "semantic", {"topic": "architecture", "state": "active"},
    ),
    DemoMemory(
        "Authentication token policy: refresh tokens must be stored in HttpOnly cookies; access tokens are short-lived.",
        "semantic", {"topic": "security", "state": "active"},
    ),
    DemoMemory(
        "Task status changes must create an immutable status-history record.",
        "procedural", {"topic": "requirements", "state": "active"},
    ),
    DemoMemory(
        "Overdue tasks must appear in the dashboard and trigger a notification.",
        "semantic", {"topic": "requirements", "state": "active"},
    ),
    DemoMemory(
        "The project owner prefers TypeScript on the frontend and Python/FastAPI on the backend.",
        "long_term", {"topic": "preference", "state": "active"},
    ),
    DemoMemory(
        "The TaskFlow task-list API must use cursor-based pagination because the task table exceeded 100,000 records. Cursor pagination keeps deep pages responsive and stable.",
        "semantic", {"topic": "pagination", "state": "active", "decision": "current"},
    ),
    DemoMemory(
        "TaskFlow originally used offset pagination before the task table grew beyond 100,000 records.",
        "episodic", {"topic": "pagination", "state": "superseded", "decision": "historical"},
    ),
    DemoMemory(
        "The migration from offset pagination to cursor-based pagination was approved after slow deep-page queries were observed.",
        "episodic", {"topic": "pagination", "state": "active"},
    ),
    DemoMemory(
        "WebSocket notifications are used for task assignment and overdue-task updates.",
        "semantic", {"topic": "architecture", "state": "active"},
    ),
    DemoMemory(
        "Database queries for task lists must use indexed sort keys compatible with cursor pagination.",
        "procedural", {"topic": "performance", "state": "active"},
    ),
    DemoMemory(
        "A 2025 incident showed that returning refresh tokens in JSON response bodies can expose them to JavaScript; that pattern is prohibited.",
        "episodic", {"topic": "security", "state": "active"},
    ),
    DemoMemory(
        "TaskFlow background jobs are idempotent and use Redis queues only as transient work dispatch.",
        "procedural", {"topic": "architecture", "state": "active"},
    ),
)


def large_codebase_output() -> str:
    """Return deterministic, intentionally large search output without truncation."""
    header = """search_codebase(query=authentication backend changes)
files_scanned: 47
commit: 8d91e4a

=== backend/auth/routes.py ===
POST /auth/refresh now serializes `refresh_token` into a JSON response body for mobile compatibility.
The previous `set_cookie(..., httponly=True)` call was removed.

=== backend/auth/session.py ===
Access token expiry remains 15 minutes.
Refresh token expiry remains 30 days.

=== tests/test_auth_refresh.py ===
Assertion currently expects response.json()["refresh_token"].
"""
    filler = "\n".join(
        f"=== backend/tasks/query_{index:03}.py ===\n"
        "Cursor query uses created_at and id indexes; no authentication behavior changed.\n"
        "test output: passed pagination integration checks."
        for index in range(1, 180)
    )
    return f"{header}\n{filler}\n"


def relevant_tool_evidence() -> str:
    return (
        "search_codebase found a refresh-token regression: "
        "backend/auth/routes.py now returns refresh_token in a JSON body and "
        "removed the HttpOnly cookie call."
    )
