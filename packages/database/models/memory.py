from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.database.base import Base
from packages.database.models.enums import MemoryState

class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    namespace: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
    )

    content: Mapped[str] = mapped_column(
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(1024),
        nullable=False,
    )

    state: Mapped[MemoryState] = mapped_column(
        Enum(MemoryState),
        default=MemoryState.OBSERVED,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False,
    )

    importance: Mapped[float] = mapped_column(
        Float,
        default=0.5,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    parent_memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id"),
        nullable=True,
    )

    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )