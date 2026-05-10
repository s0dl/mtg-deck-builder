from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime, Uuid

from app.core.database import Base


class RagDocument(Base):
    __tablename__ = "rag_documents"
    __table_args__ = (UniqueConstraint("source", "source_id", name="rag_documents_source_source_id_key"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, server_default=func.gen_random_uuid())
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
