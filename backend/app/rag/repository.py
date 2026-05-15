from dataclasses import dataclass
import logging
import time
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import func
from sqlalchemy.orm import Session

from app.core.logging import log_extra
from app.rag.documents import RagDocument
from app.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagDocumentInput:
    source: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]


class RagRepository:
    def __init__(self, session: Session, embedding_provider: EmbeddingProvider) -> None:
        self.session = session
        self.embedding_provider = embedding_provider

    def upsert_document(self, document: RagDocumentInput) -> None:
        embedding = self.embedding_provider.embed(document.content)
        statement = insert(RagDocument).values(
            source=document.source,
            source_id=document.source_id,
            title=document.title,
            content=document.content,
            metadata_=document.metadata,
            embedding=embedding,
        )
        update_fields = {
            "title": statement.excluded.title,
            "content": statement.excluded.content,
            "metadata": statement.excluded["metadata"],
            "embedding": statement.excluded.embedding,
            "updated_at": func.now(),
        }
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=["source", "source_id"],
                set_=update_fields,
            )
        )

    def upsert_many(
        self,
        documents: list[RagDocumentInput],
        batch_size: int = 100,
        batch_delay_seconds: float = 0.0,
    ) -> int:
        count = 0
        batch_size = max(1, batch_size)
        logger.info(
            "RAG upsert started",
            extra=log_extra(
                total_documents=len(documents),
                batch_size=batch_size,
                batch_delay_seconds=batch_delay_seconds,
            ),
        )
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            batch_number = (start // batch_size) + 1
            remaining_after_batch = max(len(documents) - (start + len(batch)), 0)
            logger.info(
                "RAG upsert batch started",
                extra=log_extra(
                    batch_number=batch_number,
                    total_batches=total_batches,
                    batch_start=start,
                    batch_size=len(batch),
                    imported=count,
                    remaining_documents=len(documents) - count,
                ),
            )
            logger.info(
                "RAG embedding batch started",
                extra=log_extra(
                    batch_number=batch_number,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    remaining_documents=len(documents) - count,
                ),
            )
            embeddings = self.embedding_provider.embed_many([document.content for document in batch])
            logger.info(
                "RAG embedding batch completed",
                extra=log_extra(
                    batch_number=batch_number,
                    total_batches=total_batches,
                    embedded_documents=len(batch),
                    remaining_documents=remaining_after_batch,
                ),
            )
            for document, embedding in zip(batch, embeddings, strict=True):
                self._upsert_document_with_embedding(document, embedding)
                count += 1
            self.session.commit()
            logger.info(
                "RAG upsert batch committed",
                extra=log_extra(imported=count, total_documents=len(documents)),
            )
            if batch_delay_seconds > 0 and start + batch_size < len(documents):
                logger.info(
                    "RAG upsert batch delay started",
                    extra=log_extra(delay_seconds=batch_delay_seconds),
                )
                time.sleep(batch_delay_seconds)
        self.session.commit()
        logger.info("RAG upsert completed", extra=log_extra(imported=count))
        return count

    def _upsert_document_with_embedding(
        self,
        document: RagDocumentInput,
        embedding: list[float],
    ) -> None:
        statement = insert(RagDocument).values(
            source=document.source,
            source_id=document.source_id,
            title=document.title,
            content=document.content,
            metadata_=document.metadata,
            embedding=embedding,
        )
        update_fields = {
            "title": statement.excluded.title,
            "content": statement.excluded.content,
            "metadata": statement.excluded["metadata"],
            "embedding": statement.excluded.embedding,
            "updated_at": func.now(),
        }
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=["source", "source_id"],
                set_=update_fields,
            )
        )
