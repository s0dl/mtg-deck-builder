from dataclasses import dataclass
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import func
from sqlalchemy.orm import Session

from app.rag.documents import RagDocument
from app.rag.embeddings import EmbeddingProvider


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

    def upsert_many(self, documents: list[RagDocumentInput], commit_every: int = 500) -> int:
        count = 0
        for document in documents:
            self.upsert_document(document)
            count += 1
            if count % commit_every == 0:
                self.session.commit()
        self.session.commit()
        return count
