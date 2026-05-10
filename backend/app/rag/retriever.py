from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.rag.documents import RagDocument
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    content: str
    source: str
    score: float | None = None


class RagRetriever:
    def __init__(self, session: Session, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.session = session
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def search(self, query: str, limit: int = 5) -> list[RetrievedDocument]:
        if not query:
            return []

        embedding = self.embedding_provider.embed(query)
        statement = (
            select(
                RagDocument,
                RagDocument.embedding.cosine_distance(embedding).label("distance"),
            )
            .where(RagDocument.embedding.is_not(None))
            .order_by("distance")
            .limit(limit)
        )
        rows = self.session.execute(statement).all()
        if not rows:
            return self.search_text(query=query, limit=limit)

        return [
            RetrievedDocument(
                title=document.title,
                content=document.content,
                source=document.source,
                score=1.0 - float(distance),
            )
            for document, distance in rows
        ]

    def search_text(self, query: str, limit: int = 5) -> list[RetrievedDocument]:
        if not query:
            return []

        terms = [term for term in query.split() if len(term) > 2]
        if not terms:
            return []

        filters = [
            or_(
                RagDocument.title.ilike(f"%{term}%"),
                RagDocument.content.ilike(f"%{term}%"),
            )
            for term in terms[:6]
        ]
        statement = select(RagDocument).where(or_(*filters)).limit(limit)
        documents = self.session.scalars(statement).all()

        return [
            RetrievedDocument(title=doc.title, content=doc.content, source=doc.source)
            for doc in documents
        ]
