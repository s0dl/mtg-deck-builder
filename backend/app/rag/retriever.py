from dataclasses import dataclass
import logging
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.logging import log_extra
from app.rag.documents import RagDocument
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)

TOKEN_REPLACEMENTS = {
    "w": "white",
    "u": "blue",
    "b": "black",
    "r": "red",
    "g": "green",
}


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    content: str
    source: str
    metadata: dict[str, Any]
    score: float | None = None


class RagRetriever:
    def __init__(self, session: Session, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.session = session
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def search(self, query: str, limit: int = 5, source: str | None = None) -> list[RetrievedDocument]:
        if not query:
            return []

        logger.info(
            "RAG vector search started",
            extra=log_extra(source=source, limit=limit, query_length=len(query)),
        )
        try:
            embedding = self.embedding_provider.embed(query)
        except Exception as exc:
            logger.warning(
                "RAG vector search failed; falling back to text search",
                extra=log_extra(source=source, error=str(exc)),
            )
            return self.search_text(query=query, limit=limit, source=source)
        statement = (
            select(
                RagDocument,
                RagDocument.embedding.cosine_distance(embedding).label("distance"),
            )
            .where(RagDocument.embedding.is_not(None))
        )
        if source is not None:
            statement = statement.where(RagDocument.source == source)
        statement = statement.order_by("distance").limit(limit)
        rows = self.session.execute(statement).all()
        if not rows:
            logger.info(
                "RAG vector search returned no rows; falling back to text search",
                extra=log_extra(source=source, limit=limit),
            )
            return self.search_text(query=query, limit=limit, source=source)

        results = [
            RetrievedDocument(
                title=document.title,
                content=document.content,
                source=document.source,
                metadata=document.metadata_,
                score=1.0 - float(distance),
            )
            for document, distance in rows
        ]
        logger.info(
            "RAG vector search completed",
            extra=log_extra(source=source, result_count=len(results)),
        )
        return results

    def search_by_metadata_prefix(
        self,
        source: str,
        metadata_key: str,
        prefixes: list[str],
        limit: int = 10,
    ) -> list[RetrievedDocument]:
        if not prefixes:
            return []

        logger.info(
            "RAG metadata search started",
            extra=log_extra(source=source, metadata_key=metadata_key, prefixes=prefixes, limit=limit),
        )
        filters = [
            RagDocument.metadata_[metadata_key].as_string().like(f"{prefix}%")
            for prefix in prefixes
        ]
        statement = (
            select(RagDocument)
            .where(RagDocument.source == source)
            .where(or_(*filters))
            .limit(limit)
        )
        documents = self.session.scalars(statement).all()

        results = [
            RetrievedDocument(
                title=doc.title,
                content=doc.content,
                source=doc.source,
                metadata=doc.metadata_,
            )
            for doc in documents
        ]
        logger.info(
            "RAG metadata search completed",
            extra=log_extra(source=source, result_count=len(results)),
        )
        return results

    def search_text(
        self,
        query: str,
        limit: int = 5,
        source: str | None = None,
        metadata_filters: dict[str, list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        if not query:
            return []

        terms = [term for term in query.split() if len(term) > 2]
        if not terms:
            return []

        logger.info(
            "RAG text search started",
            extra=log_extra(
                source=source,
                limit=limit,
                terms=terms[:6],
                metadata_filters=metadata_filters or {},
            ),
        )
        filters = [
            or_(
                RagDocument.title.ilike(f"%{term}%"),
                RagDocument.content.ilike(f"%{term}%"),
            )
            for term in terms[:6]
        ]
        candidate_limit = max(limit * 20, 100)
        statement = select(RagDocument).where(or_(*filters))
        if source is not None:
            statement = statement.where(RagDocument.source == source)
        statement = _apply_metadata_filters(statement, metadata_filters)
        statement = statement.limit(candidate_limit)
        documents = self.session.scalars(statement).all()

        results = [
            RetrievedDocument(
                title=doc.title,
                content=doc.content,
                source=doc.source,
                metadata=doc.metadata_,
                score=_score_text_match(doc.title, doc.content, terms, doc.metadata_),
            )
            for doc in documents
        ]
        results.sort(key=lambda item: item.score or 0, reverse=True)
        results = results[:limit]
        logger.info(
            "RAG text search completed",
            extra=log_extra(source=source, result_count=len(results)),
        )
        return results


def _apply_metadata_filters(statement, metadata_filters: dict[str, list[str]] | None):
    if not metadata_filters:
        return statement

    for key, values in metadata_filters.items():
        normalized_values = [value.lower() for value in values]
        non_empty_values = [value for value in normalized_values if value]
        filters = []
        if "" in normalized_values:
            filters.append(RagDocument.metadata_[key].as_string().is_(None))
            filters.append(RagDocument.metadata_[key].as_string() == "")
        if non_empty_values:
            filters.append(func.lower(RagDocument.metadata_[key].as_string()).in_(non_empty_values))
        if filters:
            statement = statement.where(or_(*filters))
    return statement


def _score_text_match(title: str, content: str, terms: list[str], metadata: dict[str, Any]) -> float:
    title_lower = title.lower()
    content_lower = content.lower()
    metadata_text = " ".join(str(value).lower() for value in metadata.values() if value)
    searchable = " ".join([title_lower, content_lower, metadata_text])
    normalized_terms = [TOKEN_REPLACEMENTS.get(term.lower(), term.lower()) for term in terms]
    phrase = " ".join(normalized_terms)
    score = 0.0
    if phrase:
        if phrase in title_lower:
            score += 25.0
        if phrase in metadata_text:
            score += 15.0
        if phrase in content_lower:
            score += 8.0
    matched_terms = 0
    for normalized in normalized_terms:
        if normalized in title_lower:
            score += 4.0
            matched_terms += 1
            continue
        if normalized in metadata_text:
            score += 3.0
            matched_terms += 1
            continue
        if normalized in content_lower:
            score += 1.0
            matched_terms += 1
    if normalized_terms and matched_terms == len(normalized_terms):
        score += 10.0
    elif matched_terms:
        score += matched_terms / len(normalized_terms)
    if phrase and phrase not in searchable and len(normalized_terms) > 1 and matched_terms < len(normalized_terms):
        score -= 2.0
    return score
