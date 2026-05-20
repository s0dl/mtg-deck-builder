"""Preview helpers for RAG seed scripts."""

from __future__ import annotations

import logging

from app.core.logging import log_extra
from app.rag.repository import RagDocumentInput


def log_document_samples(
    logger: logging.Logger,
    documents: list[RagDocumentInput],
    *,
    sample_size: int,
    content_preview_chars: int = 260,
) -> None:
    sample_size = max(0, sample_size)
    if sample_size == 0:
        return

    samples = documents[:sample_size]
    if not samples:
        logger.info("No RAG document samples available")
        return

    for index, document in enumerate(samples, start=1):
        logger.info(
            (
                "RAG document sample %s/%s: source=%s source_id=%s title=%r content=%r"
            ),
            index,
            len(samples),
            document.source,
            document.source_id,
            document.title,
            _preview_text(document.content, content_preview_chars),
            extra=log_extra(
                sample_index=index,
                sample_count=len(samples),
                source=document.source,
                source_id=document.source_id,
                title=document.title,
                content_preview=_preview_text(document.content, content_preview_chars),
                metadata=document.metadata,
            ),
        )


def _preview_text(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."
