from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.core.config import get_settings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the supplied text."""


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embeddings for development and tests.

    This is not a semantic model. It gives the RAG pipeline a stable vector
    contract before a production embedding provider is configured.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider != "hash":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
