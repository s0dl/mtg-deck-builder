from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import log_extra

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
EMBEDDING_CACHE_VERSION = "v2"
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the supplied text."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, base_url: str, model: str, dimensions: int) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        logger.info(
            "OpenAI embeddings request started",
            extra=log_extra(model=self.model, count=len(texts), dimensions=self.dimensions),
        )
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }
        with httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=15, read=120, write=30, pool=15),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = _post_with_retry(client, "/embeddings", payload)

        data = response.json()["data"]
        logger.info(
            "OpenAI embeddings request completed",
            extra=log_extra(model=self.model, count=len(data)),
        )
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


class FastEmbedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dimensions: int, cache_dir: str) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = cache_dir
        self._model = None

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        logger.info(
            "FastEmbed embeddings request started",
            extra=log_extra(model=self.model_name, count=len(texts), dimensions=self.dimensions),
        )
        embeddings = [embedding.tolist() for embedding in model.embed(texts)]
        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"FastEmbed model {self.model_name} returned {len(embedding)} dimensions; "
                    f"expected EMBEDDING_DIMENSIONS={self.dimensions}."
                )
        logger.info(
            "FastEmbed embeddings request completed",
            extra=log_extra(model=self.model_name, count=len(embeddings)),
        )
        return embeddings

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is required when EMBEDDING_PROVIDER=fastembed. "
                "Install backend dependencies or rebuild the Docker image."
            ) from exc

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model


class CachedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider: EmbeddingProvider, namespace: str, cache_path: str) -> None:
        self.provider = provider
        self.namespace = namespace
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        keys = [_cache_key(self.namespace, text) for text in texts]
        cached = self._get_many(keys)
        results: list[list[float] | None] = [cached.get(key) for key in keys]

        missing_texts: list[str] = []
        missing_indexes: list[int] = []
        seen_missing_keys: set[str] = set()
        for index, (key, text) in enumerate(zip(keys, texts, strict=True)):
            if results[index] is None and key not in seen_missing_keys:
                missing_indexes.append(index)
                missing_texts.append(text)
                seen_missing_keys.add(key)

        if missing_texts:
            logger.info(
                "Embedding cache miss",
                extra=log_extra(namespace=self.namespace, miss_count=len(missing_texts)),
            )
            embedded = self.provider.embed_many(missing_texts)
            rows = []
            for index, vector in zip(missing_indexes, embedded, strict=True):
                key = keys[index]
                results[index] = vector
                rows.append((key, vector))
            self._set_many(rows)

            filled_by_key = {keys[index]: results[index] for index in missing_indexes}
            for index, key in enumerate(keys):
                if results[index] is None and key in filled_by_key:
                    results[index] = filled_by_key[key]
        else:
            logger.info(
                "Embedding cache hit",
                extra=log_extra(namespace=self.namespace, count=len(texts)),
            )

        if any(vector is None for vector in results):
            raise RuntimeError("Embedding cache failed to populate all requested vectors.")
        return [vector for vector in results if vector is not None]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}

        unique_keys = list(dict.fromkeys(keys))
        with self._connect() as connection:
            rows = [
                row
                for key in unique_keys
                for row in connection.execute(
                    "SELECT cache_key, embedding_json FROM embeddings WHERE cache_key = ?",
                    (key,),
                ).fetchall()
            ]
        return {key: json.loads(embedding_json) for key, embedding_json in rows}

    def _set_many(self, rows: list[tuple[str, list[float]]]) -> None:
        if not rows:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO embeddings (cache_key, embedding_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET embedding_json = excluded.embedding_json
                """,
                [(key, json.dumps(vector)) for key, vector in rows],
            )


def _cache_key(namespace: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _post_with_retry(client: httpx.Client, url: str, payload: dict, attempts: int = 4) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post(url, json=payload)
        except httpx.RequestError as exc:
            last_error = exc
            delay = min(2**attempt, 8)
            logger.warning(
                "Retryable OpenAI request error",
                extra=log_extra(error=str(exc), attempt=attempt + 1, delay_seconds=delay),
            )
            time.sleep(delay)
            continue

        if response.status_code not in {429, 500, 502, 503, 504}:
            if response.is_error:
                logger.error(
                    "OpenAI HTTP request failed",
                    extra=log_extra(
                        status_code=response.status_code,
                        response_body=response.text[:1000],
                    ),
                )
            response.raise_for_status()
            return response

        last_error = httpx.HTTPStatusError(
            f"Retryable HTTP status {response.status_code}",
            request=response.request,
            response=response,
        )
        retry_after = response.headers.get("retry-after")
        if retry_after is not None and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = min(2**attempt, 8)
        logger.warning(
            "Retryable OpenAI HTTP response",
            extra=log_extra(status_code=response.status_code, attempt=attempt + 1, delay_seconds=delay),
        )
        time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI request retry loop exited unexpectedly.")


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        provider: EmbeddingProvider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        if settings.embedding_cache_enabled:
            namespace = f"{EMBEDDING_CACHE_VERSION}:openai:{settings.embedding_model}:{settings.embedding_dimensions}"
            return CachedEmbeddingProvider(provider, namespace, settings.embedding_cache_path)
        return provider
    if settings.embedding_provider == "fastembed":
        provider = FastEmbedEmbeddingProvider(
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            cache_dir=settings.embedding_model_cache_path,
        )
        if settings.embedding_cache_enabled:
            namespace = f"{EMBEDDING_CACHE_VERSION}:fastembed:{settings.embedding_model}:{settings.embedding_dimensions}"
            return CachedEmbeddingProvider(provider, namespace, settings.embedding_cache_path)
        return provider
    if settings.embedding_provider != "hash":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
