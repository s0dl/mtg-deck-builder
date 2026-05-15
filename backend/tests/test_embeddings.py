from pathlib import Path

import httpx

from app.rag.embeddings import CachedEmbeddingProvider, EmbeddingProvider, FastEmbedEmbeddingProvider, _post_with_retry


class CountingEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


def test_cached_embedding_provider_reuses_cached_text(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.sqlite3"
    provider = CountingEmbeddingProvider()
    cached = CachedEmbeddingProvider(provider, "test:model:1", str(cache_path))

    assert cached.embed("Lightning Bolt") == [14.0]
    assert cached.embed("Lightning Bolt") == [14.0]

    assert provider.calls == [["Lightning Bolt"]]


def test_cached_embedding_provider_dedupes_missing_texts(tmp_path: Path) -> None:
    provider = CountingEmbeddingProvider()
    cached = CachedEmbeddingProvider(provider, "test:model:1", str(tmp_path / "embeddings.sqlite3"))

    assert cached.embed_many(["Island", "Island", "Mountain"]) == [[6.0], [6.0], [8.0]]

    assert provider.calls == [["Island", "Mountain"]]


class FakeFastEmbedModel:
    def embed(self, texts: list[str]):
        return [FakeVector([float(len(text)), 1.0]) for text in texts]


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


def test_fastembed_embedding_provider_uses_local_model() -> None:
    provider = FastEmbedEmbeddingProvider(
        model_name="test-model",
        dimensions=2,
        cache_dir="unused",
    )
    provider._model = FakeFastEmbedModel()

    assert provider.embed_many(["Bolt", "Island"]) == [[4.0, 1.0], [6.0, 1.0]]


def test_fastembed_embedding_provider_validates_dimensions() -> None:
    provider = FastEmbedEmbeddingProvider(
        model_name="test-model",
        dimensions=3,
        cache_dir="unused",
    )
    provider._model = FakeFastEmbedModel()

    try:
        provider.embed("Bolt")
    except ValueError as exc:
        assert "expected EMBEDDING_DIMENSIONS=3" in str(exc)
    else:
        raise AssertionError("Expected FastEmbed dimension mismatch to raise ValueError.")


def test_post_with_retry_retries_request_errors() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1") as client:
        response = _post_with_retry(client, "/embeddings", {"input": ["Lightning Bolt"]})

    assert response.status_code == 200
    assert attempts == 2
