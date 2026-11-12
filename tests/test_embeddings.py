import json

import httpx
import pytest

from audiobookshelf_rag_ingester.embeddings import (
    OllamaEmbedder,
    build_embedder,
    truncate,
)
from tests.fakes import make_settings


def _ollama(batch_size: int = 2, max_tokens: int = 0) -> OllamaEmbedder:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        assert request.headers["Authorization"] == "Bearer key"
        body = json.loads(request.content)
        assert body["model"] == "m"
        return httpx.Response(
            200, json={"embeddings": [[float(len(t)), 1.0] for t in body["input"]]}
        )

    return OllamaEmbedder(
        model_name="m",
        ollama_url="http://ollama/",
        api_key="key",
        batch_size=batch_size,
        max_tokens=max_tokens,
        transport=httpx.MockTransport(handler),
    )


def test_ollama_batches_and_dimension() -> None:
    embedder = _ollama(batch_size=2)
    assert embedder.encode(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert embedder.dimension == 2


def test_ollama_truncates() -> None:
    assert _ollama(max_tokens=1).encode(["x" * 10]) == [[4.0, 1.0]]


def test_ollama_bad_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": True})

    embedder = OllamaEmbedder(
        model_name="m", ollama_url="http://o", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ValueError, match="embeddings"):
        embedder.encode(["a"])


def test_truncate() -> None:
    assert truncate(["abcdefghij"], 0) == ["abcdefghij"]
    assert truncate(["abcdefghij", "ab"], 1) == ["abcd", "ab"]


def test_build_embedder() -> None:
    assert isinstance(build_embedder(make_settings()), OllamaEmbedder)
    with pytest.raises(ValueError, match="Unsupported"):
        build_embedder(make_settings(embedding_provider="openai"))
