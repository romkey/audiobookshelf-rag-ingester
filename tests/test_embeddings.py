import json

import httpx
import pytest

from audiobookshelf_rag_ingester.embeddings import (
    OpenAIEmbedder,
    build_embedder,
    truncate,
)
from tests.fakes import make_settings


def _openai(batch_size: int = 2, max_tokens: int = 0) -> OpenAIEmbedder:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer key"
        body = json.loads(request.content)
        assert body["model"] == "m"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(len(t)), 1.0]}
                    for i, t in reversed(list(enumerate(body["input"])))
                ]
            },
        )

    return OpenAIEmbedder(
        model_name="m",
        base_url="http://openai/v1/",
        api_key="key",
        batch_size=batch_size,
        max_tokens=max_tokens,
        transport=httpx.MockTransport(handler),
    )


def test_openai_batches_and_dimension() -> None:
    embedder = _openai(batch_size=2)
    assert embedder.encode(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert embedder.dimension == 2


def test_openai_truncates() -> None:
    assert _openai(max_tokens=1).encode(["x" * 10]) == [[4.0, 1.0]]


def test_openai_bad_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": True})

    embedder = OpenAIEmbedder(
        model_name="m", base_url="http://o/v1", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ValueError, match="data"):
        embedder.encode(["a"])


def test_openai_api_key_optional() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAIEmbedder(
        model_name="m", base_url="http://o/v1", transport=httpx.MockTransport(handler)
    )
    assert embedder.encode(["a"]) == [[1.0]]


def test_prefix_by_kind_counts_toward_limit() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.extend(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAIEmbedder(
        model_name="m",
        query_prefix="q: ",
        document_prefix="d: ",
        max_tokens=2,  # 8 chars
        transport=httpx.MockTransport(handler),
    )
    embedder.encode(["abcdefghij"])
    embedder.encode(["abcdefghij"], "query")
    assert seen == ["d: abcde", "q: abcde"]


def test_truncate() -> None:
    assert truncate(["abcdefghij"], 0) == ["abcdefghij"]
    assert truncate(["abcdefghij", "ab"], 1) == ["abcd", "ab"]
    assert truncate(["ab"], 0, "p ") == ["p ab"]
    assert truncate(["abcdefghij"], 1, "abcdef") == ["abcdef"]


def test_build_embedder() -> None:
    assert isinstance(build_embedder(make_settings()), OpenAIEmbedder)
    with pytest.raises(ValueError, match="Unsupported"):
        build_embedder(make_settings(embedding_provider="bogus"))
