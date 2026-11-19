from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import httpx
import pytest

from audiobookshelf_rag_ingester import embeddings
from audiobookshelf_rag_ingester.config import Settings
from audiobookshelf_rag_ingester.embeddings import (
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    _estimate_tokens,
    _load_sentence_transformer,
    _Prefixes,
    _request_batches,
    _truncate,
    build_embedder,
)

SettingsFactory = Callable[..., Settings]  # the make_settings fixture


class _FakeResponse:
    # Any: a JSON body.
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeClient:
    """Stands in for httpx.Client; records the request and returns `body`."""

    headers: Mapping[str, str] | None = None
    base_url: str | None = None
    # Any: JSON bodies.
    last_json: ClassVar[dict[str, Any]] = {}
    # Deliberately out of order, to check sorting by index.
    body: ClassVar[dict[str, Any]] = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        ],
    }

    def __init__(self, base_url: str, headers: Mapping[str, str] | None, timeout: int) -> None:
        type(self).headers = headers
        type(self).base_url = base_url

    # Any: a JSON request body.
    def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
        assert path == "/embeddings"
        assert json["model"] == "text-embedding-3-small"
        type(self).last_json = json
        return _FakeResponse(self.body)


class _Vector(list[float]):
    """Like a numpy row: has tolist()."""

    def tolist(self) -> list[float]:
        return list(self)


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.seen: list[str] = []

    def encode(self, chunks: list[str], show_progress_bar: bool = False) -> list[Sequence[float]]:
        self.seen = chunks
        # Mix numpy-like rows (with tolist) and plain tuples.
        return [_Vector([1.0, 2.0]) if i % 2 == 0 else (3.0, 4.0) for i in range(len(chunks))]


def _openai(
    api_key: str | None = None,
    dimensions: int | None = None,
    base_url: str = "https://api.openai.com/v1",
) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        model_name="text-embedding-3-small",
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=30,
        dimensions=dimensions,
    )


def test_openai_embedder_calls_embeddings_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    vectors = _openai(base_url="https://api.openai.com/v1/").encode(["a", "b"])
    assert _FakeClient.base_url == "https://api.openai.com/v1"
    assert _FakeClient.headers is None
    assert _FakeClient.last_json["input"] == ["a", "b"]
    assert _FakeClient.last_json["encoding_format"] == "float"
    assert "dimensions" not in _FakeClient.last_json
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_openai_embedder_sends_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _openai(dimensions=256).encode(["a", "b"])
    assert _FakeClient.last_json["dimensions"] == 256


def test_openai_embedder_uses_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _openai(api_key="secret")
    assert _FakeClient.headers == {"Authorization": "Bearer secret"}


@pytest.mark.parametrize(
    "body",
    [
        {"error": "model not found"},
        {"data": [{"index": 0, "embedding": [0.1]}]},  # one vector for two inputs
    ],
)
def test_openai_embedder_rejects_bad_response(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    class _BadClient(_FakeClient):
        pass

    _BadClient.body = body
    monkeypatch.setattr(httpx, "Client", _BadClient)
    with pytest.raises(ValueError, match="missing 'data' or wrong length"):
        _openai().encode(["a", "b"])


def test_openai_embedder_empty_input_skips_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoPostClient(_FakeClient):
        # Any: a JSON request body.
        def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
            raise AssertionError("no request expected")

    monkeypatch.setattr(httpx, "Client", _NoPostClient)
    assert _openai().encode([]) == []


def test_openai_embedder_splits_large_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _EchoClient(_FakeClient):
        # Any: a JSON request body.
        def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
            n = len(json["input"])
            calls.append(n)
            return _FakeResponse(
                {"data": [{"index": i, "embedding": [float(i)]} for i in range(n)]}
            )

    monkeypatch.setattr(httpx, "Client", _EchoClient)
    monkeypatch.setattr(embeddings, "_OPENAI_MAX_INPUTS_PER_REQUEST", 3)
    assert len(_openai().encode(["t"] * 7)) == 7
    assert calls == [3, 3, 1]


def test_request_batches_respects_input_cap() -> None:
    batches = _request_batches(["a"] * 5, max_inputs=2, max_tokens=10_000)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_request_batches_respects_token_cap() -> None:
    text = "x" * 40  # ~11 estimated tokens each
    batches = _request_batches([text] * 4, max_inputs=100, max_tokens=25)
    assert [len(b) for b in batches] == [2, 2]


def test_estimate_tokens_counts_cjk_per_character() -> None:
    assert _estimate_tokens("x" * 400) == 101
    assert _estimate_tokens("漢" * 400) == 401


def test_request_batches_splits_cjk_text() -> None:
    cjk = "漢" * 100  # ~101 estimated tokens each, not ~26
    batches = _request_batches([cjk] * 4, max_inputs=100, max_tokens=250)
    assert [len(b) for b in batches] == [2, 2]


def test_build_embedder_uses_openai_provider(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    embedder = build_embedder(make_settings(embedding_dimensions=256))
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.max_tokens == 8192
    assert embedder.dimensions == 256
    assert _FakeClient.base_url == "http://localhost:11434/v1"


def test_build_embedder_uses_st_cache_directory(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda: _FakeSentenceTransformer)
    settings = make_settings(embedding_provider="sentence_transformers", hf_cache_dir="/hf-cache")
    embedder = build_embedder(settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder._model.cache_folder == "/hf-cache"


def test_build_embedder_unsupported_provider(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER 'ollama'"):
        build_embedder(make_settings(embedding_provider="ollama"))


def test_sentence_transformer_encode_and_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda: _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder(model_name="m", max_tokens=2)
    assert embedder.encode(["x" * 100, "short"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert embedder._model.seen == ["x" * 8, "short"]


def test_load_sentence_transformer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    # Any: __import__'s signature.
    def _import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ImportError, match="pip install audiobookshelf-rag-ingester"):
        _load_sentence_transformer()


def test_load_sentence_transformer_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("sentence_transformers")
    setattr(fake, "SentenceTransformer", _FakeSentenceTransformer)  # noqa: B010 -- module has no such attribute for mypy
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    assert _load_sentence_transformer() is _FakeSentenceTransformer


def test_truncate_shortens_long_texts() -> None:
    result = _truncate(["hello", "x" * 100], max_tokens=10)
    assert result[0] == "hello"
    assert len(result[1]) == 10 * 4  # _CHARS_PER_TOKEN = 4


def test_truncate_noop_when_disabled() -> None:
    long_text = "x" * 100
    assert _truncate([long_text], max_tokens=0) == [long_text]


def test_truncate_prepends_prefix_and_counts_it() -> None:
    result = _truncate(["x" * 100], max_tokens=10, prefix="doc: ")
    assert result[0].startswith("doc: ")
    assert len(result[0]) == 10 * 4  # prefix + text fits the budget


def test_truncate_prefix_without_limit() -> None:
    assert _truncate(["abc"], max_tokens=0, prefix="q: ") == ["q: abc"]


def test_openai_embedder_applies_query_and_document_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    embedder = _openai()
    embedder.prefixes = _Prefixes(
        query_prefix="search_query: ", document_prefix="search_document: "
    )
    embedder.encode(["a", "b"])
    assert _FakeClient.last_json["input"] == ["search_document: a", "search_document: b"]
    embedder.encode(["a", "b"], kind="query")
    assert _FakeClient.last_json["input"] == ["search_query: a", "search_query: b"]


def test_openai_embedder_no_prefix_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _openai().encode(["a", "b"], kind="query")
    assert _FakeClient.last_json["input"] == ["a", "b"]


def test_embedder_rejects_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    with pytest.raises(ValueError, match="Unknown embedding kind"):
        _openai().encode(["a"], kind="passage")  # type: ignore[arg-type]


def test_build_embedder_passes_prefixes_from_settings(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    settings = make_settings(
        embedding_query_prefix="search_query: ", embedding_document_prefix="search_document: "
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.prefixes.for_kind("query") == "search_query: "
    assert embedder.prefixes.for_kind("document") == "search_document: "
