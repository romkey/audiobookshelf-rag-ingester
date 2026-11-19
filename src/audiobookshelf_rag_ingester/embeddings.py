from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from audiobookshelf_rag_ingester.config import Settings

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4

# OpenAI rejects /embeddings requests with more than 2048 inputs or more than
# 300k tokens in total. Stay well under both.
_OPENAI_MAX_INPUTS_PER_REQUEST = 2048
_OPENAI_MAX_TOKENS_PER_REQUEST = 250_000


EmbedKind = Literal["query", "document"]


class Embedder(Protocol):
    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]: ...


def _truncate(texts: list[str], max_tokens: int, prefix: str = "") -> list[str]:
    """Prepend ``prefix`` and truncate so the result fits the context length.

    The prefix counts toward ``max_tokens``; the text is shortened, never the
    prefix.
    """
    if max_tokens <= 0:
        return [prefix + text for text in texts] if prefix else texts
    max_chars = max(max_tokens * _CHARS_PER_TOKEN - len(prefix), 0)
    result: list[str] = []
    for text in texts:
        if len(text) > max_chars:
            logger.warning(
                "Truncating text from %d to %d chars (~%d token limit incl. %d-char prefix)",
                len(text),
                max_chars,
                max_tokens,
                len(prefix),
            )
            text = text[:max_chars]
        result.append(prefix + text)
    return result


@dataclass
class _Prefixes:
    query_prefix: str = ""
    document_prefix: str = ""

    def for_kind(self, kind: EmbedKind) -> str:
        if kind == "query":
            return self.query_prefix
        if kind == "document":
            return self.document_prefix
        raise ValueError(f"Unknown embedding kind {kind!r}; use 'query' or 'document'")


# Any: sentence-transformers is an optional extra, so it can't be typed here.
def _load_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install audiobookshelf-rag-ingester[sentence-transformers]"
        ) from None
    return SentenceTransformer


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    cache_dir: str | None = None
    max_tokens: int = 0
    prefixes: _Prefixes = field(default_factory=_Prefixes)

    def __post_init__(self) -> None:
        cls = _load_sentence_transformer()
        self._model = cls(self.model_name, cache_folder=self.cache_dir)

    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        chunks = _truncate(chunks, self.max_tokens, self.prefixes.for_kind(kind))
        vectors = self._model.encode(chunks, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


@dataclass
class OpenAIEmbedder:
    """Embedder for the OpenAI embeddings API (``POST {base_url}/embeddings``).

    Works with api.openai.com and any OpenAI-compatible server (Ollama's
    ``/v1`` endpoint, vLLM, LM Studio, LiteLLM, ...). The API key is optional
    so that unauthenticated self-hosted servers can be used.
    """

    model_name: str
    base_url: str
    api_key: str | None
    timeout_seconds: int
    max_tokens: int = 0
    dimensions: int | None = None
    prefixes: _Prefixes = field(default_factory=_Prefixes)

    def __post_init__(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        prefix = self.prefixes.for_kind(kind)
        if not chunks:
            return []
        chunks = _truncate(chunks, self.max_tokens, prefix)
        vectors: list[list[float]] = []
        for batch in _request_batches(chunks):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, chunks: list[str]) -> list[list[float]]:
        # Any: the request body is JSON with mixed value types.
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": chunks,
            "encoding_format": "float",
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        response = self._client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(chunks):
            raise ValueError("OpenAI /embeddings response missing 'data' or wrong length")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for request budgeting.

    ASCII text averages ~4 chars/token, but CJK and other non-Latin scripts
    are often ~1 token per character, so count non-ASCII chars individually.
    """
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (len(text) - non_ascii) // _CHARS_PER_TOKEN + non_ascii + 1


def _request_batches(
    texts: list[str],
    max_inputs: int | None = None,
    max_tokens: int | None = None,
) -> list[list[str]]:
    """Split texts into request-sized batches under the API's input/token caps."""
    max_inputs = max_inputs or _OPENAI_MAX_INPUTS_PER_REQUEST
    max_tokens = max_tokens or _OPENAI_MAX_TOKENS_PER_REQUEST
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text)
        if current and (len(current) >= max_inputs or current_tokens + tokens > max_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    ctx = settings.embedding_context_length
    prefixes = _Prefixes(
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.hf_cache_dir,
            max_tokens=ctx,
            prefixes=prefixes,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_seconds=settings.openai_timeout_seconds,
            max_tokens=ctx,
            dimensions=settings.embedding_dimensions,
            prefixes=prefixes,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
        "Supported: openai, sentence_transformers"
    )
