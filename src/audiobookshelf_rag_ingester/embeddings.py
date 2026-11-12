import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from audiobookshelf_rag_ingester.config import Settings

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4


Kind = Literal["document", "query"]


class Embedder(Protocol):
    def encode(
        self, texts: list[str], kind: Kind = "document"
    ) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


def truncate(texts: list[str], max_tokens: int, prefix: str = "") -> list[str]:
    """Truncate texts that would exceed the model's context length (~4 chars/token).

    The prefix is prepended after truncation and counts toward the limit.
    """
    if max_tokens <= 0:
        return [prefix + t for t in texts] if prefix else texts
    max_chars = max(0, max_tokens * _CHARS_PER_TOKEN - len(prefix))
    result: list[str] = []
    for text in texts:
        if len(text) > max_chars:
            logger.warning(
                "Truncating text from %d to %d chars (~%d token limit)",
                len(text),
                max_chars,
                max_tokens,
            )
            text = text[:max_chars]
        result.append(prefix + text)
    return result


class _DimensionMixin:
    _dimension: int | None = None
    query_prefix: str = ""
    document_prefix: str = ""

    def _prefix(self, kind: Kind) -> str:
        return self.query_prefix if kind == "query" else self.document_prefix

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode(["dimension probe"])[0])  # type: ignore[attr-defined]
        return self._dimension


@dataclass
class OpenAIEmbedder(_DimensionMixin):
    """Embeddings from OpenAI, or any server speaking its ``/embeddings`` API.

    The API key is optional so the embedder can also talk to a local
    OpenAI-compatible server that needs no authentication.
    """

    model_name: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout_seconds: int = 120
    max_tokens: int = 0
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def encode(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        texts = truncate(texts, self.max_tokens, self._prefix(kind))
        vectors: list[list[float]] = []
        step = max(1, self.batch_size)
        for start in range(0, len(texts), step):
            batch = texts[start : start + step]
            response = self._client.post(
                "/embeddings", json={"model": self.model_name, "input": batch}
            )
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ValueError(
                    "OpenAI /embeddings response missing or wrong-length 'data'"
                )
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors.extend(item["embedding"] for item in ordered)
        return vectors


@dataclass
class SentenceTransformerEmbedder(_DimensionMixin):
    model_name: str
    cache_dir: str | None = None
    max_tokens: int = 0
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: "
                "pip install 'audiobookshelf-rag-ingester[sentence-transformers]'"
            ) from None
        self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

    def encode(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        texts = truncate(texts, self.max_tokens, self._prefix(kind))
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.hf_cache_dir,
            max_tokens=settings.embedding_context_length,
            batch_size=settings.embedding_batch_size,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_seconds=max(settings.request_timeout_seconds, 120),
            max_tokens=settings.embedding_context_length,
            batch_size=settings.embedding_batch_size,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
        "Supported: openai, sentence_transformers"
    )
