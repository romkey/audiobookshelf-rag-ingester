"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

EMBEDDING_PROVIDERS = ("openai", "sentence_transformers")
DISTANCES = ("cosine", "dot", "euclid", "manhattan")


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


def _required(name: str, value: str | None) -> str:
    if not value or not value.strip():
        raise ConfigError(f"{name} is required")
    return value.strip()


def _int(name: str, value: str | None, default: int, minimum: int = 0) -> int:
    if value is None or not value.strip():
        return default
    try:
        number = int(value)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {value!r}") from None
    if number < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {number}")
    return number


def _float(name: str, value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not {value!r}") from None
    if number < 0:
        raise ConfigError(f"{name} must not be negative, not {number}")
    return number


def _bool(name: str, value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, not {value!r}")


def _choice(
    name: str, value: str | None, default: str, choices: tuple[str, ...]
) -> str:
    text = (value or "").strip().lower().replace("-", "_") or default
    if text not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return text


def _optional(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _prefix(value: str | None) -> str:
    """An embedding prefix, trimmed and followed by a single space ("" if unset)."""
    stripped = (value or "").strip()
    return f"{stripped} " if stripped else ""


def _url(name: str, value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://, not {value!r}")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    # Audiobookshelf
    abs_url: str
    abs_api_token: str
    abs_public_url: str
    abs_libraries: tuple[str, ...] = ()
    abs_page_size: int = 100
    request_timeout_seconds: int = 60
    item_delay_seconds: float = 0.0

    # What to index
    ingest_ebooks: bool = True
    ebook_max_bytes: int = 100 * 1024 * 1024
    ingest_chapters: bool = True
    ingest_episodes: bool = True
    ingest_authors: bool = True
    ingest_library_summary: bool = True

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "audiobookshelf"
    qdrant_distance: str = "cosine"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_batch_size: int = 32
    embedding_context_length: int = 8192
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    hf_cache_dir: str | None = None

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 120

    # Scheduling
    poll_interval_seconds: int = 3600
    force_reingest: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        abs_url = _url("ABS_URL", _required("ABS_URL", env.get("ABS_URL")))
        public = _optional(env.get("ABS_PUBLIC_URL"))
        libraries = tuple(
            part.strip()
            for part in env.get("ABS_LIBRARIES", "").split(",")
            if part.strip()
        )

        chunk_size = _int(
            "CHUNK_SIZE", env.get("CHUNK_SIZE"), cls.chunk_size, minimum=100
        )
        chunk_overlap = _int(
            "CHUNK_OVERLAP", env.get("CHUNK_OVERLAP"), cls.chunk_overlap
        )
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({chunk_size})"
            )

        log_level = (env.get("LOG_LEVEL") or cls.log_level).strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        ebook_mb = _int("EBOOK_MAX_MB", env.get("EBOOK_MAX_MB"), 100, minimum=1)

        return cls(
            abs_url=abs_url,
            abs_api_token=_required("ABS_API_TOKEN", env.get("ABS_API_TOKEN")),
            abs_public_url=_url("ABS_PUBLIC_URL", public) if public else abs_url,
            abs_libraries=libraries,
            abs_page_size=_int(
                "ABS_PAGE_SIZE", env.get("ABS_PAGE_SIZE"), cls.abs_page_size, minimum=1
            ),
            request_timeout_seconds=_int(
                "REQUEST_TIMEOUT_SECONDS",
                env.get("REQUEST_TIMEOUT_SECONDS"),
                cls.request_timeout_seconds,
                minimum=1,
            ),
            item_delay_seconds=_float(
                "ITEM_DELAY_SECONDS",
                env.get("ITEM_DELAY_SECONDS"),
                cls.item_delay_seconds,
            ),
            ingest_ebooks=_bool(
                "INGEST_EBOOKS", env.get("INGEST_EBOOKS"), cls.ingest_ebooks
            ),
            ebook_max_bytes=ebook_mb * 1024 * 1024,
            ingest_chapters=_bool(
                "INGEST_CHAPTERS", env.get("INGEST_CHAPTERS"), cls.ingest_chapters
            ),
            ingest_episodes=_bool(
                "INGEST_EPISODES", env.get("INGEST_EPISODES"), cls.ingest_episodes
            ),
            ingest_authors=_bool(
                "INGEST_AUTHORS", env.get("INGEST_AUTHORS"), cls.ingest_authors
            ),
            ingest_library_summary=_bool(
                "INGEST_LIBRARY_SUMMARY",
                env.get("INGEST_LIBRARY_SUMMARY"),
                cls.ingest_library_summary,
            ),
            qdrant_url=_url("QDRANT_URL", env.get("QDRANT_URL") or cls.qdrant_url),
            qdrant_api_key=_optional(env.get("QDRANT_API_KEY")),
            qdrant_collection=(
                env.get("QDRANT_COLLECTION") or cls.qdrant_collection
            ).strip(),
            qdrant_distance=_choice(
                "QDRANT_DISTANCE",
                env.get("QDRANT_DISTANCE"),
                cls.qdrant_distance,
                DISTANCES,
            ),
            embedding_provider=_choice(
                "EMBEDDING_PROVIDER",
                env.get("EMBEDDING_PROVIDER"),
                cls.embedding_provider,
                EMBEDDING_PROVIDERS,
            ),
            embedding_model=(env.get("EMBEDDING_MODEL") or cls.embedding_model).strip(),
            embedding_batch_size=_int(
                "EMBEDDING_BATCH_SIZE",
                env.get("EMBEDDING_BATCH_SIZE"),
                cls.embedding_batch_size,
                minimum=1,
            ),
            embedding_context_length=_int(
                "EMBEDDING_CONTEXT_LENGTH",
                env.get("EMBEDDING_CONTEXT_LENGTH"),
                cls.embedding_context_length,
            ),
            embedding_query_prefix=_prefix(env.get("EMBEDDING_QUERY_PREFIX")),
            embedding_document_prefix=_prefix(env.get("EMBEDDING_DOCUMENT_PREFIX")),
            openai_base_url=_url(
                "OPENAI_BASE_URL", env.get("OPENAI_BASE_URL") or cls.openai_base_url
            ),
            openai_api_key=_optional(env.get("OPENAI_API_KEY")),
            hf_cache_dir=_optional(env.get("HF_CACHE_DIR")),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            poll_interval_seconds=_int(
                "POLL_INTERVAL_SECONDS",
                env.get("POLL_INTERVAL_SECONDS"),
                cls.poll_interval_seconds,
            ),
            force_reingest=_bool(
                "FORCE_REINGEST", env.get("FORCE_REINGEST"), cls.force_reingest
            ),
            log_level=log_level,
        )
