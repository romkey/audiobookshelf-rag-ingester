"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example. Names shared with the pipeline's other
ingesters (DATABASE_URL, QDRANT_*, EMBEDDING_*, LLM_*, ...) mean the same
thing in each of them.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

EMBEDDING_PROVIDERS = frozenset({"openai", "sentence_transformers", "sentence-transformers", "st"})
DISTANCES = frozenset({"cosine", "dot", "euclid"})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


def _url(name: str, value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://, not {value!r}")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    abs_url: str
    abs_api_token: str
    database_url: str
    abs_public_url: str | None = None
    abs_libraries: tuple[str, ...] = ()
    abs_page_size: int = 100
    request_timeout_seconds: int = 60
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    member_collection: str = "hs_member"
    catalog_collection: str = "hs_catalog"
    embedding_provider: str = "openai"
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    hf_cache_dir: str | None = None
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str | None = None
    openai_timeout_seconds: int = 120
    chunk_size: int = 1800
    chunk_overlap: int = 150
    batch_size: int = 64
    distance: str = "cosine"
    vector_size: int = 768
    embedding_context_length: int = 8192
    embedding_dimensions: int | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: int = 120
    enrichment_enabled: bool = True
    enrichment_timeout_seconds: int = 20
    google_books_api_key: str | None = None
    topic_taxonomy_file: str | None = None
    library_access_info: str | None = None
    library_apps_info: str | None = None
    scan_interval_minutes: int = 60
    full_scan_interval_hours: int = 24
    log_level: str = "INFO"

    @property
    def link_base_url(self) -> str:
        """Base of the links stored with each point: ABS_PUBLIC_URL, else ABS_URL."""
        return self.abs_public_url or self.abs_url

    @property
    def gateway_base_url(self) -> str:
        """Where generation requests go: LLM_BASE_URL, else the embeddings server."""
        return self.llm_base_url or self.openai_base_url

    @property
    def gateway_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def required(name: str) -> str:
            value = (env.get(name) or "").strip()
            if not value:
                raise ConfigError(f"Missing required environment variable: {name}")
            return value

        def optional(name: str) -> str | None:
            # An empty value (NAME= in .env) means unset.
            return (env.get(name) or "").strip() or None

        def integer(name: str, default: int, minimum: int) -> int:
            text = (env.get(name) or "").strip() or str(default)
            try:
                value = int(text)
            except ValueError:
                raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
            if value < minimum:
                raise ConfigError(f"{name} must be at least {minimum}, not {value}")
            return value

        def boolean(name: str, default: bool) -> bool:
            text = (env.get(name) or "").strip().lower()
            if not text:
                return default
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ConfigError(f"{name} must be true or false, not {text!r}")

        def choice(name: str, default: str, allowed: frozenset[str]) -> str:
            value = ((env.get(name) or "").strip() or default).lower()
            if value not in allowed:
                options = ", ".join(sorted(allowed))
                raise ConfigError(f"{name} must be one of {options}, not {value!r}")
            return value

        # About 450 tokens at ~4 characters per token (plan: 300-500 tokens).
        chunk_size = integer("CHUNK_SIZE", 1800, 1)
        chunk_overlap = integer("CHUNK_OVERLAP", 150, 0)
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
            )

        vector_size = integer("VECTOR_SIZE", 768, 1)
        # 0 or unset: the model's full size.
        embedding_dimensions = integer("EMBEDDING_DIMENSIONS", 0, 0) or None
        if embedding_dimensions is not None and embedding_dimensions != vector_size:
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS ({embedding_dimensions}) must equal "
                f"VECTOR_SIZE ({vector_size})"
            )

        log_level = ((env.get("LOG_LEVEL") or "").strip() or "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        member = optional("QDRANT_MEMBER_COLLECTION") or "hs_member"
        catalog = optional("QDRANT_CATALOG_COLLECTION") or "hs_catalog"
        if member == catalog:
            raise ConfigError(
                "QDRANT_MEMBER_COLLECTION and QDRANT_CATALOG_COLLECTION must be different"
            )
        if "hs_public" in (member, catalog):
            raise ConfigError("This ingester is member tier only; it never writes to hs_public")

        public = optional("ABS_PUBLIC_URL")
        return cls(
            abs_url=_url("ABS_URL", required("ABS_URL")),
            abs_api_token=required("ABS_API_TOKEN"),
            abs_public_url=_url("ABS_PUBLIC_URL", public) if public else None,
            abs_libraries=tuple(
                part.strip() for part in (env.get("ABS_LIBRARIES") or "").split(",") if part.strip()
            ),
            abs_page_size=integer("ABS_PAGE_SIZE", 100, 1),
            request_timeout_seconds=integer("REQUEST_TIMEOUT_SECONDS", 60, 1),
            database_url=required("DATABASE_URL"),
            qdrant_url=_url("QDRANT_URL", optional("QDRANT_URL") or "http://qdrant:6333"),
            qdrant_api_key=optional("QDRANT_API_KEY"),
            member_collection=member,
            catalog_collection=catalog,
            llm_model=optional("LLM_MODEL"),
            llm_base_url=optional("LLM_BASE_URL"),
            llm_api_key=optional("LLM_API_KEY"),
            llm_timeout_seconds=integer("LLM_TIMEOUT_SECONDS", 120, 1),
            enrichment_enabled=boolean("ENRICHMENT_ENABLED", True),
            enrichment_timeout_seconds=integer("ENRICHMENT_TIMEOUT_SECONDS", 20, 1),
            google_books_api_key=optional("GOOGLE_BOOKS_API_KEY"),
            topic_taxonomy_file=optional("TOPIC_TAXONOMY_FILE"),
            library_access_info=optional("LIBRARY_ACCESS_INFO"),
            library_apps_info=optional("LIBRARY_APPS_INFO"),
            embedding_provider=choice("EMBEDDING_PROVIDER", "openai", EMBEDDING_PROVIDERS),
            embedding_model=optional("EMBEDDING_MODEL") or "nomic-ai/modernbert-embed-base",
            hf_cache_dir=(
                optional("HF_CACHE_DIR")
                or optional("SENTENCE_TRANSFORMERS_HOME")
                or optional("HF_HOME")
            ),
            openai_base_url=_url(
                "OPENAI_BASE_URL", optional("OPENAI_BASE_URL") or "http://localhost:8080/v1"
            ),
            openai_api_key=optional("OPENAI_API_KEY"),
            openai_timeout_seconds=integer("OPENAI_TIMEOUT_SECONDS", 120, 1),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=integer("BATCH_SIZE", 64, 1),
            distance=choice("QDRANT_DISTANCE", "cosine", DISTANCES),
            vector_size=vector_size,
            # 0 disables truncation.
            embedding_context_length=integer("EMBEDDING_CONTEXT_LENGTH", 8192, 0),
            embedding_dimensions=embedding_dimensions,
            # Kept exactly as given: the trailing space usually matters.
            embedding_query_prefix=env.get("EMBEDDING_QUERY_PREFIX", ""),
            embedding_document_prefix=env.get("EMBEDDING_DOCUMENT_PREFIX", ""),
            scan_interval_minutes=integer("SCAN_INTERVAL_MINUTES", 60, 1),
            full_scan_interval_hours=integer("FULL_SCAN_INTERVAL_HOURS", 24, 1),
            log_level=log_level,
        )
