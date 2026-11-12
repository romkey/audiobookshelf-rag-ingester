"""Sync loop: mirror Audiobookshelf libraries into a Qdrant collection.

Each cycle:

1. List libraries and (minified) items from the ABS API.
2. For every item whose fingerprint (updatedAt/mtime/size/episode count) or the
   pipeline settings changed, fetch the expanded item, build documents, embed
   only chunks whose content hash changed, and delete chunks that disappeared.
3. Delete points for items and libraries that no longer exist in ABS.
4. Rebuild per-library author and summary documents (again embedding only
   what changed).

A stop request (SIGTERM) is honored between items; an interrupted cycle
skips every deletion step, so it never removes points it didn't get to check.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qdrant_client.models import Filter

from audiobookshelf_rag_ingester import __version__
from audiobookshelf_rag_ingester.config import Settings
from audiobookshelf_rag_ingester.documents import (
    DocConfig,
    Document,
    books_by_author,
    build_author_documents,
    build_ebook_documents,
    build_item_documents,
    build_library_summary_documents,
    ebook_signature,
    point_id,
)
from audiobookshelf_rag_ingester.embeddings import Embedder
from audiobookshelf_rag_ingester.extractors import extract_sections
from audiobookshelf_rag_ingester.heartbeat import touch
from audiobookshelf_rag_ingester.models import (
    Library,
    LibraryItem,
    parse_author,
    parse_item,
    parse_library,
)
from audiobookshelf_rag_ingester.qdrant_store import (
    QdrantStore,
    item_chunk_scope,
    item_scope,
    library_chunk_scope,
)

logger = logging.getLogger(__name__)

# Bump when document text/payload layout changes in a way that needs re-ingestion.
DOCUMENT_SCHEMA_VERSION = "1"
SUPPORTED_EBOOK_FORMATS = {"epub", "pdf"}


class AbsSource(Protocol):
    """What the service needs from Audiobookshelf (AbsClient, or a fake in tests)."""

    # Any: decoded JSON from the ABS API; models.py validates and normalizes it.
    def list_libraries(self) -> list[dict[str, Any]]: ...
    def list_library_items(self, library_id: str) -> list[dict[str, Any]]: ...
    def get_item(self, item_id: str) -> dict[str, Any]: ...
    def list_authors(self, library_id: str) -> list[dict[str, Any]]: ...
    def download_ebook(self, item_id: str, max_bytes: int) -> bytes | None: ...


@dataclass
class CycleStats:
    libraries: int = 0
    items_seen: int = 0
    items_processed: int = 0
    items_unchanged: int = 0
    items_deleted: int = 0
    chunks_embedded: int = 0
    chunks_unchanged: int = 0
    chunks_deleted: int = 0
    errors: int = 0
    interrupted: bool = False
    failed_items: list[str] = field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        client: AbsSource,
        store: QdrantStore,
        embedder: Embedder,
        heartbeat_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self.embedder = embedder
        self.heartbeat_path = heartbeat_path
        self.stop_event = threading.Event()
        self.cfg = DocConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            public_url=settings.abs_public_url,
        )
        self.pipeline_signature = self._pipeline_signature()

    def _pipeline_signature(self) -> str:
        s = self.settings
        raw = "|".join(
            str(v)
            for v in (
                DOCUMENT_SCHEMA_VERSION,
                s.embedding_provider,
                s.embedding_model,
                s.chunk_size,
                s.chunk_overlap,
                s.abs_public_url,
                s.ingest_ebooks,
                s.ingest_chapters,
                s.ingest_episodes,
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def stop(self) -> None:
        """Ask a running cycle or loop to finish; safe to call from a signal handler."""
        self.stop_event.set()

    def _beat(self) -> None:
        if self.heartbeat_path is not None:
            touch(self.heartbeat_path)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        interval = self.settings.poll_interval_seconds
        while not self.stop_event.is_set():
            try:
                self.run_cycle()
            except Exception:
                logger.exception("Ingestion cycle failed; will retry next cycle")
            if self.stop_event.is_set():
                break
            logger.info("Sleeping %ss until next sync", interval)
            deadline = time.monotonic() + interval
            # Wake at least every minute to keep the heartbeat fresh.
            while not self.stop_event.is_set():
                self._beat()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.stop_event.wait(min(60.0, remaining))
        logger.info("Stopped")

    # ------------------------------------------------------------------
    # One sync pass
    # ------------------------------------------------------------------

    def run_cycle(self) -> CycleStats:
        stats = CycleStats()
        started = time.monotonic()
        self._beat()
        logger.info(
            "audiobookshelf-rag-ingester %s: starting sync from %s",
            __version__,
            self.settings.abs_url,
        )

        libraries = self._selected_libraries()
        stats.libraries = len(libraries)

        for library in libraries:
            if self.stop_event.is_set():
                break
            try:
                self._sync_library(library, stats)
            except Exception:
                stats.errors += 1
                logger.exception(
                    "Failed to sync library %r (%s)", library.name, library.id
                )

        if self.stop_event.is_set():
            stats.interrupted = True
            logger.info("Sync interrupted; skipped removing deleted libraries")
        else:
            # Drop libraries that were deleted in ABS or deselected via ABS_LIBRARIES.
            current = {lib.id for lib in libraries}
            for stale_lib in sorted(self.store.library_ids() - current):
                logger.info(
                    "Removing points for library %s (no longer present/selected)",
                    stale_lib,
                )
                self.store.delete_library(stale_lib)

        logger.info(
            "Sync finished in %.1fs: libraries=%d items=%d processed=%d unchanged=%d "
            "deleted=%d chunks_embedded=%d chunks_unchanged=%d chunks_deleted=%d "
            "errors=%d "
            "collection_points=%d",
            time.monotonic() - started,
            stats.libraries,
            stats.items_seen,
            stats.items_processed,
            stats.items_unchanged,
            stats.items_deleted,
            stats.chunks_embedded,
            stats.chunks_unchanged,
            stats.chunks_deleted,
            stats.errors,
            self.store.count(),
        )
        return stats

    def _selected_libraries(self) -> list[Library]:
        libraries = [parse_library(d) for d in self.client.list_libraries()]
        wanted = {w.casefold() for w in self.settings.abs_libraries}
        if wanted:
            libraries = [
                lib
                for lib in libraries
                if lib.id.casefold() in wanted or lib.name.casefold() in wanted
            ]
            if not libraries:
                logger.warning(
                    "ABS_LIBRARIES=%s matched no libraries", self.settings.abs_libraries
                )
        logger.info(
            "Libraries to sync: %s",
            ", ".join(f"{lib.name} ({lib.media_type})" for lib in libraries) or "none",
        )
        return libraries

    def _sync_library(self, library: Library, stats: CycleStats) -> None:
        items = [parse_item(d) for d in self.client.list_library_items(library.id)]
        stats.items_seen += len(items)
        logger.info("Library %r: %d items", library.name, len(items))

        stored = self.store.get_payloads(
            [point_id(f"item:{i.id}:metadata") for i in items],
            ["item_fingerprint", "pipeline_signature"],
        )

        for idx, item in enumerate(items, 1):
            if self.stop_event.is_set():
                return  # skip deletions and summaries: we didn't see everything
            meta = stored.get(point_id(f"item:{item.id}:metadata"), {})
            if (
                not self.settings.force_reingest
                and meta.get("item_fingerprint") == item.fingerprint()
                and meta.get("pipeline_signature") == self.pipeline_signature
            ):
                stats.items_unchanged += 1
                continue
            self._beat()
            logger.info(
                "[%s %d/%d] Ingesting %s %r",
                library.name,
                idx,
                len(items),
                item.media_type,
                item.title,
            )
            try:
                self._sync_item(library, item.id, stats)
                stats.items_processed += 1
            except Exception:
                stats.errors += 1
                stats.failed_items.append(item.id)
                logger.exception("Failed to ingest item %s (%r)", item.id, item.title)
            if self.settings.item_delay_seconds > 0:
                self.stop_event.wait(self.settings.item_delay_seconds)

        if self.stop_event.is_set():
            return  # stopped during the last item: still skip deletions

        # Remove items that no longer exist in this library.
        current_ids = {i.id for i in items}
        gone = sorted(self.store.item_ids_in_library(library.id) - current_ids)
        if gone:
            logger.info(
                "Library %r: removing %d deleted items", library.name, len(gone)
            )
            self.store.delete_items(gone)
            stats.items_deleted += len(gone)

        if self.settings.ingest_authors and library.media_type == "book":
            try:
                self._sync_authors(library, items, stats)
            except Exception:
                stats.errors += 1
                logger.exception("Failed to sync authors for library %r", library.name)
        else:
            self._sync_docs([], library_chunk_scope(library.id, "author"), stats)

        docs = (
            build_library_summary_documents(library, items, self.cfg)
            if self.settings.ingest_library_summary
            else []
        )
        self._sync_docs(docs, library_chunk_scope(library.id, "library_summary"), stats)

    def _sync_item(self, library: Library, item_id: str, stats: CycleStats) -> None:
        item = parse_item(self.client.get_item(item_id))
        docs = build_item_documents(
            item,
            library,
            self.cfg,
            include_chapters=self.settings.ingest_chapters,
            include_episodes=self.settings.ingest_episodes,
        )
        keep_extra: set[str] = set()
        ebook_ok = True
        if self.settings.ingest_ebooks:
            ebook_docs, keep_extra, ebook_ok = self._ebook_documents(library, item)
            docs.extend(ebook_docs)

        self._sync_docs(docs, item_scope(item.id), stats, keep_extra=keep_extra)

        # Record the fingerprint last so a failure part-way through is retried.
        if ebook_ok:
            self.store.set_payload(
                item_scope(item.id),
                {
                    "item_fingerprint": item.fingerprint(),
                    "pipeline_signature": self.pipeline_signature,
                },
            )

    def _ebook_documents(
        self, library: Library, item: LibraryItem
    ) -> tuple[list[Document], set[str], bool]:
        """Return (new ebook docs, existing point ids to keep, success flag)."""
        fmt = (item.ebook.format if item.ebook else item.ebook_format or "") or ""
        fmt = fmt.lower()
        if not item.ebook or fmt not in SUPPORTED_EBOOK_FORMATS:
            return [], set(), True

        signature = ebook_signature(item, self.cfg)
        existing = self.store.scroll_ids(
            item_chunk_scope(item.id, "ebook_content"), ["ebook_signature"]
        )
        if (
            existing
            and not self.settings.force_reingest
            and all(p.get("ebook_signature") == signature for p in existing.values())
        ):
            logger.info("  ebook unchanged; keeping %d chunks", len(existing))
            return [], set(existing), True

        try:
            data = self.client.download_ebook(item.id, self.settings.ebook_max_bytes)
        except Exception:
            logger.exception("  Failed to download ebook for item %s", item.id)
            return [], set(existing), False
        if not data:
            return [], set(), True

        sections = extract_sections(data, fmt)
        if not sections:
            logger.info(
                "  No extractable text in %s ebook %r", fmt, item.ebook.filename
            )
            return [], set(), True
        docs = build_ebook_documents(
            item, library, sections, self.cfg, ebook_signature=signature
        )
        logger.info(
            "  Extracted %d ebook chunks from %r", len(docs), item.ebook.filename
        )
        return docs, set(), True

    def _sync_authors(
        self, library: Library, items: list[LibraryItem], stats: CycleStats
    ) -> None:
        authors = [parse_author(a) for a in self.client.list_authors(library.id)]
        grouped = books_by_author(items)
        docs: list[Document] = []
        for author in authors:
            docs.extend(
                build_author_documents(
                    author, library, grouped.get(author.name.casefold(), []), self.cfg
                )
            )
        logger.info(
            "Library %r: %d authors -> %d author chunks",
            library.name,
            len(authors),
            len(docs),
        )
        self._sync_docs(docs, library_chunk_scope(library.id, "author"), stats)

    # ------------------------------------------------------------------
    # Core diff-and-upsert
    # ------------------------------------------------------------------

    def _sync_docs(
        self,
        docs: list[Document],
        scope: Filter,
        stats: CycleStats,
        *,
        keep_extra: set[str] | None = None,
    ) -> None:
        # De-duplicate keys (last wins): one upsert must not hold an id twice.
        unique = list({d.key: d for d in docs}.values())
        ids = [d.point_id for d in unique]
        existing = self.store.get_payloads(ids, ["content_hash"])
        changed = [
            d
            for d in unique
            if self.settings.force_reingest
            or existing.get(d.point_id, {}).get("content_hash") != d.content_hash
        ]
        stats.chunks_unchanged += len(unique) - len(changed)

        if changed:
            vectors = self.embedder.encode([d.context for d in changed])
            stats.chunks_embedded += self.store.upsert(changed, vectors)

        stats.chunks_deleted += self.store.delete_stale(
            scope, set(ids) | (keep_extra or set())
        )
