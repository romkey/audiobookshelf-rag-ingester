"""One sync run: Audiobookshelf -> staging + registry -> sanitize -> index in Qdrant.

Modes:
- ``incremental`` (hourly): items whose ``updatedAt`` changed, plus deletions.
- ``full`` (nightly): every item; points not seen in this run are deleted,
  unless the run saw less than half the items of the previous full run.
- ``reindex``: rebuild every point from staging and the registry without
  reading Audiobookshelf, re-embedding everything (after a model change).

Audiobooks become manifestations of shared works in the entity registry and
are indexed through one ``hs_catalog`` point per work, built from every
format the registry holds. Podcasts are not works: each podcast and episode
is indexed in ``hs_member`` with ``source: podcast``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from audiobookshelf_rag_ingester.abs_client import AbsClient
from audiobookshelf_rag_ingester.chunking import split_text
from audiobookshelf_rag_ingester.config import Settings
from audiobookshelf_rag_ingester.db import Database, RunCounts, StagedDocument
from audiobookshelf_rag_ingester.documents import (
    PODCAST_SOURCE,
    TIER,
    base_payload,
    catalog_text,
    html_to_text,
    json_hash,
    language_code,
    meaningful_chapters,
    ms_to_rfc3339,
    payload_hash,
    podcast_points,
    point_id,
    year_of,
)
from audiobookshelf_rag_ingester.documents import overview_sections as build_overview
from audiobookshelf_rag_ingester.embeddings import Embedder, build_embedder
from audiobookshelf_rag_ingester.enrichment import Enricher
from audiobookshelf_rag_ingester.linker import Linker
from audiobookshelf_rag_ingester.llm import LLM, LLMError
from audiobookshelf_rag_ingester.models import (
    Library,
    LibraryItem,
    ListedItem,
    parse_item,
    parse_library,
    parse_listed,
)
from audiobookshelf_rag_ingester.qdrant_store import (
    CATALOG_INDEXES,
    COMMON_INDEXES,
    PODCAST_INDEXES,
    QdrantStore,
)
from audiobookshelf_rag_ingester.sanitize import redact
from audiobookshelf_rag_ingester.sparse import SPARSE_ENCODER, sparse_vector
from audiobookshelf_rag_ingester.works import (
    author_surname,
    clean_audiobook_title,
    normalize_isbn,
    series_id_for,
    similarity,
    work_id_for,
)

logger = logging.getLogger(__name__)

SOURCE = "audiobookshelf"
CATALOG_SOURCE = "catalog"
OVERVIEW_DOC_ID = "audiobookshelf:library-overview"
# Fuzzy work matching against other sources' books (same first-author surname),
# the same thresholds as the ebook source. A series and sequence match lifts a
# review-range score to a merge (plan 03: series plus sequence as tiebreaker).
AUTO_MERGE_SIMILARITY = 0.93
REVIEW_SIMILARITY = 0.80
# Enrichment misses are retried after this long.
ENRICHMENT_RETRY = timedelta(days=30)
# A run that sees less than this share of the previous run's items deletes nothing.
SAFETY_RATIO = 0.5

Mode = Literal["incremental", "full", "reindex"]
# Any: registry rows, staged records and payloads are JSON.
JSON = dict[str, Any]


class AbsSource(Protocol):
    """What the pipeline needs from Audiobookshelf (AbsClient, or a fake in tests)."""

    def list_libraries(self) -> list[JSON]: ...
    def list_library_items(self, library_id: str) -> list[JSON]: ...
    def get_item(self, item_id: str) -> JSON: ...


@dataclass(frozen=True)
class RunResult:
    status: Literal["ok", "busy", "valve", "partial"]
    counts: RunCounts
    message: str = ""


def load_taxonomy(path: str | None) -> list[str]:
    text = (
        Path(path).read_text(encoding="utf-8")
        if path
        else (files("audiobookshelf_rag_ingester") / "taxonomy.txt").read_text(encoding="utf-8")
    )
    slugs = [line.strip() for line in text.splitlines()]
    return [s for s in slugs if s and not s.startswith("#")]


def item_url(base_url: str, item_id: str) -> str:
    return f"{base_url.rstrip('/')}/item/{item_id}"


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        qdrant: QdrantClient,
        client: AbsSource,
        llm: LLM | None = None,
        enricher: Enricher | None = None,
        stop: threading.Event | None = None,
        beat: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.client = client
        self.llm = llm
        self.enricher = enricher
        self.stop = stop or threading.Event()
        self.qdrant = qdrant
        # Touched while work progresses, for the serve loop's health check.
        self.beat = beat or (lambda: None)
        self.taxonomy = load_taxonomy(settings.topic_taxonomy_file)
        metadata = {
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.vector_size,
            "embedding_document_prefix": settings.embedding_document_prefix,
            "embedding_query_prefix": settings.embedding_query_prefix,
            "sparse_encoder": SPARSE_ENCODER,
        }

        def store(name: str, indexes: dict[str, qm.PayloadSchemaType]) -> QdrantStore:
            return QdrantStore(
                client=qdrant,
                collection=name,
                vector_size=settings.vector_size,
                distance=settings.distance,
                metadata=metadata,
                indexes=indexes,
            )

        self.member = store(settings.member_collection, COMMON_INDEXES | PODCAST_INDEXES)
        self.catalog = store(settings.catalog_collection, COMMON_INDEXES | CATALOG_INDEXES)
        self.linker = Linker(
            ((r["entity_type"], r["entity_id"], r["alias"]) for r in db.aliases()),
            self._confirm_link if llm is not None else None,
        )
        self._candidates: dict[str, list[JSON]] | None = None

    def close(self) -> None:
        """Close the HTTP clients this pipeline opened; serve builds one per scan."""
        self.qdrant.close()
        for client in (self.embedder, self.llm, self.enricher):
            closer = getattr(client, "close", None)
            if closer is not None:
                closer()

    def _confirm_link(self, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        if self.llm is None:
            return False
        self.beat()
        try:
            return self.llm.confirm_link(
                text=text, alias=alias, entity_type=entity_type, entity_id=entity_id
            )
        except LLMError as e:
            logger.warning("Link confirmation failed: %s", e)
            return False

    def _link(self, text: str) -> dict[str, list[str]]:
        return self.linker.link(text).ids

    # -- run ------------------------------------------------------------------

    def run(self, mode: Mode) -> RunResult:
        counts = RunCounts()
        with self.db.source_lock(SOURCE) as locked:
            if not locked:
                return RunResult("busy", counts, "another audiobookshelf run holds the lock")
            run_id = f"{SOURCE}-{mode}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            self.db.start_run(run_id=run_id, source=SOURCE, mode=mode)
            try:
                result = self._run(mode, run_id, counts)
            except Exception as e:
                # Record the failure on the status row, then let main() report it.
                counts.errors.append(f"{type(e).__name__}: {e}")
                self.db.conn.rollback()
                self.db.finish_run(run_id=run_id, counts=counts, ok=False, message=str(e))
                raise
            counts.quarantined = self.db.quarantine_count([SOURCE])
            self.db.finish_run(
                run_id=run_id, counts=counts, ok=result.status == "ok", message=result.message
            )
            return result

    def _run(self, mode: Mode, run_id: str, counts: RunCounts) -> RunResult:
        affected: set[str] = set()
        changed_podcasts: set[str] = set()
        if mode == "reindex":
            self._index(mode, run_id, affected, changed_podcasts, allow_delete=True)
            return RunResult("ok", counts)

        listed, failed = self._list_items(counts)
        counts.seen = len(listed)
        staged = self.db.staged_versions(SOURCE)
        candidates = (
            sorted(listed)
            if mode == "full"
            else [
                i
                for i in sorted(listed)
                if i not in staged or staged[i][0] != str(listed[i].updated_at)
            ]
        )
        changed: list[str] = []
        self._stage_items(candidates, staged, run_id, counts, affected, changed_podcasts, changed)
        if self.stop.is_set():
            # Index what was staged: the next incremental run sees those
            # items as unchanged and would never index them.
            self._index_or_forget(
                "incremental", run_id, affected, changed_podcasts, changed, allow_delete=False
            )
            return RunResult("ok", counts, "stopped early")

        media = self.db.staged_media(SOURCE)
        # Never delete for a library whose listing failed.
        deleted = sorted(
            i for i in staged if i not in listed and (media.get(i) or (None, None))[0] not in failed
        )
        previous = self.db.previous_full_run_seen(SOURCE, run_id)
        baseline = max(previous or 0, len(staged))
        status: Literal["ok", "valve", "partial"] = "ok"
        message = ""
        if deleted and counts.seen < baseline * SAFETY_RATIO:
            status = "valve"
            message = (
                f"Saw {counts.seen} items but {baseline} were expected; not deleting "
                f"{len(deleted)} items. Check ABS_URL, ABS_API_TOKEN and ABS_LIBRARIES."
            )
            logger.error("Safety valve: %s", message)
        elif deleted:
            affected |= self._delete_items(deleted, media, counts)
        if failed:
            status = "partial" if status == "ok" else status
            message = message or (
                f"Listing failed for libraries {', '.join(sorted(failed))}; their items were "
                "neither updated nor deleted"
            )
        self._index_or_forget(
            mode, run_id, affected, changed_podcasts, changed, allow_delete=status == "ok"
        )
        return RunResult(status, counts, message)

    def _index_or_forget(
        self,
        mode: Mode,
        run_id: str,
        affected: set[str],
        changed_podcasts: set[str],
        changed: list[str],
        *,
        allow_delete: bool,
    ) -> None:
        """Index; if that fails, make the next incremental run restage what changed.

        Staging is committed first, and incremental runs skip items whose
        updatedAt matches staging, so a failed index would otherwise leave
        them unindexed until the next full run.
        """
        try:
            self._index(mode, run_id, affected, changed_podcasts, allow_delete=allow_delete)
        except Exception:
            self.db.conn.rollback()
            self.db.forget_versions(SOURCE, changed)
            self.db.commit()
            raise

    def _libraries(self) -> list[Library]:
        libraries = [parse_library(d) for d in self.client.list_libraries()]
        wanted = {w.casefold() for w in self.settings.abs_libraries}
        if wanted:
            libraries = [
                lib
                for lib in libraries
                if lib.id.casefold() in wanted or lib.name.casefold() in wanted
            ]
            if not libraries:
                logger.warning("ABS_LIBRARIES=%s matched no libraries", self.settings.abs_libraries)
        logger.info(
            "Libraries: %s",
            ", ".join(f"{lib.name} ({lib.media_type})" for lib in libraries) or "none",
        )
        return libraries

    def _list_items(self, counts: RunCounts) -> tuple[dict[str, ListedItem], set[str]]:
        """Every listed item by id, and the ids of libraries whose listing failed."""
        listed: dict[str, ListedItem] = {}
        failed: set[str] = set()
        for library in self._libraries():
            self.beat()
            try:
                for data in self.client.list_library_items(library.id):
                    item = parse_listed(data)
                    listed[item.id] = item
            except Exception as e:
                failed.add(library.id)
                counts.errors.append(f"library {library.id}: {type(e).__name__}: {e}")
                logger.exception("Failed to list library %r (%s)", library.name, library.id)
        return listed, failed

    # -- staging ----------------------------------------------------------------

    def _stage_items(
        self,
        item_ids: list[str],
        staged: dict[str, tuple[str | None, str]],
        run_id: str,
        counts: RunCounts,
        affected: set[str],
        changed_podcasts: set[str],
        changed: list[str],
    ) -> None:
        unchanged: list[str] = []
        for item_id in item_ids:
            if self.stop.is_set():
                logger.info("Stopping early as requested")
                break
            self.beat()
            try:
                item = parse_item(self.client.get_item(item_id))
                raw = item.to_json()
                raw_hash = json_hash(raw)
                if item_id in staged and staged[item_id] == (str(item.updated_at), raw_hash):
                    unchanged.append(item_id)
                    continue
                if item.media_type == "podcast":
                    self.stage_podcast(item, raw_hash, run_id)
                    changed_podcasts.add(item_id)
                else:
                    affected |= self.stage_book(item, raw_hash, run_id)
            except Exception as e:
                # One bad item never stops the run.
                self.db.conn.rollback()
                logger.exception("Failed to stage item %s", item_id)
                counts.errors.append(f"item {item_id}: {type(e).__name__}: {e}")
                continue
            self.db.commit()
            changed.append(item_id)
            if item_id in staged:
                counts.updated += 1
            else:
                counts.added += 1
        self.db.touch_staging(SOURCE, unchanged, run_id)
        self.db.commit()

    def _sanitize(self, text: str, doc_id: str, field: str, run_id: str) -> str:
        result = redact(text)
        if result.findings:
            logger.warning("Redacted %s secret(s) in %s %s", len(result.findings), doc_id, field)
            self.db.add_quarantine(
                source=SOURCE,
                source_doc_id=doc_id,
                reason="secret-redacted",
                detail={"field": field, "rules": result.rules},
                run_id=run_id,
            )
        return result.text

    def _enrich(self, item_id: str, isbn13: str | None, title: str, authors: list[str]) -> JSON:
        rows = self.db.enrichment_get(SOURCE, item_id)
        fresh = bool(rows) and all(
            r["matched"] or datetime.now(UTC) - r["fetched_at"] < ENRICHMENT_RETRY for r in rows
        )
        if not fresh and self.enricher is not None:
            for result in self.enricher.lookup(isbn13=isbn13, title=title, authors=authors):
                self.db.enrichment_put(
                    source=SOURCE,
                    source_doc_id=item_id,
                    origin=result.origin,
                    lookup_key=result.lookup_key,
                    matched=result.matched,
                    confidence=result.confidence,
                    fields=result.fields,
                )
            rows = self.db.enrichment_get(SOURCE, item_id)
        for row in rows:
            if row["matched"] and (
                row["fields"].get("description") or row["fields"].get("subjects")
            ):
                return {"origin": row["origin"], **row["fields"]}
        return {}

    def stage_book(self, item: LibraryItem, raw_hash: str, run_id: str) -> set[str]:
        """Stage one audiobook and attach it to its work; returns the works it touched."""
        doc_id = item.id
        description = self._sanitize(html_to_text(item.description), doc_id, "description", run_id)
        chapters = [
            line
            for line in self._sanitize(
                "\n".join(meaningful_chapters([c.title for c in item.chapters])),
                doc_id,
                "chapters",
                run_id,
            ).splitlines()
            if line
        ]
        series_names = [s.name for s in item.series]
        title = clean_audiobook_title(item.title, series_names)
        isbn13 = normalize_isbn(item.isbn)
        tags = sorted(set(item.genres) | set(item.tags))

        # Enriched fields are kept apart from the library's own, and labelled.
        enriched: JSON = {}
        if self.settings.enrichment_enabled and (not description or not tags):
            enriched = self._enrich(doc_id, isbn13, title, item.authors)
        enriched_description = self._sanitize(
            str(enriched.get("description") or ""), doc_id, "enriched_description", run_id
        )
        enriched_subjects = [str(s) for s in enriched.get("subjects") or []]

        topics: list[str] = []
        marketing = False
        good_for = ""
        if self.llm is not None:
            try:
                analysis = self.llm.analyze_book(
                    title=title,
                    authors=item.authors,
                    description=description or enriched_description,
                    tags=tags + enriched_subjects,
                    toc="\n".join(chapters),
                    taxonomy=self.taxonomy,
                )
                topics = analysis.topics
                marketing = analysis.description_is_marketing and bool(description)
                good_for = analysis.good_for
            except LLMError as e:
                logger.warning("Book analysis failed for item %s: %s", doc_id, e)

        url = item_url(self.settings.link_base_url, doc_id)
        # An item with audio is an audiobook; an ebook-only item is the work's ebook.
        fmt = "audiobook" if item.duration or item.audio_files else "ebook"
        first_series = item.series[0] if item.series else None
        series_index = first_series.index() if first_series else None
        languages = [code for code in [language_code(item.language)] if code]
        file_formats = sorted({f.ext for f in item.audio_files if f.ext})
        needs_summary = not description or marketing
        details: JSON = {
            "title": title,
            "subtitle": item.subtitle,
            "authors": item.authors,
            "narrators": item.narrators,
            "series": first_series.name if first_series else None,
            "series_index": series_index,
            "publisher": item.publisher,
            "year": year_of(item.published_year or item.published_date),
            "languages": languages,
            "tags": tags,
            "genres": item.genres,
            "topics": topics,
            "identifiers": {k: v for k, v in (("isbn", item.isbn), ("asin", item.asin)) if v},
            "rating": None,
            "description": description,
            "description_is_marketing": marketing,
            "enriched_description": enriched_description or None,
            "enriched_subjects": enriched_subjects,
            "enrichment_origin": enriched.get("origin"),
            "good_for": good_for if needs_summary and good_for else None,
            "good_for_generated": bool(needs_summary and good_for),
            "toc": [],
            "chapters": chapters,
            "chapter_count": len(item.chapters),
            "formats": [{"format": f, "size_bytes": None, "available": True} for f in file_formats],
            "audio_files": len(item.audio_files),
            "duration_seconds": round(item.duration, 1) if item.duration else None,
            "abridged": item.abridged,
            "explicit": item.explicit,
            "has_ebook_file": item.has_ebook,
            "cover_path": item.cover_path,
            "date_added": ms_to_rfc3339(item.added_at),
            "updated_at": ms_to_rfc3339(item.updated_at),
            "location": None,
        }
        record = {**item.to_json(), "description": description, "chapters_text": chapters}
        flags = {"secret_redacted": "[REDACTED]" in (description + "\n".join(chapters))}
        self.db.upsert_staging(
            StagedDocument(
                source=SOURCE,
                source_doc_id=doc_id,
                title=item.title,
                url=url,
                tier=TIER,
                raw_content=record,
                extracted_text="\n\n".join(p for p in (description, "\n".join(chapters)) if p),
                content_hash=raw_hash,
                source_updated_at=str(item.updated_at),
                flags=flags,
            ),
            run_id=run_id,
        )

        series_id = series_id_for(first_series.name) if first_series else None
        work_id = self._resolve_work(title, item.authors, isbn13, series_id, series_index)
        if series_id and first_series:
            self.db.upsert_series(series_id, first_series.name)
        self.db.upsert_work(
            work_id=work_id,
            title=title,
            authors=item.authors,
            isbn13=isbn13,
            series_id=series_id,
            series_index=series_index,
        )
        previous = self.db.upsert_manifestation(
            source=SOURCE, source_id=doc_id, work_id=work_id, format=fmt, url=url, details=details
        )
        self.db.set_work_topics(work_id)
        touched = {work_id}
        if previous and previous != work_id:
            touched.add(previous)
            self.db.set_work_topics(previous)
            self.db.delete_work_if_empty(previous)

        self.db.upsert_catalog_item(
            {
                "source": SOURCE,
                "source_id": doc_id,
                "work_id": work_id,
                "format": fmt,
                "title": title,
                "sort_title": None,
                "subtitle": item.subtitle,
                "authors": item.authors,
                "series": details["series"],
                "series_index": series_index,
                "publisher": item.publisher,
                "published_year": details["year"],
                "languages": languages,
                "tags": tags,
                "topics": topics,
                "file_formats": file_formats,
                "rating": None,
                "url": url,
                "location": None,
                "date_added": details["date_added"],
                "narrators": item.narrators,
                "duration_seconds": details["duration_seconds"],
                "genres": item.genres,
                "abridged": item.abridged,
            }
        )
        return touched

    def stage_podcast(self, item: LibraryItem, raw_hash: str, run_id: str) -> None:
        """Stage one podcast and its episodes (indexed later from staging)."""
        doc_id = item.id
        record = item.to_json()
        record["description"] = self._sanitize(
            html_to_text(item.description), doc_id, "description", run_id
        )
        for ep in record["episodes"]:
            ep["description"] = self._sanitize(
                html_to_text(ep.get("description")), doc_id, f"episode {ep['id']}", run_id
            )
        topics: list[str] = []
        if self.llm is not None:
            try:
                topics = self.llm.podcast_topics(
                    title=item.title,
                    hosts=item.authors,
                    description=str(record["description"]),
                    genres=item.genres,
                    episode_titles=[e.title for e in item.episodes],
                    taxonomy=self.taxonomy,
                )
            except LLMError as e:
                logger.warning("Podcast analysis failed for item %s: %s", doc_id, e)
        record["topics"] = topics
        text = "\n\n".join(
            [str(record["description"])]
            + [f"{e['title']}\n{e['description']}" for e in record["episodes"]]
        )
        self.db.upsert_staging(
            StagedDocument(
                source=SOURCE,
                source_doc_id=doc_id,
                title=item.title,
                url=item_url(self.settings.link_base_url, doc_id),
                tier=TIER,
                raw_content=record,
                extracted_text=text,
                content_hash=raw_hash,
                source_updated_at=str(item.updated_at),
                flags={"secret_redacted": "[REDACTED]" in text},
            ),
            run_id=run_id,
        )

    # -- works ------------------------------------------------------------------

    def _candidate_index(self) -> dict[str, list[JSON]]:
        if self._candidates is None:
            index: dict[str, list[JSON]] = defaultdict(list)
            for row in self.db.match_candidates(exclude_source=SOURCE):
                surname = author_surname(row["authors"][0] if row["authors"] else None)
                index[surname].append(row)
            self._candidates = dict(index)
        return self._candidates

    def _resolve_work(
        self,
        title: str,
        authors: list[str],
        isbn13: str | None,
        series_id: str | None,
        series_index: float | None,
    ) -> str:
        natural = work_id_for(title=title, authors=authors, isbn13=isbn13)
        canonical = self.db.merge_target(natural)
        if canonical != natural or self.db.get_work(natural) is not None:
            return canonical
        surname = author_surname(authors[0] if authors else None)
        distinct = self.db.distinct_pairs(natural)

        def same_series(row: JSON) -> bool:
            return bool(
                series_id
                and row.get("series_id") == series_id
                and (series_index is None or row.get("series_index") == series_index)
            )

        best: tuple[float, bool, JSON] | None = None
        for row in self._candidate_index().get(surname, []) if surname else []:
            if row["work_id"] in distinct:
                continue
            score = similarity(title, row["title"])
            key = (score, same_series(row))
            if best is None or key > (best[0], best[1]):
                best = (score, key[1], row)
        if best is None:
            return natural
        score, series_match, row = best
        if score >= AUTO_MERGE_SIMILARITY or (score >= REVIEW_SIMILARITY and series_match):
            decided_by = "auto" if score >= AUTO_MERGE_SIMILARITY else "auto-series"
            logger.info("Merging %s into %s (similarity %.2f)", natural, row["work_id"], score)
            self.db.record_merge(
                from_work_id=natural,
                to_work_id=row["work_id"],
                decided_by=decided_by,
                similarity=score,
            )
            return str(row["work_id"])
        if score >= REVIEW_SIMILARITY:
            self.db.queue_review(
                kind="work_merge",
                left_id=natural,
                right_id=str(row["work_id"]),
                similarity=score,
                details={"left_title": title, "right_title": row["title"]},
            )
        return natural

    def _delete_items(
        self,
        item_ids: list[str],
        media: dict[str, tuple[str | None, str | None]],
        counts: RunCounts,
    ) -> set[str]:
        podcasts = [i for i in item_ids if (media.get(i) or (None, None))[1] == "podcast"]
        books = [i for i in item_ids if i not in set(podcasts)]
        works = set(self.db.delete_manifestations(SOURCE, books)) if books else set()
        self.db.delete_catalog_items(SOURCE, books)
        self.db.delete_enrichment(SOURCE, item_ids)
        counts.deleted += self.db.delete_staging(SOURCE, item_ids)
        for work_id in works:
            self.db.set_work_topics(work_id)
        self.db.commit()
        self.member.delete_item_points(PODCAST_SOURCE, podcasts, keep=set())
        logger.info("Deleted %s items no longer in Audiobookshelf", len(item_ids))
        return works

    # -- indexing ---------------------------------------------------------------

    def _embed_points(
        self, store: QdrantStore, items: list[tuple[str, JSON]], *, force: bool
    ) -> int:
        """Upsert (point id, payload) items, embedding only changed text.

        A point whose text is unchanged but whose other payload fields changed
        (year, languages, url, manifestations) gets its payload rewritten
        without re-embedding.
        """
        if not items:
            return 0
        for _, payload in items:
            payload["payload_hash"] = payload_hash(payload)
        existing = {} if force else store.existing_state(i for i, _ in items)
        todo = [
            (i, p)
            for i, p in items
            if force or (existing.get(i) or (None,))[0] != p["content_hash"]
        ]
        same_text = [
            (i, p)
            for i, p in items
            if not force and i in existing and existing[i][0] == p["content_hash"]
        ]
        stale = [(i, p) for i, p in same_text if existing[i][1] != p["payload_hash"]]
        current = [i for i, p in same_text if existing[i][1] == p["payload_hash"]]
        store.replace_payloads(stale)
        if current:
            store.set_payload(current, {"sync_run_id": items[0][1]["sync_run_id"]})
        size = self.settings.batch_size
        for start in range(0, len(todo), size):
            self.beat()
            batch = todo[start : start + size]
            texts = [
                f"{p['context']}\n\n{p['text']}" if p["context"] else p["text"] for _, p in batch
            ]
            vectors = self.embedder.encode(texts, kind="document")
            store.upsert(
                [
                    qm.PointStruct(
                        id=pid,
                        vector={"dense": vector, "sparse": sparse_vector(text)},
                        payload=payload,
                    )
                    for (pid, payload), text, vector in zip(batch, texts, vectors, strict=True)
                ]
            )
        return len(todo)

    def _index(
        self,
        mode: Mode,
        run_id: str,
        affected: set[str],
        changed_podcasts: set[str],
        *,
        allow_delete: bool,
    ) -> None:
        force = mode == "reindex"
        if mode == "incremental":
            work_ids = sorted(affected)
            podcasts: list[str] | None = sorted(changed_podcasts)
        else:
            work_ids = sorted(set(self.db.work_ids_for_source(SOURCE)) | affected)
            podcasts = None
        self.index_catalog(work_ids, run_id, force=force)
        self.index_podcasts(podcasts, run_id, force=force)
        self.index_overview(run_id, force=force)
        if mode in ("full", "reindex") and allow_delete:
            for source in (PODCAST_SOURCE, SOURCE):
                removed = self.member.delete_stale(source, run_id)
                if removed:
                    logger.info("Deleted %s stale %s points", removed, source)

    def index_catalog(self, work_ids: list[str], run_id: str, *, force: bool = False) -> None:
        items: list[tuple[str, JSON]] = []
        gone: list[str] = []
        for work_id in work_ids:
            self.beat()
            pid = point_id(CATALOG_SOURCE, work_id, 0)
            work = self.db.get_work(work_id)
            manifestations = self.db.manifestations(work_id) if work else []
            if work is None or not manifestations:
                gone.append(pid)
                if work is not None:
                    self.db.delete_work_if_empty(work_id)
                continue
            items.append((pid, self._catalog_payload(work, manifestations, run_id)))
        self.db.commit()
        self.catalog.delete_ids(gone)
        embedded = self._embed_points(self.catalog, items, force=force)
        logger.info(
            "Catalog: %s works checked, %s embedded, %s removed", len(items), embedded, len(gone)
        )

    def _catalog_payload(self, work: JSON, manifestations: list[JSON], run_id: str) -> JSON:
        series = self.db.series_name(work["series_id"]) if work.get("series_id") else None
        text = catalog_text(work, manifestations, series)
        details = [m.get("details") or {} for m in manifestations]
        authors = list(work.get("authors") or [])
        context = (
            f"Hackerspace library catalog record for {work['title']!r}"
            + (f" by {', '.join(authors)}" if authors else "")
            + "."
        )
        links = self.linker.link(text)
        added = sorted(d["date_added"] for d in details if d.get("date_added"))
        updated = sorted(d["updated_at"] for d in details if d.get("updated_at"))
        payload = base_payload(
            source=CATALOG_SOURCE,
            doc_id=str(work["work_id"]),
            chunk_index=0,
            title=str(work["title"]),
            text=text,
            context=context,
            url=next((m["url"] for m in manifestations if m.get("url")), None),
            run_id=run_id,
            created_at=added[0] if added else None,
            updated_at=updated[-1] if updated else None,
            lang=next((d["languages"][0] for d in details if d.get("languages")), None),
            topics=list(work.get("topics") or []),
            work_ids=[str(work["work_id"])],
            links=links.ids,
        )
        payload.update(
            {
                "work_id": work["work_id"],
                "authors": authors,
                "series": series,
                "series_id": work.get("series_id"),
                "series_index": work.get("series_index"),
                "year": next((d["year"] for d in details if d.get("year")), None),
                "languages": sorted({lang for d in details for lang in d.get("languages") or []}),
                "tags": sorted({t for d in details for t in d.get("tags") or []}),
                "formats": sorted({str(m["format"]) for m in manifestations}),
                "narrators": sorted({n for d in details for n in d.get("narrators") or []}),
                "duration_seconds": next(
                    (d["duration_seconds"] for d in details if d.get("duration_seconds")), None
                ),
                "manifestations": [
                    {
                        "format": m["format"],
                        "source": m["source"],
                        "source_id": m["source_id"],
                        "url": m.get("url"),
                        "location": (m.get("details") or {}).get("location"),
                        "files": (m.get("details") or {}).get("formats", []),
                        "cover_path": (m.get("details") or {}).get("cover_path"),
                        "narrators": (m.get("details") or {}).get("narrators", []),
                        "duration_seconds": (m.get("details") or {}).get("duration_seconds"),
                    }
                    for m in manifestations
                ],
                "date_added": added[0] if added else None,
                "description_generated": any(d.get("good_for_generated") for d in details),
            }
        )
        return payload

    def _staged_podcasts(self, item_ids: list[str] | None) -> list[JSON]:
        rows = self.db.staged_documents(SOURCE, item_ids)
        return [r for r in rows if (r["raw_content"] or {}).get("media_type") == "podcast"]

    def index_podcasts(self, item_ids: list[str] | None, run_id: str, *, force: bool) -> None:
        rows = self._staged_podcasts(item_ids)
        embedded = 0
        for row in rows:
            items = podcast_points(
                row["raw_content"],
                url=item_url(self.settings.link_base_url, row["source_doc_id"]),
                run_id=run_id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                link=self._link,
            )
            embedded += self._embed_points(self.member, items, force=force)
            # Episodes removed from the podcast, and chunks a shorter text no longer has.
            self.member.delete_item_points(
                PODCAST_SOURCE, [row["source_doc_id"]], keep={i for i, _ in items}
            )
        if rows:
            logger.info("Podcasts: %s checked, %s points embedded", len(rows), embedded)

    def index_overview(self, run_id: str, *, force: bool = False) -> None:
        sections = build_overview(
            self.db.catalog_rows(SOURCE),
            [r["raw_content"] for r in self._staged_podcasts(None)],
            library_url=self.settings.link_base_url,
            access_info=self.settings.library_access_info,
            apps_info=self.settings.library_apps_info,
        )
        document = "\n\n".join(f"{h}\n{b}" for h, b in sections)
        title = sections[0][0]
        now = datetime.now(UTC).isoformat()
        # Chunk 0 is the document-level summary point; then one per section.
        pieces: list[tuple[list[str], str]] = [([title], f"{title}\n{sections[0][1]}")]
        for heading, body in sections[1:]:
            for chunk in split_text(body, self.settings.chunk_size, self.settings.chunk_overlap):
                pieces.append(([title, heading], f"{heading}\n{chunk}"))
        items: list[tuple[str, JSON]] = []
        for index, (path, text) in enumerate(pieces):
            context = self._context(
                title, document, text, fallback=f"Part of the {title} overview."
            )
            items.append(
                (
                    point_id(SOURCE, OVERVIEW_DOC_ID, index),
                    base_payload(
                        source=SOURCE,
                        doc_id=OVERVIEW_DOC_ID,
                        chunk_index=index,
                        title=title,
                        text=text,
                        context=context,
                        url=self.settings.link_base_url,
                        run_id=run_id,
                        created_at=None,
                        updated_at=now,
                        heading_path=path,
                        lang="en",
                        topics=["reference", "hackerspace-and-community"],
                        links=self._link(text),
                    ),
                )
            )
        existing = self.member.existing_hashes(i for i, _ in items)
        if force or any(existing.get(i) != p["content_hash"] for i, p in items):
            self.member.delete_docs(SOURCE, [OVERVIEW_DOC_ID])
            self._embed_points(self.member, items, force=True)
        else:
            self.member.set_payload([i for i, _ in items], {"sync_run_id": run_id})

    def _context(self, title: str, document: str, chunk: str, *, fallback: str) -> str:
        if self.llm is None:
            return fallback
        self.beat()
        try:
            return self.llm.chunk_context(document_title=title, document=document, chunk=chunk)
        except LLMError as e:
            logger.warning("Chunk context failed: %s", e)
            return fallback


def build_pipeline(
    settings: Settings,
    db: Database,
    client: AbsSource,
    stop: threading.Event | None,
    beat: Callable[[], None] | None = None,
) -> Pipeline:
    llm = (
        LLM(
            model=settings.llm_model,
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            cache=db,
        )
        if settings.llm_model
        else None
    )
    if llm is None:
        logger.warning("LLM_MODEL is not set: no topics, summaries or chunk context this run")
    enricher = (
        Enricher(
            timeout_seconds=settings.enrichment_timeout_seconds,
            google_api_key=settings.google_books_api_key,
        )
        if settings.enrichment_enabled
        else None
    )
    return Pipeline(
        settings=settings,
        db=db,
        embedder=build_embedder(settings),
        qdrant=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        client=client,
        llm=llm,
        enricher=enricher,
        stop=stop,
        beat=beat,
    )


def run(
    settings: Settings,
    mode: Mode,
    stop: threading.Event | None = None,
    beat: Callable[[], None] | None = None,
) -> RunResult:
    with (
        Database.connect(settings.database_url) as db,
        AbsClient(
            base_url=settings.abs_url,
            token=settings.abs_api_token,
            timeout_seconds=settings.request_timeout_seconds,
            page_size=settings.abs_page_size,
        ) as client,
    ):
        built = build_pipeline(settings, db, client, stop, beat)
        try:
            return built.run(mode)
        finally:
            built.close()


def last_full_run(settings: Settings) -> datetime | None:
    """When the last full run finished: the serve loop's cue for the next one."""
    with Database.connect(settings.database_url) as db:
        return db.last_full_run_at(SOURCE)


def status(settings: Settings) -> JSON:
    """Recent runs, quarantine and catalog counts: the source's status page."""
    with Database.connect(settings.database_url) as db:
        media = db.staged_media(SOURCE)
        return {
            "source": SOURCE,
            "runs": db.recent_runs(SOURCE),
            "quarantined": db.quarantine_count([SOURCE]),
            "catalog_audiobooks": db.catalog_count(source=SOURCE, format="audiobook"),
            "catalog_ebooks": db.catalog_count(source=SOURCE, format="ebook"),
            "podcasts": sum(1 for _, kind in media.values() if kind == "podcast"),
        }
