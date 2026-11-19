"""PostgreSQL: staging store, entity registry, catalog table, run status.

Plain SQL with psycopg 3. The schema is in ``migrations/`` and is applied by
``migrate()``, which every command calls first.
"""

from __future__ import annotations

import contextlib
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

import psycopg
from alembic import command
from alembic.config import Config
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Any: rows are dicts of column name to whatever type the column holds.
Row = dict[str, Any]


def sqlalchemy_url(database_url: str) -> str:
    """DATABASE_URL as SQLAlchemy wants it, using the psycopg 3 driver."""
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix) :]
    return database_url


def migrate(database_url: str) -> None:
    """Bring the schema up to date (``alembic upgrade head``)."""
    config = Config()
    config.set_main_option(
        "script_location", str(files("audiobookshelf_rag_ingester") / "migrations")
    )
    # Alembic's config uses %-interpolation; a password may contain %.
    config.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url).replace("%", "%%"))
    command.upgrade(config, "head")


@dataclass(frozen=True)
class StagedDocument:
    source: str
    source_doc_id: str
    title: str | None
    url: str | None
    tier: str
    # Any: normalized source content, stored as JSON.
    raw_content: dict[str, Any]
    extracted_text: str
    content_hash: str
    source_updated_at: str | None
    # Any: flags are free-form JSON (e.g. {"secret_redacted": true}).
    flags: dict[str, Any] = field(default_factory=dict)
    indexable: bool = True


@dataclass
class RunCounts:
    seen: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0
    quarantined: int = 0
    errors: list[str] = field(default_factory=list)


class Database:
    def __init__(self, conn: psycopg.Connection[Row]) -> None:
        self.conn = conn

    @classmethod
    @contextlib.contextmanager
    def connect(cls, database_url: str) -> Iterator[Database]:
        with psycopg.connect(database_url, row_factory=dict_row, connect_timeout=30) as conn:
            yield cls(conn)

    def commit(self) -> None:
        self.conn.commit()

    def _all(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> list[Row]:
        with self.conn.cursor() as cur:
            cur.execute(query, params)  # type: ignore[arg-type]  # psycopg types params loosely
            return cur.fetchall()

    def _one(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> Row | None:
        rows = self._all(query, params)
        return rows[0] if rows else None

    def _exec(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> int:
        with self.conn.cursor() as cur:
            cur.execute(query, params)  # type: ignore[arg-type]  # psycopg types params loosely
            return cur.rowcount

    # -- locks and runs -----------------------------------------------------

    @contextlib.contextmanager
    def source_lock(self, source: str) -> Iterator[bool]:
        """A session advisory lock for one source; yields False if it's held."""
        key = zlib.crc32(f"audiobookshelf-rag-ingester:{source}".encode())
        row = self._one("SELECT pg_try_advisory_lock(%s) AS locked", (key,))
        self.commit()
        locked = bool(row and row["locked"])
        try:
            yield locked
        finally:
            if locked:
                self._exec("SELECT pg_advisory_unlock(%s)", (key,))
                self.commit()

    def start_run(self, *, run_id: str, source: str, mode: str) -> None:
        self._exec(
            "INSERT INTO sync_runs (run_id, source, mode, started_at) VALUES (%s, %s, %s, %s)",
            (run_id, source, mode, datetime.now(UTC)),
        )
        self.commit()

    def finish_run(
        self, *, run_id: str, counts: RunCounts, ok: bool, message: str | None = None
    ) -> None:
        self._exec(
            """
            UPDATE sync_runs SET finished_at = %s, ok = %s, seen = %s, added = %s,
                updated = %s, deleted = %s, quarantined = %s, errors = %s, message = %s
            WHERE run_id = %s
            """,
            (
                datetime.now(UTC),
                ok,
                counts.seen,
                counts.added,
                counts.updated,
                counts.deleted,
                counts.quarantined,
                Jsonb(counts.errors[:100]),
                message,
                run_id,
            ),
        )
        self.commit()

    def previous_full_run_seen(self, source: str, exclude_run_id: str) -> int | None:
        row = self._one(
            """
            SELECT seen FROM sync_runs
            WHERE source = %s AND mode = 'full' AND ok AND run_id <> %s
            ORDER BY started_at DESC LIMIT 1
            """,
            (source, exclude_run_id),
        )
        return int(row["seen"]) if row else None

    def recent_runs(self, source: str, limit: int = 10) -> list[Row]:
        return self._all(
            "SELECT * FROM sync_runs WHERE source = %s ORDER BY started_at DESC LIMIT %s",
            (source, limit),
        )

    def quarantine_count(self, sources: Iterable[str]) -> int:
        row = self._one(
            "SELECT count(*) AS n FROM quarantine WHERE source = ANY(%s)", (list(sources),)
        )
        return int(row["n"]) if row else 0

    def add_quarantine(
        # Any: detail is free-form JSON.
        self,
        *,
        source: str,
        source_doc_id: str,
        reason: str,
        detail: dict[str, Any],
        run_id: str,
    ) -> None:
        self._exec(
            "INSERT INTO quarantine (source, source_doc_id, reason, detail, sync_run_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (source, source_doc_id, reason, Jsonb(detail), run_id),
        )

    # -- staging ------------------------------------------------------------

    def staged_versions(self, source: str) -> dict[str, tuple[str | None, str]]:
        """source_doc_id -> (source_updated_at, content_hash) for one source."""
        rows = self._all(
            "SELECT source_doc_id, source_updated_at, content_hash FROM staging_documents "
            "WHERE source = %s",
            (source,),
        )
        return {
            row["source_doc_id"]: (row["source_updated_at"], row["content_hash"]) for row in rows
        }

    def upsert_staging(self, doc: StagedDocument, *, run_id: str) -> None:
        self._exec(
            """
            INSERT INTO staging_documents (source, source_doc_id, title, url, tier, raw_content,
                extracted_text, content_hash, flags, indexable, source_updated_at, fetched_at,
                sync_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_doc_id) DO UPDATE SET
                title = EXCLUDED.title, url = EXCLUDED.url, tier = EXCLUDED.tier,
                raw_content = EXCLUDED.raw_content, extracted_text = EXCLUDED.extracted_text,
                content_hash = EXCLUDED.content_hash, flags = EXCLUDED.flags,
                indexable = EXCLUDED.indexable, source_updated_at = EXCLUDED.source_updated_at,
                fetched_at = EXCLUDED.fetched_at, sync_run_id = EXCLUDED.sync_run_id
            """,
            (
                doc.source,
                doc.source_doc_id,
                doc.title,
                doc.url,
                doc.tier,
                Jsonb(doc.raw_content),
                doc.extracted_text,
                doc.content_hash,
                Jsonb(doc.flags),
                doc.indexable,
                doc.source_updated_at,
                datetime.now(UTC),
                run_id,
            ),
        )

    def staged_media(self, source: str) -> dict[str, tuple[str | None, str | None]]:
        """source_doc_id -> (library_id, media_type) from the staged records."""
        rows = self._all(
            "SELECT source_doc_id, raw_content->>'library_id' AS library_id, "
            "raw_content->>'media_type' AS media_type FROM staging_documents WHERE source = %s",
            (source,),
        )
        return {row["source_doc_id"]: (row["library_id"], row["media_type"]) for row in rows}

    def forget_versions(self, source: str, doc_ids: list[str]) -> None:
        """Make the next incremental run fetch these documents again (indexing failed)."""
        self._exec(
            "UPDATE staging_documents SET source_updated_at = NULL "
            "WHERE source = %s AND source_doc_id = ANY(%s)",
            (source, doc_ids),
        )

    def touch_staging(self, source: str, doc_ids: list[str], run_id: str) -> None:
        self._exec(
            "UPDATE staging_documents SET sync_run_id = %s "
            "WHERE source = %s AND source_doc_id = ANY(%s)",
            (run_id, source, doc_ids),
        )

    def delete_staging(self, source: str, doc_ids: list[str]) -> int:
        return self._exec(
            "DELETE FROM staging_documents WHERE source = %s AND source_doc_id = ANY(%s)",
            (source, doc_ids),
        )

    def staged_documents(self, source: str, doc_ids: list[str] | None = None) -> list[Row]:
        if doc_ids is None:
            return self._all(
                "SELECT * FROM staging_documents WHERE source = %s AND indexable "
                "ORDER BY source_doc_id",
                (source,),
            )
        return self._all(
            "SELECT * FROM staging_documents WHERE source = %s AND indexable "
            "AND source_doc_id = ANY(%s) ORDER BY source_doc_id",
            (source, doc_ids),
        )

    # -- caches ---------------------------------------------------------------

    # Any: cached outputs are JSON.
    def llm_cache_get(self, task: str, model: str, input_hash: str) -> Any:
        row = self._one(
            "SELECT output FROM llm_cache WHERE task = %s AND model = %s AND input_hash = %s",
            (task, model, input_hash),
        )
        return row["output"] if row else None

    # Any: cached outputs are JSON.
    def llm_cache_put(self, task: str, model: str, input_hash: str, output: Any) -> None:
        self._exec(
            """
            INSERT INTO llm_cache (task, model, input_hash, output) VALUES (%s, %s, %s, %s)
            ON CONFLICT (task, model, input_hash) DO UPDATE SET output = EXCLUDED.output,
                created_at = now()
            """,
            (task, model, input_hash, Jsonb(output)),
        )

    def enrichment_get(self, source: str, source_doc_id: str) -> list[Row]:
        return self._all(
            "SELECT * FROM enrichment WHERE source = %s AND source_doc_id = %s ORDER BY origin",
            (source, source_doc_id),
        )

    def enrichment_put(
        self,
        *,
        source: str,
        source_doc_id: str,
        origin: str,
        lookup_key: str,
        matched: bool,
        confidence: float | None,
        # Any: enriched fields are JSON.
        fields: dict[str, Any],
    ) -> None:
        self._exec(
            """
            INSERT INTO enrichment (source, source_doc_id, origin, lookup_key, matched,
                confidence, fields, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_doc_id, origin) DO UPDATE SET
                lookup_key = EXCLUDED.lookup_key, matched = EXCLUDED.matched,
                confidence = EXCLUDED.confidence, fields = EXCLUDED.fields,
                fetched_at = EXCLUDED.fetched_at
            """,
            (
                source,
                source_doc_id,
                origin,
                lookup_key,
                matched,
                confidence,
                Jsonb(fields),
                datetime.now(UTC),
            ),
        )

    def delete_enrichment(self, source: str, doc_ids: list[str]) -> None:
        self._exec(
            "DELETE FROM enrichment WHERE source = %s AND source_doc_id = ANY(%s)",
            (source, doc_ids),
        )

    # -- registry -------------------------------------------------------------

    def upsert_series(self, series_id: str, name: str) -> None:
        self._exec(
            """
            INSERT INTO registry_series (series_id, name) VALUES (%s, %s)
            ON CONFLICT (series_id) DO UPDATE SET
                aliases = CASE WHEN registry_series.name = EXCLUDED.name
                               OR EXCLUDED.name = ANY(registry_series.aliases)
                          THEN registry_series.aliases
                          ELSE registry_series.aliases || EXCLUDED.name END,
                updated_at = now()
            """,
            (series_id, name),
        )

    def get_work(self, work_id: str) -> Row | None:
        return self._one("SELECT * FROM registry_works WHERE work_id = %s", (work_id,))

    def upsert_work(
        self,
        *,
        work_id: str,
        title: str,
        authors: list[str],
        isbn13: str | None,
        series_id: str | None,
        series_index: float | None,
    ) -> None:
        """Create the work, or fill in what it lacks; never overwrite its title."""
        self._exec(
            """
            INSERT INTO registry_works (work_id, title, authors, isbn13, series_id, series_index,
                aliases)
            VALUES (%s, %s, %s, %s, %s, %s, ARRAY[%s])
            ON CONFLICT (work_id) DO UPDATE SET
                authors = CASE WHEN cardinality(registry_works.authors) = 0
                               THEN EXCLUDED.authors ELSE registry_works.authors END,
                isbn13 = COALESCE(registry_works.isbn13, EXCLUDED.isbn13),
                series_id = COALESCE(EXCLUDED.series_id, registry_works.series_id),
                series_index = COALESCE(EXCLUDED.series_index, registry_works.series_index),
                aliases = CASE WHEN EXCLUDED.title = registry_works.title
                               OR EXCLUDED.title = ANY(registry_works.aliases)
                          THEN registry_works.aliases
                          ELSE registry_works.aliases || EXCLUDED.title END,
                updated_at = now()
            """,
            (work_id, title, authors, isbn13, series_id, series_index, title),
        )

    def set_work_topics(self, work_id: str) -> None:
        """Topics of a work: the union of its manifestations' topics."""
        self._exec(
            """
            UPDATE registry_works SET topics = COALESCE((
                SELECT array_agg(DISTINCT t ORDER BY t)
                FROM registry_manifestations m,
                     jsonb_array_elements_text(COALESCE(m.details->'topics', '[]'::jsonb)) t
                WHERE m.work_id = %s), '{}'), updated_at = now()
            WHERE work_id = %s
            """,
            (work_id, work_id),
        )

    def upsert_manifestation(
        self,
        *,
        source: str,
        source_id: str,
        work_id: str,
        format: str,
        url: str | None,
        # Any: details are JSON.
        details: dict[str, Any],
    ) -> str | None:
        """Attach a manifestation to a work; returns the work it was on before."""
        previous = self._one(
            "SELECT work_id FROM registry_manifestations WHERE source = %s AND source_id = %s",
            (source, source_id),
        )
        self._exec(
            """
            INSERT INTO registry_manifestations (source, source_id, work_id, format, url, details)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_id) DO UPDATE SET work_id = EXCLUDED.work_id,
                format = EXCLUDED.format, url = EXCLUDED.url, details = EXCLUDED.details,
                updated_at = now()
            """,
            (source, source_id, work_id, format, url, Jsonb(details)),
        )
        return str(previous["work_id"]) if previous else None

    def delete_manifestations(self, source: str, source_ids: list[str]) -> list[str]:
        """Remove manifestations; returns the works they belonged to."""
        rows = self._all(
            "DELETE FROM registry_manifestations WHERE source = %s AND source_id = ANY(%s) "
            "RETURNING work_id",
            (source, source_ids),
        )
        return sorted({str(row["work_id"]) for row in rows})

    def manifestations(self, work_id: str) -> list[Row]:
        return self._all(
            "SELECT * FROM registry_manifestations WHERE work_id = %s ORDER BY format, source, "
            "source_id",
            (work_id,),
        )

    def work_ids_for_source(self, source: str) -> list[str]:
        rows = self._all(
            "SELECT DISTINCT work_id FROM registry_manifestations WHERE source = %s "
            "ORDER BY work_id",
            (source,),
        )
        return [str(row["work_id"]) for row in rows]

    def delete_work_if_empty(self, work_id: str) -> bool:
        return (
            self._exec(
                "DELETE FROM registry_works w WHERE work_id = %s AND NOT EXISTS "
                "(SELECT 1 FROM registry_manifestations m WHERE m.work_id = w.work_id)",
                (work_id,),
            )
            > 0
        )

    def series_name(self, series_id: str) -> str | None:
        row = self._one("SELECT name FROM registry_series WHERE series_id = %s", (series_id,))
        return str(row["name"]) if row else None

    def merge_target(self, work_id: str) -> str:
        """Follow 'merge' decisions from a computed work ID to its canonical work."""
        seen = {work_id}
        current = work_id
        while True:
            row = self._one(
                "SELECT to_work_id FROM registry_merges WHERE from_work_id = %s "
                "AND decision = 'merge' ORDER BY decided_at DESC LIMIT 1",
                (current,),
            )
            if row is None or row["to_work_id"] in seen:
                return current
            current = str(row["to_work_id"])
            seen.add(current)

    def distinct_pairs(self, work_id: str) -> set[str]:
        rows = self._all(
            "SELECT from_work_id, to_work_id FROM registry_merges WHERE decision = 'distinct' "
            "AND (from_work_id = %s OR to_work_id = %s)",
            (work_id, work_id),
        )
        return {
            str(row["to_work_id"] if row["from_work_id"] == work_id else row["from_work_id"])
            for row in rows
        }

    def record_merge(
        self, *, from_work_id: str, to_work_id: str, decided_by: str, similarity: float
    ) -> None:
        self._exec(
            """
            INSERT INTO registry_merges (from_work_id, to_work_id, decision, decided_by,
                similarity)
            VALUES (%s, %s, 'merge', %s, %s)
            ON CONFLICT (from_work_id, to_work_id) DO NOTHING
            """,
            (from_work_id, to_work_id, decided_by, similarity),
        )

    def queue_review(
        self,
        *,
        kind: str,
        left_id: str,
        right_id: str,
        similarity: float,
        # Any: details are JSON.
        details: dict[str, Any],
    ) -> None:
        self._exec(
            """
            INSERT INTO registry_review_queue (kind, left_id, right_id, similarity, details)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kind, left_id, right_id) DO NOTHING
            """,
            (kind, left_id, right_id, similarity, Jsonb(details)),
        )

    def match_candidates(self, *, exclude_source: str) -> list[Row]:
        """Works with no manifestation from ``exclude_source``: other sources' books."""
        return self._all(
            """
            SELECT w.work_id, w.title, w.authors, w.series_id, w.series_index
            FROM registry_works w
            WHERE NOT EXISTS (SELECT 1 FROM registry_manifestations m
                              WHERE m.work_id = w.work_id AND m.source = %s)
            """,
            (exclude_source,),
        )

    def aliases(self) -> list[Row]:
        return self._all("SELECT entity_type, entity_id, alias FROM entity_aliases")

    # -- catalog table --------------------------------------------------------

    # Any: catalog columns hold text, arrays, numbers and timestamps.
    def upsert_catalog_item(self, item: dict[str, Any]) -> None:
        columns = list(item)
        placeholders = ", ".join(["%s"] * len(columns))
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in columns if c not in {"source", "source_id"}
        )
        # Column names come from catalog_item() in pipeline.py, never from input.
        self._exec(
            f"INSERT INTO catalog_items ({', '.join(columns)}, updated_at) "  # noqa: S608 - fixed names
            f"VALUES ({placeholders}, now()) "
            f"ON CONFLICT (source, source_id) DO UPDATE SET {updates}, updated_at = now()",
            [item[c] for c in columns],
        )

    def delete_catalog_items(self, source: str, source_ids: list[str]) -> None:
        self._exec(
            "DELETE FROM catalog_items WHERE source = %s AND source_id = ANY(%s)",
            (source, source_ids),
        )

    def catalog_count(self, *, source: str | None = None, format: str | None = None) -> int:
        row = self._one(
            "SELECT count(*) AS n FROM catalog_items "
            "WHERE (%(source)s::text IS NULL OR source = %(source)s) "
            "AND (%(format)s::text IS NULL OR format = %(format)s)",
            {"source": source, "format": format},
        )
        return int(row["n"]) if row else 0

    # Any: rows are JSON-like.
    def catalog_rows(self, source: str) -> list[Row]:
        return self._all("SELECT * FROM catalog_items WHERE source = %s", (source,))
