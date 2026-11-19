import subprocess
import sys
import uuid
from urllib.parse import urlsplit

import psycopg

from audiobookshelf_rag_ingester.db import (
    Database,
    RunCounts,
    StagedDocument,
    migrate,
    sqlalchemy_url,
)


def test_sqlalchemy_url() -> None:
    assert sqlalchemy_url("postgresql://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert sqlalchemy_url("postgres://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert sqlalchemy_url("sqlite://") == "sqlite://"


def test_migrate_is_idempotent(database_url: str) -> None:
    migrate(database_url)
    migrate(database_url)


def test_concurrent_migrations_on_a_fresh_database(database_url: str) -> None:
    # Hourly and nightly runs both migrate first; on an empty database they
    # must not race to create the schema.
    name = f"cw2r_fresh_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    fresh_url = urlsplit(database_url)._replace(path=f"/{name}").geturl()
    # Separate processes, like real runs: Alembic's state is per process.
    script = "import sys; from audiobookshelf_rag_ingester.db import migrate; migrate(sys.argv[1])"
    try:
        procs = [
            # S603: this interpreter running a fixed script.
            subprocess.Popen([sys.executable, "-c", script, fresh_url], stderr=subprocess.PIPE)  # noqa: S603
            for _ in range(3)
        ]
        for proc in procs:
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, err.decode()
        with Database.connect(fresh_url) as fresh:
            assert fresh.aliases() == []
    finally:
        with psycopg.connect(database_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def test_staging_round_trip(db: Database) -> None:
    doc = StagedDocument(
        source="calibre",
        source_doc_id="1",
        title="T",
        url=None,
        tier="member",
        raw_content={"a": 1},
        extracted_text="text",
        content_hash="h",
        source_updated_at="2024",
    )
    db.upsert_staging(doc, run_id="r1")
    db.upsert_staging(doc, run_id="r2")
    assert db.staged_versions("calibre") == {"1": ("2024", "h")}
    assert [r["sync_run_id"] for r in db.staged_documents("calibre")] == ["r2"]
    assert db.staged_documents("calibre", ["2"]) == []
    assert db.delete_staging("calibre", ["1"]) == 1


def test_runs_and_previous_full_run(db: Database) -> None:
    db.start_run(run_id="a", source="calibre", mode="full")
    db.finish_run(run_id="a", counts=RunCounts(seen=10), ok=True)
    db.start_run(run_id="b", source="calibre", mode="full")
    assert db.previous_full_run_seen("calibre", "b") == 10
    assert db.previous_full_run_seen("other", "b") is None
    assert [r["run_id"] for r in db.recent_runs("calibre")] == ["b", "a"]


def test_registry_merges_and_series(db: Database) -> None:
    db.upsert_series("series:s", "S")
    db.upsert_series("series:s", "S Again")
    db.upsert_series("series:s", "S Again")
    assert db.series_name("series:s") == "S"
    row = db.conn.execute("SELECT aliases FROM registry_series").fetchone()
    assert row == {"aliases": ["S Again"]}
    assert db.series_name("series:none") is None

    db.upsert_work(
        work_id="w", title="T", authors=[], isbn13=None, series_id=None, series_index=None
    )
    db.upsert_work(
        work_id="w", title="T2", authors=["A"], isbn13="1", series_id="series:s", series_index=2
    )
    work = db.get_work("w")
    assert work is not None
    assert (work["title"], work["authors"], work["aliases"]) == ("T", ["A"], ["T", "T2"])

    db.record_merge(from_work_id="a", to_work_id="b", decided_by="human", similarity=1.0)
    db.record_merge(from_work_id="b", to_work_id="a", decided_by="human", similarity=1.0)
    assert db.merge_target("a") == "b"  # cycles stop
    db.conn.execute(
        "INSERT INTO registry_merges (from_work_id, to_work_id, decision, decided_by) "
        "VALUES ('x', 'y', 'distinct', 'human')"
    )
    assert db.distinct_pairs("y") == {"x"}
    assert db.distinct_pairs("x") == {"y"}
    assert db.delete_work_if_empty("w") is True
