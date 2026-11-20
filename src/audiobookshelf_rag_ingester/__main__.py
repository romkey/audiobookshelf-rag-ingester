"""Command line: `audiobookshelf-rag-ingester` or `python -m audiobookshelf_rag_ingester`.

Syncs the Audiobookshelf audiobook and podcast libraries into the
member-tier RAG pipeline: the Postgres staging store, entity registry and
catalog table, and the Qdrant hs_catalog and hs_member collections.

  audiobookshelf-rag-ingester                         # serve: scan, sleep, repeat (default)
  audiobookshelf-rag-ingester run --mode incremental  # one scan of changed and deleted items
  audiobookshelf-rag-ingester run --mode full         # one scan of everything, then reconcile
  audiobookshelf-rag-ingester run --mode reindex      # rebuild points from staging
  audiobookshelf-rag-ingester migrate                 # only create or update the schema
  audiobookshelf-rag-ingester status                  # recent runs and counts, as JSON
  audiobookshelf-rag-ingester healthcheck             # exit 0 if serve is alive

`serve` sleeps SCAN_INTERVAL_MINUTES (default 60) between scans. Scans are
incremental, plus a full one when none has finished in the last
FULL_SCAN_INTERVAL_HOURS (default 24), so the first scan is full. Every
command except healthcheck first brings the database schema up to date.
"""

import argparse
import json
import logging
import signal
import sys
import tempfile
import threading
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from types import FrameType

from dotenv import load_dotenv

from audiobookshelf_rag_ingester import __version__, pipeline
from audiobookshelf_rag_ingester.config import ConfigError, Settings
from audiobookshelf_rag_ingester.db import migrate
from audiobookshelf_rag_ingester.pipeline import RunResult
from audiobookshelf_rag_ingester.schedule import Heartbeat, ScanMode, serve

logger = logging.getLogger("audiobookshelf_rag_ingester")

HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "audiobookshelf-rag-ingester.heartbeat"


def _report(result: RunResult, stop: threading.Event) -> int:
    """Log a run's outcome; return its exit code."""
    counts = result.counts
    summary = (
        f"seen={counts.seen} added={counts.added} updated={counts.updated} "
        f"deleted={counts.deleted} quarantined={counts.quarantined} errors={len(counts.errors)}"
    )
    match result.status:
        case "busy":
            logger.warning("Skipped: %s", result.message)
            return 0
        case "valve" | "partial":
            logger.error("Sync finished without deletions: %s (%s)", result.message, summary)
            return 1
    if stop.is_set():
        logger.info("Stopped early. %s", summary)
    else:
        logger.info("Sync complete. %s", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audiobookshelf-rag-ingester",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "run", "migrate", "status", "healthcheck"],
        help="what to do (default: serve)",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["incremental", "full", "reindex"],
        help="for run: incremental, full (default) or reindex",
    )
    args = parser.parse_args(argv)

    heartbeat = Heartbeat(HEARTBEAT_PATH)
    if args.command == "healthcheck":
        return 0 if heartbeat.is_fresh() else 1

    # A local .env is a convenience for running outside Docker; variables
    # already set in the environment win.
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"audiobookshelf-rag-ingester: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Every command needs the schema, so apply any pending migrations first;
    # when it is already current this is one query.
    try:
        migrate(settings.database_url)
    except Exception:
        logger.exception("Migration failed")
        return 1
    if args.command == "migrate":
        logger.info("Database schema is up to date")
        return 0

    if args.command == "status":
        try:
            report = pipeline.status(settings)
        except Exception:
            logger.exception("Cannot read status")
            return 1
        sys.stdout.write(json.dumps(report, indent=2, default=str) + "\n")
        return 0

    # `docker stop` sends SIGTERM: finish the item being read, index what is
    # staged, then stop (serve also stops sleeping).
    stop = threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping after the current item", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if args.command == "serve":
        logger.info(
            "audiobookshelf-rag-ingester %s serving: a scan every %d minutes, "
            "a full one at least every %d hours",
            __version__,
            settings.scan_interval_minutes,
            settings.full_scan_interval_hours,
        )

        def scan(mode: ScanMode) -> None:
            logger.info("Starting a %s scan", mode)
            _report(pipeline.run(settings, mode, stop, heartbeat.beat), stop)

        serve(
            scan=scan,
            last_full=lambda: pipeline.last_full_run(settings),
            interval=timedelta(minutes=settings.scan_interval_minutes),
            full_every=timedelta(hours=settings.full_scan_interval_hours),
            stop=stop,
            beat=heartbeat.beat,
        )
        return 0

    logger.info("audiobookshelf-rag-ingester %s starting a %s run", __version__, args.mode)
    try:
        result = pipeline.run(settings, args.mode, stop)
    except Exception:
        # Top level of a one-shot run: log the traceback and fail the run.
        logger.exception("Sync failed")
        return 1
    return _report(result, stop)


if __name__ == "__main__":
    sys.exit(main())
