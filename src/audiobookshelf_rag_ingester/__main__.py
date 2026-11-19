"""Command line: `audiobookshelf-rag-ingester` or `python -m audiobookshelf_rag_ingester`.

Syncs the Audiobookshelf audiobook and podcast libraries into the
member-tier RAG pipeline: the Postgres staging store, entity registry and
catalog table, and the Qdrant hs_catalog and hs_member collections. Runs
once and exits.

  audiobookshelf-rag-ingester migrate                 # only create or update the schema
  audiobookshelf-rag-ingester run --mode incremental  # hourly: changed and deleted items
  audiobookshelf-rag-ingester run --mode full         # nightly: everything, then reconcile
  audiobookshelf-rag-ingester run --mode reindex      # rebuild points from staging
  audiobookshelf-rag-ingester status                  # recent runs and counts, as JSON

Every command first brings the database schema up to date.
"""

import argparse
import json
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from types import FrameType

from dotenv import load_dotenv

from audiobookshelf_rag_ingester import __version__, pipeline
from audiobookshelf_rag_ingester.config import ConfigError, Settings
from audiobookshelf_rag_ingester.db import migrate

logger = logging.getLogger("audiobookshelf_rag_ingester")


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
        default="run",
        choices=["run", "migrate", "status"],
        help="what to do (default: run)",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["incremental", "full", "reindex"],
        help="for run: incremental (hourly), full (nightly, default) or reindex",
    )
    args = parser.parse_args(argv)

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
    # staged, then stop.
    stop = threading.Event()

    def on_sigterm(signum: int, frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping after the current item", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_sigterm)

    logger.info("audiobookshelf-rag-ingester %s starting a %s run", __version__, args.mode)
    try:
        result = pipeline.run(settings, args.mode, stop)
    except Exception:
        # Top level of a batch job: log the traceback and fail the run.
        logger.exception("Sync failed")
        return 1
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


if __name__ == "__main__":
    sys.exit(main())
