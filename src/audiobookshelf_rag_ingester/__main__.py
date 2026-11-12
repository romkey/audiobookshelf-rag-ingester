"""Command line: `audiobookshelf-rag-ingester`, or
`python -m audiobookshelf_rag_ingester`.

With no command it syncs Audiobookshelf into Qdrant, every
POLL_INTERVAL_SECONDS (or once with --once). `query` searches the
collection; `healthcheck` is for the production container's health check.
"""

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from audiobookshelf_rag_ingester import __version__, heartbeat
from audiobookshelf_rag_ingester.abs_client import AbsClient, AbsClientError
from audiobookshelf_rag_ingester.config import ConfigError, Settings
from audiobookshelf_rag_ingester.documents import CHUNK_TYPES
from audiobookshelf_rag_ingester.embeddings import Embedder, build_embedder
from audiobookshelf_rag_ingester.qdrant_store import QdrantStore
from audiobookshelf_rag_ingester.query import format_hits, search
from audiobookshelf_rag_ingester.service import IngestionService

logger = logging.getLogger("audiobookshelf_rag_ingester")

# Errors that mean "a service is down or misconfigured", reported without a traceback.
SERVICE_ERRORS = (
    AbsClientError,
    httpx.HTTPError,
    ResponseHandlingException,
    UnexpectedResponse,
    ConnectionError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobookshelf-rag-ingester", description=__doc__
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="sync once and exit (like POLL_INTERVAL_SECONDS=0)",
    )
    commands = parser.add_subparsers(dest="command")

    query = commands.add_parser("query", help="search the collection")
    query.add_argument("text", help="natural-language query")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--type", choices=CHUNK_TYPES, help="only this chunk_type")
    query.add_argument("--library", help="only this library id")
    query.add_argument("--min-score", type=float, default=None)
    query.add_argument(
        "--show-text", action="store_true", help="print each chunk's text"
    )
    query.add_argument("--width", type=int, default=100)

    commands.add_parser("healthcheck", help="exit 0 if the running sync is alive")
    return parser


def connect_store(
    settings: Settings,
    vector_size: int,
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> QdrantStore:
    """Open the collection, waiting for Qdrant to come up (it may start after us)."""
    for attempt in range(1, attempts + 1):
        try:
            store = QdrantStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection=settings.qdrant_collection,
            )
            store.ensure_collection(
                vector_size=vector_size, distance=settings.qdrant_distance
            )
        except (
            ResponseHandlingException,
            httpx.TransportError,
            ConnectionError,
        ) as exc:
            if attempt == attempts:
                raise
            logger.warning(
                "Qdrant not ready (attempt %d/%d): %s; retrying in 5s",
                attempt,
                attempts,
                exc,
            )
            sleep(5)
        else:
            return store
    raise AssertionError("unreachable")


def run_sync(
    settings: Settings,
    *,
    once: bool,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
    heartbeat_path: Path = heartbeat.DEFAULT_PATH,
) -> int:
    embedder = embedder or build_embedder(settings)
    vector_size = embedder.dimension
    logger.info("Embedding dimension: %d", vector_size)
    store = store or connect_store(settings, vector_size)
    with AbsClient(
        base_url=settings.abs_url,
        token=settings.abs_api_token,
        timeout_seconds=settings.request_timeout_seconds,
        page_size=settings.abs_page_size,
    ) as client:
        service = IngestionService(
            settings=settings,
            client=client,
            store=store,
            embedder=embedder,
            heartbeat_path=None if once else heartbeat_path,
        )

        # `docker stop` sends SIGTERM: finish the current item, then exit.
        def stop(signum: int, frame: FrameType | None) -> None:
            logger.info("received signal %d, stopping after the current item", signum)
            service.stop()

        signal.signal(signal.SIGTERM, stop)

        if once or settings.poll_interval_seconds <= 0:
            stats = service.run_cycle()
            return 2 if stats.errors else 0
        logger.info(
            "Continuous mode: syncing every %ss", settings.poll_interval_seconds
        )
        service.run_forever()
    return 0


def run_query(settings: Settings, args: argparse.Namespace) -> int:
    embedder = build_embedder(settings)
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    hits = search(
        store,
        embedder,
        args.text,
        limit=args.limit,
        chunk_type=args.type,
        library_id=args.library,
        min_score=args.min_score,
    )
    format_hits(hits, sys.stdout, show_text=args.show_text, width=args.width)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "healthcheck":
        return 0 if heartbeat.is_fresh(heartbeat.DEFAULT_PATH) else 1

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"audiobookshelf-rag-ingester: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info(
        "audiobookshelf-rag-ingester %s (embedding=%s/%s, collection=%s)",
        __version__,
        settings.embedding_provider,
        settings.embedding_model,
        settings.qdrant_collection,
    )

    try:
        if args.command == "query":
            return run_query(settings, args)
        return run_sync(settings, once=args.once)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except (
        *SERVICE_ERRORS,
        ValueError,
    ) as exc:  # ValueError: e.g. vector size mismatch
        logger.error("Stopping: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
