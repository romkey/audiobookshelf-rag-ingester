"""Liveness for a worker with no port: a file touched while the app works.

The production compose health check runs `audiobookshelf-rag-ingester
healthcheck`, which fails if the file is missing or older than MAX_AGE.
"""

import tempfile
import time
from pathlib import Path

DEFAULT_PATH = Path(tempfile.gettempdir()) / "audiobookshelf-rag-ingester.heartbeat"

# The service touches the file before each item it ingests and every minute
# while it sleeps. A single large ebook can take a while to embed.
MAX_AGE_SECONDS = 900.0


def touch(path: Path) -> None:
    path.touch()


def is_fresh(
    path: Path, max_age: float = MAX_AGE_SECONDS, now: float | None = None
) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() if now is None else now) - mtime <= max_age
