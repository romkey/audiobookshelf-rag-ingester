"""The `serve` loop: scan, sleep, scan again, until stopped.

Each scan is an incremental run, except that a full run (which also
reconciles deletions) happens when none has finished within the full-scan
interval, so the first scan on a fresh database is a full one. A scan that
fails is logged and tried again after the next sleep; only a stop request
ends the loop.

Identical in every ingester of the pipeline: a change here is a change to
all of them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ScanMode = Literal["full", "incremental"]

# The loop touches the heartbeat at least this often while it sleeps.
BEAT_SECONDS = 60.0
# `healthcheck` fails when the heartbeat is older than this. The pipeline
# touches it before each item it reads and each batch it embeds.
HEARTBEAT_MAX_AGE_SECONDS = 900.0


class Heartbeat:
    """A file touched while the app works: liveness for a worker with no port."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def beat(self) -> None:
        try:
            self.path.touch()
        except OSError as e:
            logger.warning("Cannot touch heartbeat %s: %s", self.path, e)

    def is_fresh(
        self, max_age: float = HEARTBEAT_MAX_AGE_SECONDS, now: float | None = None
    ) -> bool:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        return (time.time() if now is None else now) - mtime <= max_age


def next_mode(last_full: datetime | None, now: datetime, full_every: timedelta) -> ScanMode:
    """Return "full" if no full run finished within ``full_every``, else "incremental"."""
    if last_full is None or now - last_full >= full_every:
        return "full"
    return "incremental"


def serve(
    *,
    scan: Callable[[ScanMode], object],
    last_full: Callable[[], datetime | None],
    interval: timedelta,
    full_every: timedelta,
    stop: threading.Event,
    beat: Callable[[], None],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Run ``scan(mode)`` every ``interval`` (counted from the end of a scan)."""
    minutes = interval.total_seconds() / 60
    while not stop.is_set():
        beat()
        try:
            mode = next_mode(last_full(), now(), full_every)
            scan(mode)
        except Exception:
            logger.exception("Scan failed; trying again in %g minutes", minutes)
        if stop.is_set():
            break
        logger.info("Sleeping %g minutes until the next scan", minutes)
        deadline = time.monotonic() + interval.total_seconds()
        while not stop.is_set():
            beat()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            stop.wait(min(BEAT_SECONDS, remaining))
    logger.info("Stopped")
