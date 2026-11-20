from __future__ import annotations

import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from audiobookshelf_rag_ingester.schedule import Heartbeat, ScanMode, next_mode, serve

NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)
DAY = timedelta(hours=24)


def test_next_mode() -> None:
    assert next_mode(None, NOW, DAY) == "full"
    assert next_mode(NOW - timedelta(hours=23), NOW, DAY) == "incremental"
    assert next_mode(NOW - DAY, NOW, DAY) == "full"


def test_heartbeat(tmp_path: Path) -> None:
    beat = Heartbeat(tmp_path / "hb")
    assert not beat.is_fresh()
    beat.beat()
    assert beat.is_fresh()
    os.utime(beat.path, (1000, 1000))
    assert beat.is_fresh(max_age=60, now=1050)
    assert not beat.is_fresh(max_age=60, now=1061)
    Heartbeat(tmp_path / "missing" / "hb").beat()  # logs, doesn't raise


def test_serve_scans_sleeps_and_survives_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = threading.Event()
    waits: list[float] = []
    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])

    def wait(seconds: float) -> bool:
        waits.append(seconds)
        clock[0] += seconds
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", wait)
    last_full: list[datetime | None] = [None]
    scans: list[ScanMode] = []
    beats: list[int] = []

    def scan(mode: ScanMode) -> None:
        scans.append(mode)
        if mode == "full":
            last_full[0] = NOW
        if len(scans) == 2:
            raise RuntimeError("abs down")
        if len(scans) == 3:
            stop.set()

    serve(
        scan=scan,
        last_full=lambda: last_full[0],
        interval=timedelta(minutes=2.5),
        full_every=DAY,
        stop=stop,
        beat=lambda: beats.append(1),
        now=lambda: NOW,
    )
    # The first scan is full; a failure doesn't end the loop; stop does.
    assert scans == ["full", "incremental", "incremental"]
    # 150 seconds of sleep between scans, woken every minute to beat.
    assert waits == [60.0, 60.0, 30.0, 60.0, 60.0, 30.0]
    assert len(beats) >= len(scans) + len(waits)


def test_serve_survives_a_database_error() -> None:
    stop = threading.Event()
    calls: list[str] = []

    def last_full() -> datetime | None:
        calls.append("last_full")
        stop.set()
        raise RuntimeError("db down")

    serve(
        scan=lambda mode: calls.append(mode),
        last_full=last_full,
        interval=timedelta(minutes=1),
        full_every=DAY,
        stop=stop,
        beat=lambda: None,
    )
    assert calls == ["last_full"]
