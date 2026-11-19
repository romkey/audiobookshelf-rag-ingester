from __future__ import annotations

import json
import signal
import threading
from collections.abc import Callable
from typing import Any

import pytest

from audiobookshelf_rag_ingester import __main__ as cli
from audiobookshelf_rag_ingester import __version__, pipeline
from audiobookshelf_rag_ingester.config import Settings
from audiobookshelf_rag_ingester.db import RunCounts
from audiobookshelf_rag_ingester.pipeline import RunResult

SettingsFactory = Callable[..., Settings]  # the make_settings fixture


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never read a developer's real .env during tests.
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


@pytest.fixture(autouse=True)
def migrations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # Records the database URLs main() migrated instead of touching Postgres.
    calls: list[str] = []
    monkeypatch.setattr(cli, "migrate", calls.append)
    return calls


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))


def _fake_run(
    monkeypatch: pytest.MonkeyPatch, result: RunResult | Exception, seen: list[Any] | None = None
) -> None:
    def run(settings: Settings, mode: str, stop: threading.Event) -> RunResult:
        if seen is not None:
            seen.append((mode, stop))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pipeline, "run", run)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"audiobookshelf-rag-ingester {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("ABS_URL", "ABS_API_TOKEN", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "ABS_URL" in capsys.readouterr().err


def test_run_defaults_to_full(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())
    seen: list[Any] = []
    _fake_run(monkeypatch, RunResult("ok", RunCounts(seen=3, added=2)), seen)
    with caplog.at_level("INFO"):
        assert cli.main([]) == 0
    assert seen[0][0] == "full"
    assert "Sync complete. seen=3 added=2" in caplog.text


@pytest.mark.parametrize(
    ("result", "code", "log"),
    [
        (RunResult("busy", RunCounts(), "locked"), 0, "Skipped: locked"),
        (RunResult("valve", RunCounts(), "too few"), 1, "without deletions: too few"),
        (RunResult("partial", RunCounts(), "listing failed"), 1, "listing failed"),
        (RuntimeError("qdrant unreachable"), 1, "qdrant unreachable"),
    ],
)
def test_run_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
    result: RunResult | Exception,
    code: int,
    log: str,
) -> None:
    _use_settings(monkeypatch, make_settings())
    _fake_run(monkeypatch, result)
    assert cli.main(["run", "--mode", "incremental"]) == code
    assert log in caplog.text


def test_sigterm_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())
    # Any: signal handlers take (signum, frame).
    handlers: dict[int, Callable[..., Any]] = {}
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)

    def run(settings: Settings, mode: str, stop: threading.Event) -> RunResult:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert stop.is_set()
        return RunResult("ok", RunCounts(seen=1))

    monkeypatch.setattr(pipeline, "run", run)
    with caplog.at_level("INFO"):
        assert cli.main([]) == 0
    assert "Stopped early. seen=1" in caplog.text


@pytest.mark.parametrize("command", ["migrate", "run", "status"])
def test_every_command_migrates_first(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    migrations: list[str],
    command: str,
) -> None:
    # A run against a database nobody migrated must not fail on missing tables.
    _use_settings(monkeypatch, make_settings(database_url="postgresql://x"))
    order: list[str] = []
    monkeypatch.setattr(cli, "migrate", lambda url: order.append(f"migrate {url}"))

    def run(*args: Any) -> RunResult:
        order.append("run")
        return RunResult("ok", RunCounts())

    def status(settings: Settings) -> dict[str, Any]:
        order.append("status")
        return {}

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(pipeline, "status", status)
    assert cli.main([command]) == 0
    assert order[0] == "migrate postgresql://x"
    assert order[1:] == ([] if command == "migrate" else [command])


@pytest.mark.parametrize("command", ["migrate", "run", "status"])
def test_failed_migration_stops_the_command(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, command: str
) -> None:
    _use_settings(monkeypatch, make_settings())

    def broken(url: str) -> None:
        raise RuntimeError("no db")

    def never(*args: Any) -> Any:
        raise AssertionError("ran without a schema")

    monkeypatch.setattr(cli, "migrate", broken)
    monkeypatch.setattr(pipeline, "run", never)
    monkeypatch.setattr(pipeline, "status", never)
    assert cli.main([command]) == 1


def test_status(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_settings(monkeypatch, make_settings())
    monkeypatch.setattr(pipeline, "status", lambda settings: {"source": "audiobookshelf"})
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"source": "audiobookshelf"}

    def broken(settings: Settings) -> dict[str, Any]:
        raise RuntimeError("no db")

    monkeypatch.setattr(pipeline, "status", broken)
    assert cli.main(["status"]) == 1
