import io
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from audiobookshelf_rag_ingester import __main__ as cli
from audiobookshelf_rag_ingester import __version__, heartbeat
from audiobookshelf_rag_ingester.qdrant_store import QdrantStore
from audiobookshelf_rag_ingester.query import Hit, format_hits, search
from audiobookshelf_rag_ingester.service import CycleStats, IngestionService
from tests.fakes import FakeEmbedder, make_settings

ENV = {"ABS_URL": "http://abs.local", "ABS_API_TOKEN": "tok"}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"audiobookshelf-rag-ingester {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ABS_URL", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "ABS_URL is required" in capsys.readouterr().err


def test_healthcheck(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    monkeypatch.setattr(heartbeat, "DEFAULT_PATH", beat)
    assert cli.main(["healthcheck"]) == 1
    heartbeat.touch(beat)
    assert cli.main(["healthcheck"]) == 0


def _fake_run_sync(result: int | BaseException) -> Callable[..., int]:
    # Any: stands in for run_sync's keyword arguments.
    def run(settings: Any, *, once: bool) -> int:
        if isinstance(result, BaseException):
            raise result
        return result

    return run


@pytest.mark.usefixtures("env")
@pytest.mark.parametrize(
    ("result", "code"),
    [
        (0, 0),
        (2, 2),
        (KeyboardInterrupt(), 130),
        (httpx.ConnectError("refused"), 1),
        (ValueError("vector size"), 1),
    ],
)
def test_main_exit_codes(
    monkeypatch: pytest.MonkeyPatch, result: int | BaseException, code: int
) -> None:
    monkeypatch.setattr(cli, "run_sync", _fake_run_sync(result))
    assert cli.main(["--once"]) == code


@pytest.mark.usefixtures("env")
def test_main_query(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_query(settings: Any, args: Any) -> int:
        seen["text"] = args.text
        seen["type"] = args.type
        return 0

    monkeypatch.setattr(cli, "run_query", fake_query)
    assert cli.main(["query", "--type", "episode", "laser cutters"]) == 0
    assert seen == {"text": "laser cutters", "type": "episode"}


class FakeService:
    """Records how run_sync drives the service."""

    def __init__(self, errors: int = 0) -> None:
        self.errors = errors
        self.ran: list[str] = []
        self.stopped = False

    def run_cycle(self) -> CycleStats:
        self.ran.append("cycle")
        return CycleStats(errors=self.errors)

    def run_forever(self) -> None:
        self.ran.append("forever")

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.parametrize(
    ("once", "interval", "errors", "ran", "code"),
    [
        (True, 3600, 0, ["cycle"], 0),
        (False, 0, 1, ["cycle"], 2),
        (False, 3600, 0, ["forever"], 0),
    ],
)
def test_run_sync_modes_and_sigterm(
    monkeypatch: pytest.MonkeyPatch,
    store: QdrantStore,
    once: bool,
    interval: int,
    errors: int,
    ran: list[str],
    code: int,
) -> None:
    service = FakeService(errors)
    handlers: dict[int, Callable[..., Any]] = {}
    built: dict[str, Any] = {}

    def fake_service(**kwargs: Any) -> FakeService:
        built.update(kwargs)
        return service

    monkeypatch.setattr(cli, "IngestionService", fake_service)
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)

    settings = make_settings(poll_interval_seconds=interval)
    result = cli.run_sync(settings, once=once, embedder=FakeEmbedder(), store=store)
    assert result == code
    assert service.ran == ran
    assert (built["heartbeat_path"] is None) == once

    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert service.stopped


def test_connect_store_retries_until_qdrant_is_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    class FlakyStore:
        def __init__(self, **kwargs: Any) -> None:
            attempts.append(1)

        def ensure_collection(self, *, vector_size: int, distance: str) -> None:
            if len(attempts) < 3:
                raise ResponseHandlingException(ConnectionError("not yet"))

    monkeypatch.setattr(cli, "QdrantStore", FlakyStore)
    cli.connect_store(make_settings(), 8, sleep=sleeps.append)
    assert len(attempts) == 3
    assert sleeps == [5, 5]

    attempts.clear()
    with pytest.raises(ResponseHandlingException):
        cli.connect_store(make_settings(), 8, attempts=2, sleep=sleeps.append)


def test_search_and_format(store: QdrantStore) -> None:
    embedder = FakeEmbedder()
    service = IngestionService(
        settings=make_settings(ingest_authors=False),
        client=_one_book_client(),
        store=store,
        embedder=embedder,
    )
    service.run_cycle()

    hits = search(store, embedder, "The Long Way", limit=3, chunk_type="item_metadata")
    assert len(hits) == 1
    out = io.StringIO()
    format_hits(hits, out, show_text=True, width=60)
    text = out.getvalue()
    assert "[item_metadata] The Long Way" in text
    assert "https://abs.example.com/item/b1" in text

    out = io.StringIO()
    format_hits([], out)
    assert out.getvalue() == "No results.\n"

    out = io.StringIO()
    payload = {"title": "Pod", "episode_title": "Ep", "section_title": "S"}
    format_hits([Hit(0.5, payload)], out)
    assert "Pod — Ep § S" in out.getvalue()


def test_run_query(
    monkeypatch: pytest.MonkeyPatch,
    store: QdrantStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(cli, "QdrantStore", lambda **kwargs: store)
    args = cli.build_parser().parse_args(["query", "anything"])
    assert cli.run_query(make_settings(), args) == 0
    assert capsys.readouterr().out == "No results.\n"


def _one_book_client() -> Any:
    from tests.fakes import FakeAbsClient, expanded_book

    client = FakeAbsClient()
    client.add(expanded_book("b1"))
    return client
