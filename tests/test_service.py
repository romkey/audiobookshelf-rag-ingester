"""End-to-end sync tests against an in-memory Qdrant and a fake ABS server."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from audiobookshelf_rag_ingester.qdrant_store import Payload, QdrantStore
from audiobookshelf_rag_ingester.service import IngestionService
from tests.fakes import (
    PODCAST_LIBRARY,
    FakeAbsClient,
    FakeEmbedder,
    build_epub_bytes,
    expanded_book,
    expanded_podcast,
    make_settings,
)

# Builds a service; takes heartbeat_path and Settings overrides.
type MakeService = Callable[..., IngestionService]


@pytest.fixture
def abs_client() -> FakeAbsClient:
    return FakeAbsClient()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make(
    abs_client: FakeAbsClient, store: QdrantStore, embedder: FakeEmbedder
) -> MakeService:
    # Any: overrides for any Settings field.
    def build(heartbeat_path: Path | None = None, **overrides: Any) -> IngestionService:
        return IngestionService(
            settings=make_settings(**overrides),
            client=abs_client,
            store=store,
            embedder=embedder,
            heartbeat_path=heartbeat_path,
        )

    return build


def _payloads(store: QdrantStore, **match: str) -> list[Payload]:
    flt = Filter(
        must=[
            FieldCondition(key=k, match=MatchValue(value=v)) for k, v in match.items()
        ]
    )
    fields = ["chunk_type", "item_id", "text", "author_id"]
    return list(store.scroll_ids(flt, fields).values())


def _types(store: QdrantStore, **match: str) -> list[str]:
    return sorted(str(p["chunk_type"]) for p in _payloads(store, **match))


def test_first_sync_ingests_everything(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    stats = make().run_cycle()

    assert stats.errors == 0
    assert stats.items_processed == 1
    assert _types(store, item_id="b1") == ["chapters", "item_metadata"]
    assert len(_payloads(store, chunk_type="author")) == 2
    assert _payloads(store, chunk_type="library_summary")
    assert stats.chunks_embedded == store.count()


def test_second_sync_is_a_noop(
    abs_client: FakeAbsClient, embedder: FakeEmbedder, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    make().run_cycle()
    embedder.texts.clear()
    abs_client.calls.clear()

    stats = make().run_cycle()
    assert stats.items_unchanged == 1
    assert stats.items_processed == 0
    assert stats.chunks_embedded == 0
    assert embedder.texts == []
    assert "item:b1" not in abs_client.calls  # no expanded fetch


def test_changed_item_reembeds_only_changed_chunks(
    abs_client: FakeAbsClient,
    store: QdrantStore,
    embedder: FakeEmbedder,
    make: MakeService,
) -> None:
    abs_client.add(expanded_book("b1"))
    make().run_cycle()
    embedder.texts.clear()

    abs_client.add(expanded_book("b1", updated_at=1_800_000_000_000, title="New Title"))
    stats = make().run_cycle()
    assert stats.items_processed == 1
    assert stats.chunks_embedded > 0
    meta = _payloads(store, item_id="b1", chunk_type="item_metadata")
    assert str(meta[0]["text"]).startswith("New Title")

    # Touching updatedAt alone re-fetches the item but embeds none of its chunks.
    embedder.texts.clear()
    abs_client.add(expanded_book("b1", updated_at=1_900_000_000_000, title="New Title"))
    stats = make().run_cycle()
    assert stats.items_processed == 1
    assert not any(t.startswith("Audiobook: New Title") for t in embedder.texts)


def test_deleted_item_is_removed(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    abs_client.add(expanded_book("b2", title="Second"))
    make().run_cycle()
    assert _payloads(store, item_id="b2")

    del abs_client.items["b2"]
    stats = make().run_cycle()
    assert stats.items_deleted == 1
    assert _payloads(store, item_id="b2") == []
    assert _payloads(store, item_id="b1")
    summary = _payloads(store, chunk_type="library_summary")
    assert "Second" not in "\n".join(str(p["text"]) for p in summary)


def test_shrinking_description_removes_extra_chunks(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    long_desc = " ".join(f"Descriptive sentence number {i}." for i in range(100))
    abs_client.add(expanded_book("b1", description=long_desc))
    make().run_cycle()
    assert "description" in _types(store, item_id="b1")

    abs_client.add(expanded_book("b1", description="Short.", updated_at=2))
    stats = make().run_cycle()
    assert "description" not in _types(store, item_id="b1")
    assert stats.chunks_deleted > 0


def test_ebook_is_ingested_once(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.ebook_bytes = build_epub_bytes(
        [("Chapter One", "The engine hummed. " * 40), ("Chapter Two", "Stars. " * 40)]
    )
    abs_client.add(expanded_book("b1", ebook=True))
    make().run_cycle()
    ebook_chunks = _payloads(store, item_id="b1", chunk_type="ebook_content")
    assert ebook_chunks
    assert abs_client.calls.count("ebook:b1") == 1

    # Metadata-only edit (same title/authors): the ebook isn't downloaded again.
    abs_client.add(
        expanded_book("b1", ebook=True, updated_at=5, description="Edited blurb.")
    )
    make().run_cycle()
    assert abs_client.calls.count("ebook:b1") == 1
    after = _payloads(store, item_id="b1", chunk_type="ebook_content")
    assert len(after) == len(ebook_chunks)


def test_unreadable_ebook_is_skipped(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.ebook_bytes = b"not an epub"
    abs_client.add(expanded_book("b1", ebook=True))
    stats = make().run_cycle()
    assert stats.errors == 0
    assert _payloads(store, chunk_type="ebook_content") == []


def test_missing_ebook_download_is_skipped(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.ebook_bytes = None
    abs_client.add(expanded_book("b1", ebook=True))
    assert make().run_cycle().errors == 0
    assert _payloads(store, chunk_type="ebook_content") == []


def test_ebook_download_failure_is_retried(
    abs_client: FakeAbsClient, make: MakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    abs_client.add(expanded_book("b1", ebook=True))

    def broken(item_id: str, max_bytes: int) -> bytes | None:
        raise ConnectionError("down")

    monkeypatch.setattr(abs_client, "download_ebook", broken)
    make().run_cycle()
    # The fingerprint wasn't recorded, so the next cycle tries the item again.
    stats = make().run_cycle()
    assert stats.items_processed == 1


def test_ebooks_can_be_disabled(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.ebook_bytes = build_epub_bytes([("One", "Text here.")])
    abs_client.add(expanded_book("b1", ebook=True))
    make(ingest_ebooks=False).run_cycle()
    assert "ebook:b1" not in abs_client.calls
    assert _payloads(store, chunk_type="ebook_content") == []


def test_podcast_library(
    abs_client: FakeAbsClient,
    store: QdrantStore,
    embedder: FakeEmbedder,
    make: MakeService,
) -> None:
    abs_client.libraries = [PODCAST_LIBRARY]
    abs_client.add(expanded_podcast("p1", episodes=3))
    make().run_cycle()
    assert _types(store, item_id="p1").count("episode") == 3
    assert _payloads(store, chunk_type="author") == []

    # A new episode changes numEpisodes, so the item re-syncs; only it embeds.
    embedder.texts.clear()
    abs_client.add(expanded_podcast("p1", episodes=4))
    make().run_cycle()
    assert _types(store, item_id="p1").count("episode") == 4
    episode_texts = [t for t in embedder.texts if "| Episode:" in t]
    assert len(episode_texts) == 1
    assert "Episode 4" in episode_texts[0]


def test_library_filter_and_removed_library(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.libraries.append(PODCAST_LIBRARY)
    abs_client.add(expanded_book("b1"))
    abs_client.add(expanded_podcast("p1"))
    make().run_cycle()
    assert _payloads(store, library_id="lib-pods")

    make(abs_libraries=("audiobooks",)).run_cycle()
    assert _payloads(store, library_id="lib-pods") == []
    assert _payloads(store, library_id="lib-books")

    stats = make(abs_libraries=("nothing",)).run_cycle()
    assert stats.libraries == 0


def test_optional_documents_can_be_disabled(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    make().run_cycle()
    make(ingest_authors=False, ingest_library_summary=False).run_cycle()
    assert _payloads(store, chunk_type="author") == []
    assert _payloads(store, chunk_type="library_summary") == []


def test_pipeline_change_forces_resync(
    abs_client: FakeAbsClient, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    make().run_cycle()
    stats = make(chunk_size=300, chunk_overlap=30).run_cycle()
    assert stats.items_processed == 1


def test_force_reingest_embeds_everything(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    make().run_cycle()
    stats = make(force_reingest=True).run_cycle()
    assert stats.chunks_embedded == store.count()


def test_failed_item_does_not_stop_cycle(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    abs_client.add(expanded_book("b2", title="Two"))
    abs_client.fail_items = {"b1"}
    stats = make().run_cycle()
    assert stats.errors == 1
    assert stats.failed_items == ["b1"]
    assert _payloads(store, item_id="b2")

    abs_client.fail_items = set()
    stats = make().run_cycle()
    assert stats.errors == 0
    assert stats.items_processed == 1  # b1 retried, b2 unchanged


def test_failed_library_is_counted(
    abs_client: FakeAbsClient, make: MakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any: matches AbsSource.list_library_items.
    def broken(library_id: str) -> list[dict[str, Any]]:
        raise ConnectionError("down")

    monkeypatch.setattr(abs_client, "list_library_items", broken)
    assert make().run_cycle().errors == 1


def test_stop_mid_cycle_skips_deletions(
    abs_client: FakeAbsClient, store: QdrantStore, make: MakeService
) -> None:
    abs_client.add(expanded_book("b1"))
    abs_client.add(expanded_book("b2", title="Two"))
    make().run_cycle()

    # b2 is gone. Stop as soon as the first item is fetched.
    del abs_client.items["b2"]
    service = make(force_reingest=True)
    abs_client.on_item = lambda _item_id: service.stop()
    stats = service.run_cycle()

    assert stats.interrupted
    assert stats.items_deleted == 0
    assert _payloads(store, item_id="b2")  # kept: the cycle didn't finish


def test_run_forever_sleeps_and_stops(make: MakeService, tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    service = make(heartbeat_path=beat, poll_interval_seconds=3600)
    waits = 0

    def wait(timeout: float | None = None) -> bool:
        nonlocal waits
        waits += 1
        if waits == 2:
            service.stop_event.set()
        return service.stop_event.is_set()

    # Don't sleep in tests: stand in for Event.wait.
    service.stop_event.wait = wait  # type: ignore[method-assign]
    service.run_forever()
    assert beat.exists()
    assert waits == 2


def test_run_forever_survives_a_failed_cycle(
    make: MakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make(poll_interval_seconds=0)
    calls = 0

    def failing_cycle() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            service.stop()
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "run_cycle", failing_cycle)
    service.run_forever()
    assert calls == 2
