"""End-to-end runs: real PostgreSQL, in-memory Qdrant, fake Audiobookshelf and models."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient

from audiobookshelf_rag_ingester.config import Settings
from audiobookshelf_rag_ingester.db import Database
from audiobookshelf_rag_ingester.documents import PODCAST_SOURCE, point_id
from audiobookshelf_rag_ingester.llm import LLM
from audiobookshelf_rag_ingester.pipeline import (
    CATALOG_SOURCE,
    OVERVIEW_DOC_ID,
    SOURCE,
    Pipeline,
    load_taxonomy,
)
from tests.fakes import (
    BOOK_LIBRARY,
    PODCAST_LIBRARY,
    FakeAbsClient,
    FakeEmbedder,
    expanded_book,
    expanded_podcast,
)

SettingsFactory = Callable[..., Settings]
# Any: payloads are JSON.
Payload = dict[str, Any]


def llm_transport() -> tuple[httpx.MockTransport, list[str]]:
    """A gateway that answers book and podcast analysis, chunk context and link checks."""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        prompts.append(system)
        if "catalog books" in system:
            content = json.dumps(
                {
                    "topics": ["science-fiction", "not-a-topic"],
                    "description_is_marketing": False,
                    "good_for": "A cozy space opera about found family.",
                }
            )
        elif "catalog podcasts" in system:
            content = json.dumps({"topics": ["laser-cutting", "hackerspace-and-community"]})
        elif "place the excerpt" in system:
            content = "This part of the overview describes the audiobook library."
        else:
            content = '{"match": true}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler), prompts


class Harness:
    def __init__(self, *, settings: Settings, db: Database, llm: bool = False) -> None:
        self.settings = settings
        self.db = db
        self.qdrant = QdrantClient(location=":memory:")
        self.embedder = FakeEmbedder()
        self.abs = FakeAbsClient()
        self.stop = threading.Event()
        self.beats = 0
        self.prompts: list[str] = []
        self.llm: LLM | None = None
        if llm:
            transport, self.prompts = llm_transport()
            self.llm = LLM(
                model="test-llm",
                base_url="http://gateway/v1",
                api_key="k",
                timeout_seconds=5,
                cache=db,
                transport=transport,
            )

    def pipeline(self) -> Pipeline:
        return Pipeline(
            settings=self.settings,
            db=self.db,
            embedder=self.embedder,
            qdrant=self.qdrant,
            client=self.abs,
            llm=self.llm,
            stop=self.stop,
            beat=self.beat,
        )

    def beat(self) -> None:
        self.beats += 1

    def run(self, mode: str = "full") -> Any:
        return self.pipeline().run(mode)  # type: ignore[arg-type]  # tests pass valid modes

    def payloads(self, collection: str, source: str | None = None) -> list[Payload]:
        p = self.pipeline()
        store = p.catalog if collection == "catalog" else p.member
        return store.payloads(source)

    def catalog(self) -> list[Payload]:
        return self.payloads("catalog", CATALOG_SOURCE)


@pytest.fixture
def harness(make_settings: SettingsFactory, db: Database) -> Harness:
    return Harness(settings=make_settings(), db=db)


def add_ebook_work(db: Database, *, title: str, author: str, isbn13: str | None = None) -> str:
    """What the ebook source (calibre-web2rag) writes for one book."""
    from audiobookshelf_rag_ingester.works import work_id_for

    work_id = work_id_for(title=title, authors=[author], isbn13=isbn13)
    db.upsert_work(
        work_id=work_id,
        title=title,
        authors=[author],
        isbn13=isbn13,
        series_id=None,
        series_index=None,
    )
    db.upsert_manifestation(
        source="calibre",
        source_id="7",
        work_id=work_id,
        format="ebook",
        url="https://books.example/book/7",
        details={"title": title, "description": "An ebook description.", "tags": ["sf"]},
    )
    db.commit()
    return work_id


# -- audiobooks and works -----------------------------------------------------


def test_first_run_stages_registers_and_indexes(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1"))
    result = harness.run()
    assert result.status == "ok"
    assert result.counts.added == 1

    staged = db.staged_documents(SOURCE)
    assert [r["source_doc_id"] for r in staged] == ["b1"]
    (work,) = db.work_ids_for_source(SOURCE)
    assert work == "title:long-way-to-a-small-angry-planet--chambers"
    (m,) = db.manifestations(work)
    assert m["format"] == "audiobook"
    assert m["details"]["narrators"] == ["Rachel Dulude"]
    assert m["details"]["chapters"] == ["Opening Credits", "Transit", "The Long Way"]

    (row,) = db.catalog_rows(SOURCE)
    assert row["narrators"] == ["Rachel Dulude"]
    assert row["duration_seconds"] == pytest.approx(55800)
    assert row["genres"] == ["Science Fiction"]
    assert row["series"] == "Wayfarers"
    assert row["series_index"] == 1.0
    assert row["languages"] == ["eng"]

    (point,) = harness.catalog()
    assert point["formats"] == ["audiobook"]
    assert point["tier"] == "member"
    assert "Audiobook chapters:\nOpening Credits\nTransit" in point["text"]
    assert "Narrated by: Rachel Dulude" in point["text"]
    assert "Audiobook length: 15 hr 30 min" in point["text"]
    assert point["work_ids"] == [work]
    assert point["manifestations"][0]["source"] == SOURCE
    assert harness.beats >= 2  # the item and its embedding batch


def test_audiobook_joins_the_ebook_work(harness: Harness, db: Database) -> None:
    # "(Unabridged)" and the series prefix are stripped before matching.
    ebook_work = add_ebook_work(
        db, title="The Long Way to a Small, Angry Planet", author="Becky Chambers"
    )
    harness.abs.add(
        expanded_book("b1", title="Wayfarers 1: The Long Way to a Small, Angry Planet (Unabridged)")
    )
    harness.run()
    (point,) = harness.catalog()
    assert point["work_id"] == ebook_work
    assert point["formats"] == ["audiobook", "ebook"]
    assert {m["source"] for m in point["manifestations"]} == {"calibre", SOURCE}
    assert "An ebook description." in point["text"] or "misfits" in point["text"]


def test_is_there_an_audiobook_version(harness: Harness, db: Database) -> None:
    """Acceptance: ebook only, audiobook only and both answer from the catalog."""
    add_ebook_work(db, title="Ebook Only Book", author="Ann Writer")
    both = add_ebook_work(db, title="A Book In Both Formats", author="Bo Th")
    harness.abs.add(
        expanded_book("b1", title="A Book in Both Formats", authors=("Bo Th",), series=None)
    )
    harness.abs.add(expanded_book("b2", title="Only On Audio", authors=("Al Audio",), series=None))
    harness.run()
    by_title = {p["title"]: p["formats"] for p in harness.catalog()}
    assert by_title["A Book In Both Formats"] == ["audiobook", "ebook"]
    assert by_title["Only On Audio"] == ["audiobook"]
    # The ebook-only work wasn't touched by this source; calibre-web2rag indexes it.
    assert "Ebook Only Book" not in by_title
    assert db.get_work(both) is not None


def test_series_order_across_formats(harness: Harness, db: Database) -> None:
    for n, title in enumerate(["Leviathan Wakes", "Caliban's War", "Abaddon's Gate"], start=1):
        harness.abs.add(
            expanded_book(
                f"b{n}",
                title=f"{title}: The Expanse, Book {n}",
                authors=("James S. A. Corey",),
                series="The Expanse",
                sequence=str(n),
            )
        )
    harness.run()
    ordered = sorted(
        (p for p in harness.catalog() if p["series"] == "The Expanse"),
        key=lambda p: p["series_index"],
    )
    assert [p["title"] for p in ordered] == ["Leviathan Wakes", "Caliban's War", "Abaddon's Gate"]


def test_series_and_sequence_break_a_close_title(harness: Harness, db: Database) -> None:
    # Similar but not auto-merge-similar titles merge when series and number agree.
    work_id = "title:the-dragon-reborn--jordan"
    db.upsert_series("series:the-wheel-of-time", "The Wheel of Time")
    db.upsert_work(
        work_id=work_id,
        title="The Dragon Reborn",
        authors=["Robert Jordan"],
        isbn13=None,
        series_id="series:the-wheel-of-time",
        series_index=3.0,
    )
    db.upsert_manifestation(
        source="calibre", source_id="3", work_id=work_id, format="ebook", url=None, details={}
    )
    db.commit()
    harness.abs.add(
        expanded_book(
            "b1",
            title="Dragon Reborn, The",
            authors=("Robert Jordan",),
            series="The Wheel of Time",
            sequence="3",
        )
    )
    harness.run()
    assert db.manifestations(work_id)[0]["format"] == "audiobook"


def test_review_queue_for_uncertain_matches(harness: Harness, db: Database) -> None:
    add_ebook_work(db, title="The Martian Chronicles", author="Ray Bradbury")
    harness.abs.add(
        expanded_book(
            "b1", title="The Martian Chronicle Stories", authors=("Ray Bradbury",), series=None
        )
    )
    harness.run()
    rows = db._all("SELECT kind, left_id, right_id FROM registry_review_queue")
    assert len(rows) == 1
    assert rows[0]["kind"] == "work_merge"


def test_isbn_gives_the_work_id(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1", isbn="0-306-40615-2"))
    harness.run()
    assert db.work_ids_for_source(SOURCE) == ["isbn:9780306406157"]


def test_llm_topics_and_overview(make_settings: SettingsFactory, db: Database) -> None:
    h = Harness(settings=make_settings(library_access_info="Ask in #library."), db=db, llm=True)
    h.abs.libraries = [BOOK_LIBRARY, PODCAST_LIBRARY]
    h.abs.add(expanded_book("b1"))
    h.abs.add(expanded_podcast())
    h.run()
    (point,) = h.catalog()
    assert point["topics"] == ["science-fiction"]  # not-a-topic is dropped
    overview = [p for p in h.payloads("member", SOURCE) if p["doc_id"] == OVERVIEW_DOC_ID]
    text = "\n".join(p["text"] for p in overview)
    assert "audiobook library of 1 audiobooks" in text
    assert "Ask in #library." in text
    assert "1 podcast with 2 episodes" in text
    assert overview[1]["context"].startswith("This part of the overview")
    pods = h.payloads("member", PODCAST_SOURCE)
    assert {p["topics"][0] for p in pods} == {"hackerspace-and-community"}


def test_enrichment_and_summary_are_labelled(make_settings: SettingsFactory, db: Database) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/search.json":
            return httpx.Response(
                200,
                json={
                    "docs": [
                        {
                            "key": "/works/OL1W",
                            "title": "The Long Way to a Small, Angry Planet",
                            "author_name": ["Becky Chambers"],
                            "subject": ["Space opera"],
                        }
                    ]
                },
            )
        if request.url.path == "/works/OL1W.json":
            return httpx.Response(200, json={"description": "A found-family space opera."})
        return httpx.Response(404)

    from audiobookshelf_rag_ingester.enrichment import Enricher

    h = Harness(settings=make_settings(enrichment_enabled=True), db=db)
    h.abs.add(expanded_book("b1", description=""))
    pipeline = h.pipeline()
    pipeline.enricher = Enricher(
        timeout_seconds=5, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    pipeline.run("full")
    (m,) = db.manifestations(db.work_ids_for_source(SOURCE)[0])
    assert m["details"]["description"] == ""
    assert m["details"]["enriched_description"] == "A found-family space opera."
    assert m["details"]["enrichment_origin"] == "openlibrary"
    (point,) = h.catalog()
    assert "Subjects: Space opera" in point["text"]


def test_no_listening_progress_or_user_data_is_staged(harness: Harness, db: Database) -> None:
    book = expanded_book("b1")
    book["userMediaProgress"] = {"progress": 0.5, "currentTime": 99}
    book["media"]["metadata"]["bookmarks"] = [{"title": "secret spot"}]
    book["mediaProgress"] = [{"userId": "u1"}]
    harness.abs.add(book)
    harness.run()
    blob = json.dumps(db._all("SELECT * FROM staging_documents"), default=str)
    blob += json.dumps(db._all("SELECT * FROM registry_manifestations"), default=str)
    blob += json.dumps(harness.catalog())
    for word in ("userMediaProgress", "currentTime", "bookmarks", "secret spot", "userId"):
        assert word not in blob
    assert all(c.startswith(("libraries", "items:", "item:")) for c in harness.abs.calls)


def test_secrets_are_redacted_and_quarantined(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1", description="<p>wifi password = hunter2hunter2</p>"))
    harness.run()
    (m,) = db.manifestations(db.work_ids_for_source(SOURCE)[0])
    assert "hunter2" not in m["details"]["description"]
    assert db.quarantine_count([SOURCE]) == 1


# -- sync: incremental, deletion, safety valve -------------------------------


def test_incremental_fetches_only_changed_items(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_book("b2", title="Record of a Spaceborn Few", sequence="3"))
    harness.run()
    harness.abs.calls.clear()
    harness.embedder.calls.clear()
    harness.abs.add(
        expanded_book("b2", title="Record of a Spaceborn Few", sequence="3", updated_at=2)
    )
    result = harness.run("incremental")
    assert [c for c in harness.abs.calls if c.startswith("item:")] == ["item:b2"]
    # Only updatedAt moved: restaged, but the catalog text is the same, so nothing is embedded.
    assert result.counts.updated == 1
    assert not [t for t in harness.embedder.texts if "Spaceborn" in t]
    harness.abs.add(
        expanded_book(
            "b2",
            title="Record of a Spaceborn Few",
            sequence="3",
            updated_at=3,
            narrators=("New Voice",),
        )
    )
    result = harness.run("incremental")
    assert result.counts.updated == 1
    point = next(p for p in harness.catalog() if p["title"] == "Record of a Spaceborn Few")
    assert point["narrators"] == ["New Voice"]


def test_deleted_audiobook_leaves_its_work(harness: Harness, db: Database) -> None:
    ebook_work = add_ebook_work(
        db, title="The Long Way to a Small, Angry Planet", author="Becky Chambers"
    )
    for n in range(3):
        harness.abs.add(expanded_book(f"x{n}", title=f"Other {n}", series=None))
    harness.abs.add(expanded_book("b1"))
    harness.run()
    del harness.abs.items["b1"]
    result = harness.run("incremental")
    assert result.counts.deleted == 1
    assert db.manifestations(ebook_work)[0]["source"] == "calibre"
    point = next(p for p in harness.catalog() if p["work_id"] == ebook_work)
    assert point["formats"] == ["ebook"]
    assert sorted(r["source_id"] for r in db.catalog_rows(SOURCE)) == ["x0", "x1", "x2"]


def test_safety_valve_blocks_mass_deletion(harness: Harness, db: Database) -> None:
    for n in range(4):
        harness.abs.add(expanded_book(f"b{n}", title=f"Book {n}", series=None))
    harness.run()
    harness.abs.items = {"b0": harness.abs.items["b0"]}
    result = harness.run()
    assert result.status == "valve"
    assert len(db.staged_documents(SOURCE)) == 4
    assert len(harness.catalog()) == 4


def test_failed_library_listing_deletes_nothing_there(harness: Harness, db: Database) -> None:
    harness.abs.libraries = [BOOK_LIBRARY, PODCAST_LIBRARY]
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_podcast())
    harness.run()
    harness.abs.fail_libraries = {"lib-pods"}
    result = harness.run()
    assert result.status == "partial"
    assert harness.payloads("member", PODCAST_SOURCE)
    assert len(db.staged_documents(SOURCE)) == 2


def test_stop_indexes_what_was_staged_and_deletes_nothing(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_book("b2", title="Second Book", series=None))
    harness.abs.on_item = lambda item_id: harness.stop.set() if item_id == "b1" else None
    result = harness.run()
    assert result.message == "stopped early"
    assert len(harness.catalog()) == 1


def test_one_bad_item_does_not_stop_the_run(harness: Harness) -> None:
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_book("b2", title="Second", series=None))
    harness.abs.fail_items = {"b1"}
    result = harness.run()
    assert result.status == "ok"
    assert len(result.counts.errors) == 1
    assert len(harness.catalog()) == 1


def test_lock_held_means_busy(harness: Harness, db: Database, database_url: str) -> None:
    with Database.connect(database_url) as other, other.source_lock(SOURCE) as locked:
        assert locked
        assert harness.run().status == "busy"


def test_reindex_rebuilds_from_staging_without_abs(harness: Harness) -> None:
    harness.abs.libraries = [BOOK_LIBRARY, PODCAST_LIBRARY]
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_podcast())
    harness.run()
    harness.abs.calls.clear()
    harness.embedder.calls.clear()
    result = harness.run("reindex")
    assert result.status == "ok"
    assert harness.abs.calls == []
    assert len(harness.embedder.texts) >= 4  # catalog, podcast, episodes, overview


def test_unchanged_full_run_reembeds_nothing(harness: Harness) -> None:
    harness.abs.libraries = [BOOK_LIBRARY, PODCAST_LIBRARY]
    harness.abs.add(expanded_book("b1"))
    harness.abs.add(expanded_podcast())
    harness.run()
    harness.embedder.calls.clear()
    harness.run()
    assert harness.embedder.texts == []
    runs = {p["sync_run_id"] for p in harness.payloads("member")}
    assert len(runs) == 1  # every point restamped by the second run


# -- podcasts -------------------------------------------------------------------


def test_podcasts_and_episodes_in_member(harness: Harness, db: Database) -> None:
    harness.abs.libraries = [PODCAST_LIBRARY]
    harness.abs.add(expanded_podcast(episodes=3))
    harness.run()
    pods = harness.payloads("member", PODCAST_SOURCE)
    kinds = sorted(p["kind"] for p in pods)
    assert kinds == ["episode", "episode", "episode", "podcast"]
    assert db.work_ids_for_source(SOURCE) == []  # podcasts are not works
    episode = next(p for p in pods if p["kind"] == "episode")
    assert episode["heading_path"][0] == "Hackerspace Hour"
    assert "Published:" in episode["text"]
    assert episode["url"] == "https://abs.example.com/item/li-p1"

    # An episode removed from Audiobookshelf disappears on the next change.
    item = harness.abs.items["li-p1"]
    item["media"]["episodes"] = item["media"]["episodes"][:1]
    item["updatedAt"] += 1
    harness.run("incremental")
    assert len(harness.payloads("member", PODCAST_SOURCE)) == 2

    del harness.abs.items["li-p1"]
    harness.abs.add(expanded_book("b1"))
    harness.abs.libraries = [BOOK_LIBRARY, PODCAST_LIBRARY]
    harness.run()  # 1 seen, 1 staged: the valve allows it
    assert harness.payloads("member", PODCAST_SOURCE) == []


def test_ebook_only_item_is_an_ebook_manifestation(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1", duration=0, ebook=True))
    harness.run()
    (point,) = harness.catalog()
    assert point["formats"] == ["ebook"]


def test_taxonomy_matches_the_ebook_source() -> None:
    slugs = load_taxonomy(None)
    assert "science-fiction" in slugs
    assert len(slugs) == len(set(slugs)) >= 60


def test_point_ids_are_stable() -> None:
    assert point_id(CATALOG_SOURCE, "isbn:1", 0) == point_id(CATALOG_SOURCE, "isbn:1", 0)


def test_failed_index_is_retried_by_the_next_incremental_run(
    harness: Harness, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.abs.add(expanded_book("b1"))
    pipeline = harness.pipeline()

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(pipeline, "index_catalog", broken)
    with pytest.raises(RuntimeError, match="qdrant down"):
        pipeline.run("incremental")
    assert harness.catalog() == []
    # Staging kept the item, but forgot its version: the next run fetches it again.
    harness.abs.calls.clear()
    harness.run("incremental")
    assert "item:b1" in harness.abs.calls
    assert len(harness.catalog()) == 1


def test_payload_only_change_is_written_without_reembedding(harness: Harness, db: Database) -> None:
    harness.abs.add(expanded_book("b1"))
    harness.run()
    (before,) = harness.catalog()
    assert before["year"] == 2014
    harness.embedder.calls.clear()
    book = expanded_book("b1", updated_at=5)
    book["media"]["metadata"]["publishedYear"] = "2015"  # not in the catalog text
    harness.abs.add(book)
    harness.run("incremental")
    (after,) = harness.catalog()
    assert after["year"] == 2015
    assert after["text"] == before["text"]
    assert harness.embedder.texts == []


def test_close_releases_http_clients(make_settings: SettingsFactory, db: Database) -> None:
    # serve builds a pipeline per scan, so each one must close what it opened.
    harness = Harness(settings=make_settings(), db=db, llm=True)
    p = harness.pipeline()
    from audiobookshelf_rag_ingester.enrichment import Enricher

    p.enricher = Enricher(timeout_seconds=5)
    p.close()
    assert harness.llm is not None
    assert harness.llm._client.is_closed
    assert p.enricher._client.is_closed
