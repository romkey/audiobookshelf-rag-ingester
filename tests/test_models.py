from audiobookshelf_rag_ingester.models import (
    parse_chapters,
    parse_item,
    parse_library,
    parse_listed,
)
from tests.fakes import expanded_book, expanded_podcast, minified


def test_parse_expanded_book() -> None:
    item = parse_item(expanded_book(ebook=True, isbn="9780306406157"))
    assert item.title == "The Long Way to a Small, Angry Planet"
    assert item.authors == ["Becky Chambers"]
    assert item.narrators == ["Rachel Dulude"]
    assert [(s.name, s.index()) for s in item.series] == [("Wayfarers", 1.0)]
    assert [c.title for c in item.chapters] == ["Opening Credits", "Transit", "The Long Way"]
    assert [(f.ext, f.size) for f in item.audio_files] == [("M4B", 9000)]
    assert item.has_ebook is True
    assert item.isbn == "9780306406157"
    assert item.asin == "B01M0N8Q5I"
    assert item.cover_path == "/audiobooks/li-1/cover.jpg"
    assert item.language == "English"


def test_parse_minified_book_uses_flat_names() -> None:
    item = parse_item(minified(expanded_book()))
    assert item.authors == ["Becky Chambers"]
    assert item.narrators == ["Rachel Dulude"]
    assert item.series[0].name == "Wayfarers"
    assert item.series[0].sequence == "1"
    assert item.chapters == []


def test_parse_listed() -> None:
    listed = parse_listed(minified(expanded_book("b9", updated_at=5)))
    assert (listed.id, listed.updated_at, listed.media_type) == ("b9", 5, "book")


def test_parse_podcast() -> None:
    full = parse_item(expanded_podcast(episodes=3))
    assert full.media_type == "podcast"
    assert full.authors == ["PDX Hackers"]
    assert len(full.episodes) == 3
    assert full.episodes[0].duration == 1800.0
    assert full.series == []


def test_episode_duration_falls_back_to_audio_file() -> None:
    data = expanded_podcast(episodes=1)
    del data["media"]["episodes"][0]["duration"]
    data["media"]["episodes"][0]["audioFile"] = {"duration": "42.5"}
    assert parse_item(data).episodes[0].duration == 42.5


def test_parse_library() -> None:
    lib = parse_library({"id": "x", "name": "Books", "mediaType": "book"})
    assert lib.media_type == "book"
    assert parse_library({"id": "y"}).name == "y"


def test_series_name_without_sequence() -> None:
    data = minified(expanded_book())
    data["media"]["metadata"]["seriesName"] = "Discworld, Other #2.5"
    item = parse_item(data)
    assert [(s.name, s.sequence) for s in item.series] == [("Discworld", None), ("Other", "2.5")]
    assert item.series[0].index() is None
    assert item.series[1].index() == 2.5


def test_series_single_object_and_bad_sequence() -> None:
    data = expanded_book()
    data["media"]["metadata"]["series"] = {"name": "Solo", "sequence": "one"}
    (series,) = parse_item(data).series
    assert series.name == "Solo"
    assert series.index() is None


def test_bad_values_are_tolerated() -> None:
    data = expanded_book()
    data["updatedAt"] = "not a number"
    data["media"]["duration"] = "oops"
    data["media"]["chapters"] = [{"title": None, "start": "x"}, "junk"]
    data["media"]["audioFiles"] = ["junk", {"metadata": {"ext": ""}}]
    data["media"]["metadata"]["title"] = "   "
    item = parse_item(data)
    assert item.updated_at is None
    assert item.duration is None
    assert item.title == "Untitled"
    assert [c.title for c in item.chapters] == ["Chapter 1"]
    assert [f.ext for f in item.audio_files] == [None]
    assert parse_chapters("nope") == []


def test_no_user_fields_survive_parsing() -> None:
    data = expanded_book()
    data["userMediaProgress"] = {"progress": 0.5}
    record = parse_item(data).to_json()
    assert "userMediaProgress" not in str(record)
