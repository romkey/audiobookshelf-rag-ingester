from audiobookshelf_rag_ingester.models import (
    parse_author,
    parse_chapters,
    parse_item,
    parse_library,
)
from tests.fakes import expanded_book, expanded_podcast, minified


def test_parse_expanded_book() -> None:
    item = parse_item(expanded_book(ebook=True))
    assert item.title == "The Long Way"
    assert item.authors == ["Becky Chambers"]
    assert item.author_ids == {"Becky Chambers": "au-1"}
    assert item.narrators == ["Patricia Rodriguez"]
    assert [s.label() for s in item.series] == ["Wayfarers #1"]
    assert len(item.chapters) == 3
    assert item.ebook is not None
    assert item.ebook.format == "epub"
    assert item.ebook_format == "epub"
    assert item.num_chapters == 3


def test_parse_minified_book_uses_flat_names() -> None:
    item = parse_item(minified(expanded_book(ebook=True)))
    assert item.authors == ["Becky Chambers"]
    assert item.narrators == ["Patricia Rodriguez"]
    assert item.series[0].name == "Wayfarers"
    assert item.series[0].sequence == "1"
    assert item.chapters == []
    assert item.ebook_format == "epub"


def test_fingerprint_matches_between_minified_and_expanded() -> None:
    data = expanded_book(ebook=True)
    assert parse_item(data).fingerprint() == parse_item(minified(data)).fingerprint()


def test_fingerprint_changes_with_updated_at() -> None:
    a = parse_item(minified(expanded_book(updated_at=1)))
    b = parse_item(minified(expanded_book(updated_at=2)))
    assert a.fingerprint() != b.fingerprint()


def test_parse_podcast() -> None:
    full = parse_item(expanded_podcast(episodes=3))
    assert full.media_type == "podcast"
    assert full.authors == ["PDX Hackers"]
    assert len(full.episodes) == 3
    assert full.episodes[0].duration == 1800.0
    assert full.num_episodes == 3
    mini = parse_item(minified(expanded_podcast(episodes=3)))
    assert mini.num_episodes == 3
    assert mini.fingerprint() == full.fingerprint()


def test_episode_duration_falls_back_to_audio_file() -> None:
    data = expanded_podcast(episodes=1)
    del data["media"]["episodes"][0]["duration"]
    data["media"]["episodes"][0]["audioFile"] = {"duration": "42.5"}
    assert parse_item(data).episodes[0].duration == 42.5


def test_parse_library_and_author() -> None:
    lib = parse_library({"id": "x", "name": "Books", "mediaType": "book"})
    assert lib.media_type == "book"
    au = parse_author({"id": "a", "name": "N", "numBooks": 4, "description": None})
    assert au.num_books == 4
    assert au.description is None


def test_series_name_without_sequence() -> None:
    data = minified(expanded_book())
    data["media"]["metadata"]["seriesName"] = "Discworld, Other #2.5"
    item = parse_item(data)
    assert [(s.name, s.sequence) for s in item.series] == [
        ("Discworld", None),
        ("Other", "2.5"),
    ]


def test_bad_values_are_tolerated() -> None:
    data = expanded_book()
    data["updatedAt"] = "not a number"
    data["media"]["duration"] = "oops"
    data["media"]["chapters"] = [{"title": None, "start": "x"}, "junk"]
    data["media"]["metadata"]["title"] = "   "
    item = parse_item(data)
    assert item.updated_at is None
    assert item.duration is None
    assert item.title == "Untitled"
    assert [c.title for c in item.chapters] == ["Chapter 1"]
    assert parse_chapters("nope") == []
