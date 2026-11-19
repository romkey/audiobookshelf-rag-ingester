from audiobookshelf_rag_ingester.documents import (
    catalog_text,
    format_duration,
    html_to_text,
    language_code,
    meaningful_chapters,
    ms_to_rfc3339,
    overview_sections,
    podcast_points,
    year_of,
)
from audiobookshelf_rag_ingester.models import parse_item
from tests.fakes import expanded_podcast


def test_small_helpers() -> None:
    assert format_duration(None) == ""
    assert format_duration(30) == "1 min"
    assert format_duration(3600) == "1 hr"
    assert format_duration(3600 * 15.5) == "15 hr 30 min"
    assert language_code("English") == "eng"
    assert language_code("en") == "eng"
    assert language_code("Klingon") == "klingon"
    assert language_code(None) is None
    assert year_of("2014-05-01") == 2014
    assert year_of(None) is None
    assert ms_to_rfc3339(None) is None
    assert ms_to_rfc3339(10**20) is None
    assert html_to_text("<p>One<br>two</p><p>three</p>") == "One\ntwo\n\nthree"
    assert html_to_text(None) == ""
    assert meaningful_chapters(["Chapter 1", "Chapter 2"]) == []
    assert meaningful_chapters(["Intro"]) == []
    assert meaningful_chapters(["Intro", "Chapter 2"]) == ["Intro", "Chapter 2"]


def test_catalog_text_lists_every_format() -> None:
    work = {
        "title": "Dune",
        "authors": ["Frank Herbert"],
        "topics": ["science-fiction"],
        "series_index": 1.0,
    }
    ebook = {
        "format": "ebook",
        "details": {"description": "Spice.", "toc": ["Book One"], "subtitle": "Deluxe"},
    }
    audio = {
        "format": "audiobook",
        "details": {
            "narrators": ["Scott Brick"],
            "duration_seconds": 75600,
            "chapters": ["Prologue", "Arrakis"],
            "description": "Blurb!",
            "description_is_marketing": True,
            "good_for": "Classic SF.",
            "enriched_subjects": ["Deserts"],
        },
    }
    text = catalog_text(work, [audio, ebook], "Dune")
    assert text.splitlines()[:3] == ["Dune: Deluxe", "by Frank Herbert", "Series: Dune #1"]
    assert "Formats in the library: audiobook, ebook" in text
    assert "Narrated by: Scott Brick" in text
    assert "Audiobook length: 21 hr" in text
    assert "Spice." in text
    assert "Blurb!" not in text
    assert "machine-generated summary): Classic SF." in text
    assert "Contents:\nBook One" in text
    assert "Audiobook chapters:\nPrologue\nArrakis" in text


def test_catalog_text_falls_back_to_enriched_description() -> None:
    text = catalog_text(
        {"title": "X"},
        [{"format": "audiobook", "details": {"enriched_description": "From Open Library."}}],
        None,
    )
    assert text == "X\nFormats in the library: audiobook\n\nFrom Open Library."


def test_overview_defaults() -> None:
    sections = overview_sections(
        [], [], library_url="https://abs", access_info=None, apps_info=None
    )
    assert sections[0][1].startswith("PDX Hackerspace members have access to an audiobook")
    assert "ask them" in sections[2][1]
    assert "https://abs" in sections[3][1]
    rows = [
        {
            "format": "audiobook",
            "authors": ["A"],
            "series": f"S{n}",
            "topics": ["t"],
            "genres": ["g"],
            "languages": ["eng"],
            "duration_seconds": 3600,
        }
        for n in range(150)
    ]
    sections = overview_sections(rows, [], library_url="u", access_info="a", apps_info="b")
    assert "about 150 audiobooks, about 150 hours" in sections[0][1]
    assert "150 series" in sections[1][1]
    assert (sections[2][1], sections[3][1]) == ("a", "b")


def test_podcast_points_long_notes_and_numbering() -> None:
    record = parse_item(expanded_podcast(episodes=1)).to_json()
    record["description"] = "Long. " * 100
    record["topics"] = ["electronics"]
    record["explicit"] = True
    record["episodes"][0]["description"] = "Notes. " * 100
    record["episodes"][0]["subtitle"] = "A subtitle"
    record["episodes"][0]["published_at"] = None
    points = podcast_points(
        record,
        url="u",
        run_id="r",
        chunk_size=200,
        chunk_overlap=20,
        link=lambda text: {"asset_ids": ["laser"]},
    )
    payloads = [p for _, p in points]
    assert payloads[0]["text"].startswith("Podcast: Hackerspace Hour\nHosted by PDX Hackers")
    assert "Explicit content" in payloads[0]["text"]
    assert any("(description, continued)" in p["text"] for p in payloads)
    episode = [p for p in payloads if p["kind"] == "episode"]
    assert episode[0]["text"].splitlines()[2:4] == ["A subtitle", "Season 1, Episode 1"]
    assert "Published: Mon, 01 Jan 2024" in episode[0]["text"]
    assert any("(show notes, continued)" in p["text"] for p in episode)
    assert all(p["asset_ids"] == ["laser"] for p in payloads)
