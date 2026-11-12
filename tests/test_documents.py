from audiobookshelf_rag_ingester.documents import (
    DocConfig,
    books_by_author,
    build_author_documents,
    build_ebook_documents,
    build_item_documents,
    build_library_summary_documents,
    chapters_are_meaningful,
    ebook_signature,
)
from audiobookshelf_rag_ingester.extractors import TextSection
from audiobookshelf_rag_ingester.models import (
    Chapter,
    parse_author,
    parse_item,
    parse_library,
)
from tests.fakes import (
    BOOK_LIBRARY,
    PODCAST_LIBRARY,
    expanded_book,
    expanded_podcast,
    minified,
)

CFG = DocConfig(chunk_size=400, chunk_overlap=40, public_url="https://abs.example.com")
BOOKS = parse_library(BOOK_LIBRARY)
PODS = parse_library(PODCAST_LIBRARY)


def test_book_metadata_document() -> None:
    item = parse_item(expanded_book())
    docs = build_item_documents(item, BOOKS, CFG)
    types = [d.chunk_type for d in docs]
    assert types[0] == "item_metadata"
    meta = docs[0]
    assert "Audiobook by Becky Chambers, narrated by Patricia Rodriguez" in meta.text
    assert "Series: Wayfarers #1" in meta.text
    assert "Length: 15 hr 30 min" in meta.text
    assert "A crew of misfits" in meta.text
    assert meta.context.startswith(
        "Audiobook: The Long Way: Wayfarers 1 | By: Becky Chambers"
    )
    assert meta.payload["url"] == "https://abs.example.com/item/li-1"
    assert meta.payload["authors"] == ["Becky Chambers"]
    assert meta.payload["series"] == ["Wayfarers"]
    # short description -> no separate description chunks
    assert "description" not in types
    # meaningful chapter names -> chapter listing
    assert "chapters" in types
    chapters = next(d for d in docs if d.chunk_type == "chapters")
    assert "2. Transit (starts at 0:20:00" in chapters.text


def test_long_description_is_chunked() -> None:
    desc = " ".join(f"This is descriptive sentence {i}." for i in range(80))
    item = parse_item(expanded_book(description=desc))
    docs = build_item_documents(item, BOOKS, CFG)
    desc_docs = [d for d in docs if d.chunk_type == "description"]
    assert len(desc_docs) > 1
    assert all(d.context.startswith("Audiobook: The Long Way") for d in desc_docs)
    assert len({d.point_id for d in docs}) == len(docs)


def test_generic_chapters_are_skipped() -> None:
    assert not chapters_are_meaningful(
        [Chapter("Chapter 1", 0, 1), Chapter("Chapter 2", 1, 2)]
    )
    assert not chapters_are_meaningful([Chapter("1", 0, 1), Chapter("2", 1, 2)])
    assert chapters_are_meaningful(
        [Chapter("Prologue", 0, 1), Chapter("Chapter 1", 1, 2)]
    )
    assert not chapters_are_meaningful([Chapter("Only", 0, 1)])


def test_chapters_can_be_disabled() -> None:
    item = parse_item(expanded_book())
    docs = build_item_documents(item, BOOKS, CFG, include_chapters=False)
    assert "chapters" not in {d.chunk_type for d in docs}


def test_podcast_episode_documents() -> None:
    item = parse_item(expanded_podcast(episodes=2))
    docs = build_item_documents(item, PODS, CFG)
    meta = docs[0]
    assert meta.context.startswith("Podcast: Hackerspace Hour | Host: PDX Hackers")
    assert "Episodes in library: 2" in meta.text
    eps = [d for d in docs if d.chunk_type == "episode"]
    assert len(eps) == 2
    ep = eps[0]
    assert ep.payload["episode_id"] == "ep-1"
    assert ep.payload["episode_published_at"].startswith("2024-01-02")
    assert "Season 1, Episode 1" in ep.text
    assert "laser cutters, part 1" in ep.text
    assert (
        "| Episode: Episode 1: Laser cutters | Published: Tuesday, January 2, 2024"
        in ep.context
    )


def test_content_hash_is_stable_and_sensitive() -> None:
    a = build_item_documents(parse_item(expanded_book()), BOOKS, CFG)[0]
    b = build_item_documents(parse_item(expanded_book()), BOOKS, CFG)[0]
    c = build_item_documents(parse_item(expanded_book(title="Other")), BOOKS, CFG)[0]
    assert a.content_hash == b.content_hash
    assert a.point_id == b.point_id
    assert a.content_hash != c.content_hash


def test_ebook_documents_and_signature() -> None:
    item = parse_item(expanded_book(ebook=True))
    sections = [
        TextSection("Chapter One", "Words. " * 200),
        TextSection(None, "Tail text."),
    ]
    sig = ebook_signature(item, CFG)
    docs = build_ebook_documents(item, BOOKS, sections, CFG, ebook_signature=sig)
    assert len(docs) > 2
    assert docs[0].payload["section_title"] == "Chapter One"
    assert "| Section: Chapter One" in docs[0].context
    assert docs[0].payload["ebook_format"] == "epub"
    assert all(d.payload["ebook_signature"] == sig for d in docs)
    other = parse_item(expanded_book(ebook=True, title="Renamed"))
    assert ebook_signature(other, CFG) != sig


def test_author_documents() -> None:
    items = [parse_item(minified(expanded_book()))]
    grouped = books_by_author(items)
    author = parse_author(
        {
            "id": "au-1",
            "name": "Becky Chambers",
            "description": "<p>Writes hopeful SF.</p>",
        }
    )
    docs = build_author_documents(author, BOOKS, grouped["becky chambers"], CFG)
    assert len(docs) == 2
    assert "Books by Becky Chambers in the Audiobooks library (1):" in docs[0].text
    assert "- The Long Way (Wayfarers #1), 2014" in docs[0].text
    assert docs[1].text == "About Becky Chambers: Writes hopeful SF."
    assert docs[0].payload["url"] == "https://abs.example.com/author/au-1"
    assert docs[0].payload["item_id"] is None


def test_author_without_books_or_description_is_skipped() -> None:
    author = parse_author({"id": "x", "name": "Nobody"})
    assert build_author_documents(author, BOOKS, [], CFG) == []


def test_book_library_summary() -> None:
    items = [
        parse_item(minified(expanded_book("a", title="Alpha"))),
        parse_item(minified(expanded_book("b", title="Beta"))),
    ]
    docs = build_library_summary_documents(BOOKS, items, CFG)
    sections = {d.payload["summary_section"] for d in docs}
    assert {
        "overview",
        "authors",
        "narrators",
        "series",
        "genres",
        "catalog",
    } <= sections
    overview = next(d for d in docs if d.payload["summary_section"] == "overview")
    assert 'The library "Audiobooks" contains 2 books' in overview.text
    authors = next(d for d in docs if d.payload["summary_section"] == "authors")
    assert "- Becky Chambers: 2 books" in authors.text
    assert all(d.chunk_type == "library_summary" for d in docs)


def test_podcast_library_summary() -> None:
    items = [parse_item(minified(expanded_podcast(episodes=4)))]
    docs = build_library_summary_documents(PODS, items, CFG)
    text = "\n".join(d.text for d in docs)
    assert "contains 1 podcast with 4 downloaded episodes" in text
    assert "- Hackerspace Hour (hosted by PDX Hackers): 4 episodes" in text
