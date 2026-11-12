"""Fake Audiobookshelf data, client and embedder shared by the tests."""

import copy
import hashlib
import io
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from audiobookshelf_rag_ingester.config import Settings

# Any: these build and return ABS API JSON, which is free-form.
type JSON = dict[str, Any]

BOOK_LIBRARY: JSON = {"id": "lib-books", "name": "Audiobooks", "mediaType": "book"}
PODCAST_LIBRARY: JSON = {"id": "lib-pods", "name": "Podcasts", "mediaType": "podcast"}


def expanded_book(
    item_id: str = "li-1",
    *,
    title: str = "The Long Way",
    updated_at: int = 1_700_000_000_000,
    description: str = "<p>A crew of misfits tunnels through space.</p>",
    ebook: bool = False,
) -> JSON:
    return {
        "id": item_id,
        "libraryId": "lib-books",
        "mediaType": "book",
        "addedAt": 1_690_000_000_000,
        "updatedAt": updated_at,
        "mtimeMs": 1_690_000_000_000,
        "size": 123456,
        "relPath": f"Author/{title}",
        "isMissing": False,
        "isInvalid": False,
        "media": {
            "metadata": {
                "title": title,
                "subtitle": "Wayfarers 1",
                "authors": [{"id": "au-1", "name": "Becky Chambers"}],
                "narrators": ["Patricia Rodriguez"],
                "series": [{"id": "se-1", "name": "Wayfarers", "sequence": "1"}],
                "genres": ["Science Fiction"],
                "publishedYear": "2014",
                "publisher": "Harper Voyager",
                "description": description,
                "isbn": "9780062444134",
                "asin": "B01M0N8Q5I",
                "language": "English",
                "explicit": False,
                "abridged": False,
                "authorName": "Becky Chambers",
                "narratorName": "Patricia Rodriguez",
                "seriesName": "Wayfarers #1",
            },
            "tags": ["favorites"],
            "duration": 3600 * 15.5,
            "numChapters": 3,
            "chapters": [
                {"id": 0, "start": 0, "end": 1200, "title": "Opening Credits"},
                {"id": 1, "start": 1200, "end": 4000, "title": "Transit"},
                {"id": 2, "start": 4000, "end": 55800, "title": "The Long Way"},
            ],
            "ebookFile": (
                {
                    "ino": "999",
                    "ebookFormat": "epub",
                    "metadata": {
                        "filename": "book.epub",
                        "ext": ".epub",
                        "size": 1000,
                        "mtimeMs": 1_690_000_000_000,
                    },
                }
                if ebook
                else None
            ),
        },
    }


def minified(expanded: JSON) -> JSON:
    """The shape /api/libraries/:id/items returns for an expanded item."""
    data = copy.deepcopy(expanded)
    media = data["media"]
    meta = media["metadata"]
    if data["mediaType"] == "book":
        meta.pop("authors", None)
        meta.pop("narrators", None)
        meta.pop("series", None)
        media.pop("chapters", None)
        ebook = media.pop("ebookFile", None)
        media["ebookFormat"] = ebook["ebookFormat"] if ebook else None
    else:
        media["numEpisodes"] = len(media.pop("episodes", []))
    return data


def expanded_podcast(item_id: str = "li-p1", episodes: int = 2) -> JSON:
    return {
        "id": item_id,
        "libraryId": "lib-pods",
        "mediaType": "podcast",
        "addedAt": 1_690_000_000_000,
        "updatedAt": 1_700_000_000_000,
        "mtimeMs": 1_690_000_000_000,
        "size": 5555,
        "media": {
            "metadata": {
                "title": "Hackerspace Hour",
                "author": "PDX Hackers",
                "description": "Weekly chat about making things.",
                "genres": ["Technology"],
                "feedUrl": "https://example.com/feed.xml",
                "type": "episodic",
                "language": "en",
                "explicit": False,
            },
            "tags": [],
            "episodes": [
                {
                    "id": f"ep-{n}",
                    "index": n,
                    "season": "1",
                    "episode": str(n),
                    "episodeType": "full",
                    "title": f"Episode {n}: Laser cutters",
                    "description": f"<p>We talk about laser cutters, part {n}.</p>",
                    "pubDate": "Mon, 01 Jan 2024 00:00:00 GMT",
                    "publishedAt": 1_704_067_200_000 + n * 86_400_000,
                    "audioFile": {"duration": 1800.0},
                    "duration": 1800.0,
                    "chapters": [],
                }
                for n in range(1, episodes + 1)
            ],
        },
    }


class FakeAbsClient:
    """Stands in for AbsClient: an in-memory Audiobookshelf."""

    def __init__(self) -> None:
        self.libraries: list[JSON] = [BOOK_LIBRARY]
        self.items: dict[str, JSON] = {}
        self.authors: dict[str, list[JSON]] = {
            "lib-books": [
                {
                    "id": "au-1",
                    "name": "Becky Chambers",
                    "description": "Becky Chambers writes hopeful science fiction.",
                    "numBooks": 1,
                }
            ]
        }
        self.ebook_bytes: bytes | None = None
        self.calls: list[str] = []
        self.fail_items: set[str] = set()
        self.on_item: Callable[[str], None] | None = None

    def add(self, expanded: JSON) -> None:
        self.items[expanded["id"]] = expanded

    def list_libraries(self) -> list[JSON]:
        self.calls.append("libraries")
        return list(self.libraries)

    def list_library_items(self, library_id: str) -> list[JSON]:
        self.calls.append(f"items:{library_id}")
        return [
            minified(item)
            for item in self.items.values()
            if item["libraryId"] == library_id
        ]

    def get_item(self, item_id: str) -> JSON:
        self.calls.append(f"item:{item_id}")
        if self.on_item is not None:
            self.on_item(item_id)
        if item_id in self.fail_items:
            raise RuntimeError("boom")
        return copy.deepcopy(self.items[item_id])

    def list_authors(self, library_id: str) -> list[JSON]:
        return list(self.authors.get(library_id, []))

    def download_ebook(self, item_id: str, max_bytes: int) -> bytes | None:
        self.calls.append(f"ebook:{item_id}")
        return self.ebook_bytes


class FakeEmbedder:
    """Deterministic 8-dim vectors derived from a hash of the text."""

    dimension = 8

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 + 0.01 for b in digest[:8]])
        return out


BASE_SETTINGS = Settings(
    abs_url="http://abs.local",
    abs_api_token="test-token",  # noqa: S106 - dummy test token; not-a-secret
    abs_public_url="https://abs.example.com",
    chunk_size=400,
    chunk_overlap=40,
    poll_interval_seconds=0,
)


# Any: test overrides for any Settings field.
def make_settings(**overrides: Any) -> Settings:
    return replace(BASE_SETTINGS, **overrides)


def build_epub_bytes(chapters: list[tuple[str, str]]) -> bytes:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id123")
    book.set_title("Test Book")
    book.set_language("en")
    items = []
    for idx, (title, body) in enumerate(chapters):
        ch = epub.EpubHtml(title=title, file_name=f"ch{idx}.xhtml", lang="en")
        ch.content = f"<html><body><h1>{title}</h1><p>{body}</p></body></html>"
        book.add_item(ch)
        items.append(ch)
    book.toc = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    buf = io.BytesIO()
    epub.write_epub(buf, book)
    return buf.getvalue()
