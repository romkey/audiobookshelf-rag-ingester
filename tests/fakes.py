"""Fake Audiobookshelf data, client and embedder shared by the tests."""

import copy
import hashlib
from collections.abc import Callable
from typing import Any

from audiobookshelf_rag_ingester.embeddings import EmbedKind

# Any: these build and return ABS API JSON, which is free-form.
type JSON = dict[str, Any]

BOOK_LIBRARY: JSON = {"id": "lib-books", "name": "Audiobooks", "mediaType": "book"}
PODCAST_LIBRARY: JSON = {"id": "lib-pods", "name": "Podcasts", "mediaType": "podcast"}


def expanded_book(
    item_id: str = "li-1",
    *,
    title: str = "The Long Way to a Small, Angry Planet",
    updated_at: int = 1_700_000_000_000,
    description: str = "<p>A crew of misfits tunnels through space.</p>",
    authors: tuple[str, ...] = ("Becky Chambers",),
    narrators: tuple[str, ...] = ("Rachel Dulude",),
    series: str | None = "Wayfarers",
    sequence: str | None = "1",
    isbn: str | None = None,
    asin: str | None = "B01M0N8Q5I",
    genres: tuple[str, ...] = ("Science Fiction",),
    duration: float = 3600 * 15.5,
    library_id: str = "lib-books",
    ebook: bool = False,
) -> JSON:
    return {
        "id": item_id,
        "libraryId": library_id,
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
                "subtitle": None,
                "authors": [{"id": f"au-{n}", "name": a} for n, a in enumerate(authors)],
                "narrators": list(narrators),
                "series": (
                    [{"id": "se-1", "name": series, "sequence": sequence}] if series else []
                ),
                "genres": list(genres),
                "publishedYear": "2014",
                "publisher": "Harper Voyager",
                "description": description,
                "isbn": isbn,
                "asin": asin,
                "language": "English",
                "explicit": False,
                "abridged": False,
                "authorName": ", ".join(authors),
                "narratorName": ", ".join(narrators),
                "seriesName": f"{series} #{sequence}" if series else "",
            },
            "coverPath": f"/audiobooks/{item_id}/cover.jpg",
            "tags": ["favorites"],
            "duration": duration,
            "numChapters": 3,
            "chapters": [
                {"id": 0, "start": 0, "end": 1200, "title": "Opening Credits"},
                {"id": 1, "start": 1200, "end": 4000, "title": "Transit"},
                {"id": 2, "start": 4000, "end": 55800, "title": "The Long Way"},
            ],
            "audioFiles": (
                [{"index": 1, "duration": duration, "metadata": {"ext": ".m4b", "size": 9000}}]
                if duration
                else []
            ),
            "ebookFile": {"ino": "9", "ebookFormat": "epub", "metadata": {}} if ebook else None,
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
        media["numAudioFiles"] = len(media.pop("audioFiles", []))
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
        self.calls: list[str] = []
        self.fail_items: set[str] = set()
        self.fail_libraries: set[str] = set()
        self.on_item: Callable[[str], None] | None = None

    def add(self, expanded: JSON) -> None:
        self.items[expanded["id"]] = expanded

    def list_libraries(self) -> list[JSON]:
        self.calls.append("libraries")
        return list(self.libraries)

    def list_library_items(self, library_id: str) -> list[JSON]:
        self.calls.append(f"items:{library_id}")
        if library_id in self.fail_libraries:
            raise RuntimeError("listing failed")
        return [minified(item) for item in self.items.values() if item["libraryId"] == library_id]

    def get_item(self, item_id: str) -> JSON:
        self.calls.append(f"item:{item_id}")
        if self.on_item is not None:
            self.on_item(item_id)
        if item_id in self.fail_items:
            raise RuntimeError("boom")
        return copy.deepcopy(self.items[item_id])


class FakeEmbedder:
    """Deterministic 8-dimension vectors from a hash of the text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        self.calls.append(list(chunks))
        out = []
        for t in chunks:
            digest = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 + 0.01 for b in digest[:8]])
        return out

    @property
    def texts(self) -> list[str]:
        return [t for call in self.calls for t in call]
