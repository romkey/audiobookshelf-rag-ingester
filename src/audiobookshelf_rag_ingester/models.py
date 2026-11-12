"""Normalized views of Audiobookshelf JSON.

The ABS API returns slightly different shapes for minified (list) and
expanded (detail) items, and for books vs. podcasts. Everything downstream
works with these dataclasses instead of raw dicts.
"""

from dataclasses import dataclass, field
from typing import Any  # Any: decoded JSON from the Audiobookshelf API


@dataclass(frozen=True)
class Library:
    id: str
    name: str
    media_type: str  # "book" | "podcast"


@dataclass(frozen=True)
class SeriesRef:
    name: str
    sequence: str | None = None

    def label(self) -> str:
        return f"{self.name} #{self.sequence}" if self.sequence else self.name


@dataclass(frozen=True)
class Chapter:
    title: str
    start: float
    end: float


@dataclass(frozen=True)
class EbookFile:
    filename: str | None
    format: str | None
    size: int | None
    ino: str | None
    mtime_ms: int | None

    def signature(self) -> str:
        return f"{self.ino}:{self.size}:{self.mtime_ms}"


@dataclass(frozen=True)
class Episode:
    id: str
    title: str
    subtitle: str | None
    description: str | None
    season: str | None
    episode: str | None
    episode_type: str | None
    pub_date: str | None
    published_at: int | None
    duration: float | None
    chapters: list[Chapter] = field(default_factory=list)


@dataclass(frozen=True)
class LibraryItem:
    id: str
    library_id: str
    media_type: str  # "book" | "podcast"
    title: str
    subtitle: str | None
    authors: list[str]
    author_ids: dict[str, str]  # name -> ABS author id (expanded items only)
    narrators: list[str]
    series: list[SeriesRef]
    genres: list[str]
    tags: list[str]
    description: str | None
    publisher: str | None
    published_year: str | None
    published_date: str | None
    language: str | None
    isbn: str | None
    asin: str | None
    explicit: bool
    abridged: bool
    duration: float | None
    size: int | None
    num_chapters: int
    num_episodes: int
    added_at: int | None
    updated_at: int | None
    mtime_ms: int | None
    is_missing: bool
    is_invalid: bool
    rel_path: str | None
    # Podcast-only metadata
    feed_url: str | None = None
    itunes_page_url: str | None = None
    podcast_type: str | None = None
    release_date: str | None = None
    # Expanded-only data
    chapters: list[Chapter] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    ebook: EbookFile | None = None
    ebook_format: str | None = None

    def fingerprint(self) -> str:
        """Cheap change detector built from fields present on minified list items."""
        return ":".join(
            str(v)
            for v in (
                self.updated_at,
                self.mtime_ms,
                self.size,
                self.num_episodes,
                self.num_chapters,
                self.ebook_format,
            )
        )


@dataclass(frozen=True)
class Author:
    id: str
    name: str
    description: str | None
    asin: str | None
    num_books: int | None
    updated_at: int | None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except TypeError, ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except TypeError, ValueError:
        return None


def _split_names(value: Any) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def parse_library(data: dict[str, Any]) -> Library:
    return Library(
        id=str(data["id"]),
        name=str(data.get("name") or data["id"]),
        media_type=str(data.get("mediaType") or "book"),
    )


def parse_chapters(raw: Any) -> list[Chapter]:
    chapters: list[Chapter] = []
    if not isinstance(raw, list):
        return chapters
    for idx, ch in enumerate(raw):
        if not isinstance(ch, dict):
            continue
        chapters.append(
            Chapter(
                title=_str_or_none(ch.get("title")) or f"Chapter {idx + 1}",
                start=_float_or_none(ch.get("start")) or 0.0,
                end=_float_or_none(ch.get("end")) or 0.0,
            )
        )
    return chapters


def _parse_series(metadata: dict[str, Any]) -> list[SeriesRef]:
    raw = metadata.get("series")
    if isinstance(raw, list) and raw:
        refs = []
        for s in raw:
            if isinstance(s, dict) and s.get("name"):
                refs.append(SeriesRef(str(s["name"]), _str_or_none(s.get("sequence"))))
        return refs
    if isinstance(raw, dict) and raw.get("name"):
        return [SeriesRef(str(raw["name"]), _str_or_none(raw.get("sequence")))]
    # Minified items only carry "Series Name #1, Other Series #2"
    refs = []
    for part in _split_names(metadata.get("seriesName")):
        name, sep, seq = part.rpartition(" #")
        if sep and name:
            refs.append(SeriesRef(name.strip(), seq.strip() or None))
        else:
            refs.append(SeriesRef(part))
    return refs


def _parse_ebook(raw: Any) -> EbookFile | None:
    if not isinstance(raw, dict):
        return None
    meta = raw.get("metadata") or {}
    return EbookFile(
        filename=_str_or_none(meta.get("filename")),
        format=_str_or_none(raw.get("ebookFormat") or meta.get("ext", "").lstrip(".")),
        size=_int_or_none(meta.get("size")),
        ino=_str_or_none(raw.get("ino")),
        mtime_ms=_int_or_none(meta.get("mtimeMs")),
    )


def _parse_episode(raw: dict[str, Any]) -> Episode:
    audio = raw.get("audioFile") or {}
    duration = _float_or_none(raw.get("duration"))
    if duration is None:
        duration = _float_or_none(audio.get("duration"))
    return Episode(
        id=str(raw.get("id")),
        title=_str_or_none(raw.get("title")) or "Untitled episode",
        subtitle=_str_or_none(raw.get("subtitle")),
        description=_str_or_none(raw.get("description")),
        season=_str_or_none(raw.get("season")),
        episode=_str_or_none(raw.get("episode")),
        episode_type=_str_or_none(raw.get("episodeType")),
        pub_date=_str_or_none(raw.get("pubDate")),
        published_at=_int_or_none(raw.get("publishedAt")),
        duration=duration,
        chapters=parse_chapters(raw.get("chapters")),
    )


def parse_item(data: dict[str, Any]) -> LibraryItem:
    """Parse a minified or expanded ABS library item."""
    media = data.get("media") or {}
    metadata = media.get("metadata") or {}
    media_type = str(data.get("mediaType") or "book")

    raw_authors = metadata.get("authors")
    author_ids: dict[str, str] = {}
    if isinstance(raw_authors, list) and raw_authors:
        authors = []
        for a in raw_authors:
            if isinstance(a, dict) and a.get("name"):
                authors.append(str(a["name"]))
                if a.get("id"):
                    author_ids[str(a["name"])] = str(a["id"])
    elif media_type == "podcast":
        authors = [metadata["author"]] if metadata.get("author") else []
    else:
        authors = _split_names(metadata.get("authorName"))

    narrators = _str_list(metadata.get("narrators")) or _split_names(
        metadata.get("narratorName")
    )

    chapters = parse_chapters(media.get("chapters"))
    episodes = [
        _parse_episode(e) for e in (media.get("episodes") or []) if isinstance(e, dict)
    ]
    ebook = _parse_ebook(media.get("ebookFile"))

    num_chapters = _int_or_none(media.get("numChapters"))
    num_episodes = _int_or_none(media.get("numEpisodes"))

    return LibraryItem(
        id=str(data["id"]),
        library_id=str(data.get("libraryId") or ""),
        media_type=media_type,
        title=_str_or_none(metadata.get("title")) or "Untitled",
        subtitle=_str_or_none(metadata.get("subtitle")),
        authors=authors,
        author_ids=author_ids,
        narrators=narrators,
        series=_parse_series(metadata) if media_type == "book" else [],
        genres=_str_list(metadata.get("genres")),
        tags=_str_list(media.get("tags")),
        description=_str_or_none(metadata.get("description")),
        publisher=_str_or_none(metadata.get("publisher")),
        published_year=_str_or_none(metadata.get("publishedYear")),
        published_date=_str_or_none(metadata.get("publishedDate")),
        language=_str_or_none(metadata.get("language")),
        isbn=_str_or_none(metadata.get("isbn")),
        asin=_str_or_none(metadata.get("asin")),
        explicit=bool(metadata.get("explicit")),
        abridged=bool(metadata.get("abridged")),
        duration=_float_or_none(media.get("duration")),
        size=_int_or_none(data.get("size") or media.get("size")),
        num_chapters=num_chapters if num_chapters is not None else len(chapters),
        num_episodes=num_episodes if num_episodes is not None else len(episodes),
        added_at=_int_or_none(data.get("addedAt")),
        updated_at=_int_or_none(data.get("updatedAt")),
        mtime_ms=_int_or_none(data.get("mtimeMs")),
        is_missing=bool(data.get("isMissing")),
        is_invalid=bool(data.get("isInvalid")),
        rel_path=_str_or_none(data.get("relPath")),
        feed_url=_str_or_none(metadata.get("feedUrl")),
        itunes_page_url=_str_or_none(metadata.get("itunesPageUrl")),
        podcast_type=_str_or_none(metadata.get("type")),
        release_date=_str_or_none(metadata.get("releaseDate")),
        chapters=chapters,
        episodes=episodes,
        ebook=ebook,
        ebook_format=_str_or_none(media.get("ebookFormat"))
        or (ebook.format if ebook else None),
    )


def parse_author(data: dict[str, Any]) -> Author:
    return Author(
        id=str(data["id"]),
        name=str(data.get("name") or "Unknown"),
        description=_str_or_none(data.get("description")),
        asin=_str_or_none(data.get("asin")),
        num_books=_int_or_none(data.get("numBooks")),
        updated_at=_int_or_none(data.get("updatedAt")),
    )
