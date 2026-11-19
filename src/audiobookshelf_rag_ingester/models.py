"""Normalized views of Audiobookshelf JSON.

The ABS API returns different shapes for minified (list) and expanded
(detail) items, and for books and podcasts; field names follow the server's
``server/models/LibraryItem.js``, ``Book.js``, ``Podcast.js`` and
``PodcastEpisode.js``. Only catalog fields are read: listening progress,
sessions, bookmarks and users are never part of an item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def index(self) -> float | None:
        try:
            return float(self.sequence) if self.sequence else None
        except ValueError:
            return None


@dataclass(frozen=True)
class Chapter:
    title: str
    start: float
    end: float


@dataclass(frozen=True)
class AudioFile:
    ext: str | None
    size: int | None
    duration: float | None


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


@dataclass(frozen=True)
class ListedItem:
    """A minified item from a library listing: enough to tell what changed."""

    id: str
    library_id: str
    media_type: str
    title: str
    updated_at: int | None


@dataclass(frozen=True)
class LibraryItem:
    id: str
    library_id: str
    media_type: str  # "book" | "podcast"
    title: str
    subtitle: str | None
    authors: list[str]
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
    cover_path: str | None
    added_at: int | None
    updated_at: int | None
    chapters: list[Chapter] = field(default_factory=list)
    audio_files: list[AudioFile] = field(default_factory=list)
    has_ebook: bool = False
    # Podcast-only
    podcast_type: str | None = None
    episodes: list[Episode] = field(default_factory=list)

    # Any: a JSON-ready dict of the record.
    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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


def parse_listed(data: dict[str, Any]) -> ListedItem:
    media = data.get("media") or {}
    metadata = media.get("metadata") or {}
    return ListedItem(
        id=str(data["id"]),
        library_id=str(data.get("libraryId") or ""),
        media_type=str(data.get("mediaType") or "book"),
        title=_str_or_none(metadata.get("title")) or "Untitled",
        updated_at=_int_or_none(data.get("updatedAt")),
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
        return [
            SeriesRef(str(s["name"]), _str_or_none(s.get("sequence")))
            for s in raw
            if isinstance(s, dict) and s.get("name")
        ]
    if isinstance(raw, dict) and raw.get("name"):
        return [SeriesRef(str(raw["name"]), _str_or_none(raw.get("sequence")))]
    # Minified items only carry "Series Name #1, Other Series #2"
    refs = []
    for part in _split_names(metadata.get("seriesName")):
        name, sep, seq = part.rpartition(" #")
        refs.append(
            SeriesRef(name.strip(), seq.strip() or None) if sep and name else SeriesRef(part)
        )
    return refs


def _parse_audio_files(raw: Any) -> list[AudioFile]:
    files: list[AudioFile] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata") or {}
        files.append(
            AudioFile(
                ext=_str_or_none(str(meta.get("ext") or "").lstrip(".").upper()),
                size=_int_or_none(meta.get("size")),
                duration=_float_or_none(entry.get("duration")),
            )
        )
    return files


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
    )


def parse_item(data: dict[str, Any]) -> LibraryItem:
    """Parse an expanded ABS library item (``GET /api/items/:id?expanded=1``)."""
    media = data.get("media") or {}
    metadata = media.get("metadata") or {}
    media_type = str(data.get("mediaType") or "book")

    raw_authors = metadata.get("authors")
    if isinstance(raw_authors, list) and raw_authors:
        authors = [str(a["name"]) for a in raw_authors if isinstance(a, dict) and a.get("name")]
    elif media_type == "podcast":
        authors = [str(metadata["author"])] if metadata.get("author") else []
    else:
        authors = _split_names(metadata.get("authorName"))

    narrators = _str_list(metadata.get("narrators")) or _split_names(metadata.get("narratorName"))
    episodes = [_parse_episode(e) for e in (media.get("episodes") or []) if isinstance(e, dict)]

    return LibraryItem(
        id=str(data["id"]),
        library_id=str(data.get("libraryId") or ""),
        media_type=media_type,
        title=_str_or_none(metadata.get("title")) or "Untitled",
        subtitle=_str_or_none(metadata.get("subtitle")),
        authors=authors,
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
        cover_path=_str_or_none(media.get("coverPath")),
        added_at=_int_or_none(data.get("addedAt")),
        updated_at=_int_or_none(data.get("updatedAt")),
        chapters=parse_chapters(media.get("chapters")),
        audio_files=_parse_audio_files(media.get("audioFiles")),
        has_ebook=bool(media.get("ebookFile")),
        podcast_type=_str_or_none(metadata.get("type")),
        episodes=episodes,
    )
