"""Turn Audiobookshelf records into embeddable documents.

Everything here is pure (no I/O) so it is easy to test. Each ``Document`` has a
deterministic ``key`` that becomes the Qdrant point id, plus a ``content_hash``
used to skip re-embedding chunks that have not changed.
"""

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any  # Any: point payloads are free-form JSON values

from audiobookshelf_rag_ingester.extractors import TextSection
from audiobookshelf_rag_ingester.models import (
    Author,
    Chapter,
    Episode,
    Library,
    LibraryItem,
)
from audiobookshelf_rag_ingester.text_utils import (
    format_duration,
    format_timestamp,
    ms_to_iso,
    split_lines,
    split_text,
    strip_html,
)

CHUNK_TYPES = (
    "item_metadata",
    "description",
    "chapters",
    "episode",
    "ebook_content",
    "author",
    "library_summary",
)


@dataclass(frozen=True)
class DocConfig:
    chunk_size: int
    chunk_overlap: int
    public_url: str


@dataclass
class Document:
    key: str
    chunk_type: str
    text: str
    context: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return point_id(self.key)

    @property
    def content_hash(self) -> str:
        blob = json.dumps(
            {"context": self.context, "payload": self.payload},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"audiobookshelf_rag_ingester:{key}"))


# ---------------------------------------------------------------------------
# URLs and shared payload
# ---------------------------------------------------------------------------


def item_url(public_url: str, item_id: str) -> str:
    return f"{public_url.rstrip('/')}/item/{item_id}"


def author_url(public_url: str, author_id: str) -> str:
    return f"{public_url.rstrip('/')}/author/{author_id}"


def library_url(public_url: str, library_id: str) -> str:
    return f"{public_url.rstrip('/')}/library/{library_id}"


def _base_payload(library: Library) -> dict[str, Any]:
    return {
        "source": "audiobookshelf",
        "library_id": library.id,
        "library_name": library.name,
        "media_type": library.media_type,
        "item_id": None,
        "title": None,
        "subtitle": None,
        "authors": [],
        "narrators": [],
        "series": [],
        "genres": [],
        "tags": [],
        "published_year": None,
        "publisher": None,
        "language": None,
        "isbn": None,
        "asin": None,
        "explicit": None,
        "duration_seconds": None,
        "duration": None,
        "url": None,
        "episode_id": None,
        "episode_title": None,
        "episode_published_at": None,
        "section_title": None,
        "author_id": None,
        "author_name": None,
    }


def _item_payload(
    item: LibraryItem, library: Library, cfg: DocConfig
) -> dict[str, Any]:
    payload = _base_payload(library)
    payload.update(
        {
            "media_type": item.media_type,
            "item_id": item.id,
            "title": item.title,
            "subtitle": item.subtitle,
            "authors": item.authors,
            "narrators": item.narrators,
            "series": [s.name for s in item.series],
            "series_labels": [s.label() for s in item.series],
            "genres": item.genres,
            "tags": item.tags,
            "published_year": item.published_year,
            "publisher": item.publisher,
            "language": item.language,
            "isbn": item.isbn,
            "asin": item.asin,
            "explicit": item.explicit,
            "duration_seconds": round(item.duration, 2) if item.duration else None,
            "duration": format_duration(item.duration),
            "has_ebook": bool(item.ebook_format),
            "added_at": ms_to_iso(item.added_at),
            "url": item_url(cfg.public_url, item.id),
        }
    )
    return payload


def _kind(item: LibraryItem) -> str:
    if item.media_type == "podcast":
        return "Podcast"
    if not item.duration and item.ebook_format:
        return "Ebook"
    return "Audiobook"


def _full_title(item: LibraryItem) -> str:
    return f"{item.title}: {item.subtitle}" if item.subtitle else item.title


def item_header(item: LibraryItem) -> str:
    """Short context prefix so every chunk's embedding knows which item it is from."""
    parts = [f"{_kind(item)}: {_full_title(item)}"]
    if item.authors:
        label = "Host" if item.media_type == "podcast" else "By"
        parts.append(f"{label}: {', '.join(item.authors)}")
    if item.narrators:
        parts.append(f"Narrated by: {', '.join(item.narrators)}")
    if item.series:
        parts.append(f"Series: {', '.join(s.label() for s in item.series)}")
    return " | ".join(parts)


def _contextualize(header: str, text: str) -> str:
    return f"{header}\n\n{text}" if header else text


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# ---------------------------------------------------------------------------
# Item documents
# ---------------------------------------------------------------------------


def _metadata_text(item: LibraryItem, library: Library, description: str) -> str:
    kind = _kind(item)
    lines = [_full_title(item)]
    if item.media_type == "podcast":
        if item.authors:
            lines.append(f"Podcast hosted by {_join_names(item.authors)}.")
        else:
            lines.append("Podcast.")
    else:
        byline = f"{kind}"
        if item.authors:
            byline += f" by {_join_names(item.authors)}"
        if item.narrators:
            byline += f", narrated by {_join_names(item.narrators)}"
        lines.append(byline + ".")

    details: list[str] = []
    if item.series:
        details.append(f"Series: {', '.join(s.label() for s in item.series)}")
    if item.genres:
        details.append(f"Genres: {', '.join(item.genres)}")
    if item.tags:
        details.append(f"Tags: {', '.join(item.tags)}")
    if item.published_year or item.publisher:
        published = "Published"
        if item.published_year:
            published += f" {item.published_year}"
        if item.publisher:
            published += f" by {item.publisher}"
        details.append(published)
    if item.language:
        details.append(f"Language: {item.language}")
    if item.isbn:
        details.append(f"ISBN: {item.isbn}")
    if item.asin:
        details.append(f"ASIN: {item.asin}")
    length = format_duration(item.duration)
    if length:
        details.append(f"Length: {length}")
    if item.media_type == "book" and item.num_chapters:
        details.append(f"Chapters: {item.num_chapters}")
    if item.media_type == "podcast":
        details.append(f"Episodes in library: {item.num_episodes}")
        if item.podcast_type:
            details.append(f"Podcast type: {item.podcast_type}")
    if item.abridged:
        details.append("Abridged")
    if item.explicit:
        details.append("Explicit content")
    if item.ebook_format and item.duration:
        details.append(f"Ebook also available ({item.ebook_format.upper()})")
    elif item.ebook_format:
        details.append(f"Format: {item.ebook_format.upper()}")
    details.append(f"Library: {library.name}")
    lines.append(". ".join(details) + ".")

    if description:
        lines.append("")
        lines.append(description)
    return "\n".join(lines)


_GENERIC_CHAPTER = re.compile(
    r"^\s*(chapter|track|part|ch\.?)?\s*[\divxlc]+\s*$", re.IGNORECASE
)


def chapters_are_meaningful(chapters: list[Chapter]) -> bool:
    if len(chapters) < 2:
        return False
    return any(not _GENERIC_CHAPTER.match(ch.title) for ch in chapters)


def _chapter_lines(chapters: list[Chapter]) -> list[str]:
    lines = []
    for idx, ch in enumerate(chapters, 1):
        length = format_duration(ch.end - ch.start) if ch.end > ch.start else None
        line = f"{idx}. {ch.title} (starts at {format_timestamp(ch.start)}"
        line += f", {length})" if length else ")"
        lines.append(line)
    return lines


def _episode_text(ep: Episode) -> str:
    lines = [f"Episode: {ep.title}"]
    if ep.subtitle and ep.subtitle != ep.title:
        lines.append(ep.subtitle)
    numbering = []
    if ep.season:
        numbering.append(f"Season {ep.season}")
    if ep.episode:
        numbering.append(f"Episode {ep.episode}")
    if ep.episode_type and ep.episode_type != "full":
        numbering.append(ep.episode_type.capitalize())
    if numbering:
        lines.append(", ".join(numbering))
    published = _episode_date(ep)
    if published:
        lines.append(f"Published: {published}")
    length = format_duration(ep.duration)
    if length:
        lines.append(f"Length: {length}")
    if chapters_are_meaningful(ep.chapters):
        lines.append("Chapters: " + "; ".join(ch.title for ch in ep.chapters))
    desc = strip_html(ep.description)
    if desc:
        lines.append("")
        lines.append(desc)
    return "\n".join(lines)


def _episode_date(ep: Episode) -> str | None:
    if ep.published_at:
        try:
            dt = datetime.fromtimestamp(ep.published_at / 1000, tz=UTC)
            return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year}"
        except OverflowError, OSError, ValueError:
            pass
    return ep.pub_date


def build_item_documents(
    item: LibraryItem,
    library: Library,
    cfg: DocConfig,
    *,
    include_chapters: bool = True,
    include_episodes: bool = True,
) -> list[Document]:
    docs: list[Document] = []
    header = item_header(item)
    base = _item_payload(item, library, cfg)

    description = strip_html(item.description)
    desc_chunks = (
        split_text(description, cfg.chunk_size, cfg.chunk_overlap)
        if description
        else []
    )

    # 1. One catalog/metadata point per item. Includes the start of the description
    #    so "a book about X" queries can match it.
    metadata_text = _metadata_text(item, library, desc_chunks[0] if desc_chunks else "")
    docs.append(
        Document(
            key=f"item:{item.id}:metadata",
            chunk_type="item_metadata",
            text=metadata_text,
            context=_contextualize(header, metadata_text),
            payload={**base, "chunk_index": 0, "total_chunks": 1},
        )
    )

    # 2. Full description, only when it did not fit in the metadata point.
    if len(desc_chunks) > 1:
        for idx, chunk in enumerate(desc_chunks):
            docs.append(
                Document(
                    key=f"item:{item.id}:description:{idx}",
                    chunk_type="description",
                    text=chunk,
                    context=_contextualize(header, chunk),
                    payload={
                        **base,
                        "chunk_index": idx,
                        "total_chunks": len(desc_chunks),
                    },
                )
            )

    # 3. Chapter listing (audiobooks).
    if (
        include_chapters
        and item.media_type == "book"
        and chapters_are_meaningful(item.chapters)
    ):
        chapter_header = f'Chapters of "{item.title}" ({len(item.chapters)} chapters):'
        chunks = split_lines(
            _chapter_lines(item.chapters), cfg.chunk_size, chapter_header
        )
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    key=f"item:{item.id}:chapters:{idx}",
                    chunk_type="chapters",
                    text=chunk,
                    context=_contextualize(header, chunk),
                    payload={**base, "chunk_index": idx, "total_chunks": len(chunks)},
                )
            )

    # 4. Podcast episodes: one or more points per episode.
    if include_episodes and item.media_type == "podcast":
        for ep in item.episodes:
            ep_header = f"{header} | Episode: {ep.title}"
            published = _episode_date(ep)
            if published:
                ep_header += f" | Published: {published}"
            chunks = split_text(_episode_text(ep), cfg.chunk_size, cfg.chunk_overlap)
            ep_payload = {
                **base,
                "episode_id": ep.id,
                "episode_title": ep.title,
                "episode_season": ep.season,
                "episode_number": ep.episode,
                "episode_published_at": ms_to_iso(ep.published_at),
                "duration_seconds": round(ep.duration, 2) if ep.duration else None,
                "duration": format_duration(ep.duration),
            }
            for idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        key=f"item:{item.id}:episode:{ep.id}:{idx}",
                        chunk_type="episode",
                        text=chunk,
                        context=_contextualize(ep_header, chunk),
                        payload={
                            **ep_payload,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        },
                    )
                )

    return docs


def build_ebook_documents(
    item: LibraryItem,
    library: Library,
    sections: list[TextSection],
    cfg: DocConfig,
    *,
    ebook_signature: str,
) -> list[Document]:
    header = item_header(item)
    base = _item_payload(item, library, cfg)
    base.update(
        {
            "ebook_format": item.ebook.format if item.ebook else item.ebook_format,
            "ebook_filename": item.ebook.filename if item.ebook else None,
            "ebook_signature": ebook_signature,
        }
    )
    items: list[tuple[str | None, str]] = []
    for section in sections:
        for chunk in split_text(section.text, cfg.chunk_size, cfg.chunk_overlap):
            items.append((section.title, chunk))

    docs: list[Document] = []
    for idx, (section_title, chunk) in enumerate(items):
        ctx_header = f"{header} | Section: {section_title}" if section_title else header
        docs.append(
            Document(
                key=f"item:{item.id}:ebook:{idx}",
                chunk_type="ebook_content",
                text=chunk,
                context=_contextualize(ctx_header, chunk),
                payload={
                    **base,
                    "section_title": section_title,
                    "chunk_index": idx,
                    "total_chunks": len(items),
                },
            )
        )
    return docs


def ebook_signature(item: LibraryItem, cfg: DocConfig) -> str:
    """Changes with the ebook file, the item's identifying header, or chunking."""
    raw = "|".join(
        [
            item.ebook.signature() if item.ebook else str(item.ebook_format),
            item_header(item),
            str(cfg.chunk_size),
            str(cfg.chunk_overlap),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------


def _book_label(item: LibraryItem) -> str:
    label = item.title
    if item.series:
        label += f" ({item.series[0].label()})"
    if item.published_year:
        label += f", {item.published_year}"
    return label


def books_by_author(items: list[LibraryItem]) -> dict[str, list[LibraryItem]]:
    grouped: dict[str, list[LibraryItem]] = defaultdict(list)
    for item in items:
        if item.media_type != "book":
            continue
        for name in item.authors:
            grouped[name.casefold()].append(item)
    return grouped


def build_author_documents(
    author: Author,
    library: Library,
    books: list[LibraryItem],
    cfg: DocConfig,
) -> list[Document]:
    header = f"Author: {author.name}"
    base = _base_payload(library)
    base.update(
        {
            "author_id": author.id,
            "author_name": author.name,
            "authors": [author.name],
            "asin": author.asin,
            "url": author_url(cfg.public_url, author.id),
            "book_count": len(books) if books else author.num_books,
            "title": author.name,
        }
    )

    texts: list[str] = []
    if books:
        sorted_books = sorted(
            books, key=lambda b: (b.series[0].name if b.series else "", b.title)
        )
        list_header = (
            f"Books by {author.name} in the {library.name} library ({len(books)}):"
        )
        texts.extend(
            split_lines(
                [f"- {_book_label(b)}" for b in sorted_books],
                cfg.chunk_size,
                list_header,
            )
        )
    elif author.num_books:
        texts.append(
            f"{author.name} has {author.num_books} books in the {library.name} library."
        )

    description = strip_html(author.description)
    if description:
        texts.extend(
            f"About {author.name}: {c}"
            for c in split_text(description, cfg.chunk_size, cfg.chunk_overlap)
        )
    if not texts:
        return []

    return [
        Document(
            key=f"author:{library.id}:{author.id}:{idx}",
            chunk_type="author",
            text=text,
            context=_contextualize(header, text),
            payload={**base, "chunk_index": idx, "total_chunks": len(texts)},
        )
        for idx, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Library-level summary
# ---------------------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _count_lines(counter: Counter[str], noun: str) -> list[str]:
    return [
        f"- {name}: {_plural(count, noun)}" for name, count in counter.most_common()
    ]


def build_library_summary_documents(
    library: Library,
    items: list[LibraryItem],
    cfg: DocConfig,
) -> list[Document]:
    sections: list[tuple[str, str, list[str]]] = []  # (section, header, lines)
    lname = library.name

    if library.media_type == "podcast":
        total_eps = sum(i.num_episodes for i in items)
        podcast_overview = (
            f'The podcast library "{lname}" contains {_plural(len(items), "podcast")} '
            f"with {_plural(total_eps, 'downloaded episode')}."
        )
        genres: Counter[str] = Counter(g for i in items for g in i.genres)
        sections.append(("overview", f"Library overview: {lname}", [podcast_overview]))
        sections.append(
            (
                "catalog",
                f'Podcasts in the "{lname}" library:',
                [
                    f"- {i.title}"
                    + (f" (hosted by {', '.join(i.authors)})" if i.authors else "")
                    + f": {_plural(i.num_episodes, 'episode')}"
                    for i in sorted(items, key=lambda x: x.title.casefold())
                ],
            )
        )
        if genres:
            sections.append(
                (
                    "genres",
                    f'Podcast genres in the "{lname}" library:',
                    _count_lines(genres, "podcast"),
                )
            )
    else:
        authors: Counter[str] = Counter(a for i in items for a in i.authors)
        narrators: Counter[str] = Counter(n for i in items for n in i.narrators)
        series: Counter[str] = Counter(s.name for i in items for s in i.series)
        genres = Counter(g for i in items for g in i.genres)
        audio = [i for i in items if i.duration]
        ebooks_only = [i for i in items if not i.duration and i.ebook_format]
        total_hours = sum(i.duration or 0 for i in items) / 3600

        overview = [
            f'The library "{lname}" contains {_plural(len(items), "book")}: '
            f"{_plural(len(audio), 'audiobook')} and "
            f"{_plural(len(ebooks_only), 'ebook')} "
            f"without audio.",
            f"Total listening time is about {total_hours:,.0f} hours.",
            f"There are {_plural(len(authors), 'author')}, "
            f"{_plural(len(narrators), 'narrator')}, "
            f"{_plural(len(series), 'series')} and {_plural(len(genres), 'genre')}.",
        ]
        sections.append(("overview", f"Library overview: {lname}", overview))
        if authors:
            sections.append(
                (
                    "authors",
                    f'Authors in the "{lname}" library:',
                    _count_lines(authors, "book"),
                )
            )
        if narrators:
            sections.append(
                (
                    "narrators",
                    f'Narrators in the "{lname}" library:',
                    _count_lines(narrators, "book"),
                )
            )
        if series:
            sections.append(
                (
                    "series",
                    f'Series in the "{lname}" library:',
                    _count_lines(series, "book"),
                )
            )
        if genres:
            sections.append(
                (
                    "genres",
                    f'Genres in the "{lname}" library:',
                    _count_lines(genres, "book"),
                )
            )

        def catalog_line(i: LibraryItem) -> str:
            line = f"- {_book_label(i)}"
            if i.authors:
                line += f" by {', '.join(i.authors)}"
            length = format_duration(i.duration)
            if length:
                line += f" [{length}]"
            return line

        sections.append(
            (
                "catalog",
                f'Complete catalog of the "{lname}" library:',
                [
                    catalog_line(i)
                    for i in sorted(items, key=lambda x: x.title.casefold())
                ],
            )
        )

    base = _base_payload(library)
    base["url"] = library_url(cfg.public_url, library.id)
    base["title"] = f"{lname} library summary"

    docs: list[Document] = []
    for section, header, lines in sections:
        chunks = split_lines(lines, cfg.chunk_size, header)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    key=f"library:{library.id}:summary:{section}:{idx}",
                    chunk_type="library_summary",
                    text=chunk,
                    context=chunk,
                    payload={
                        **base,
                        "summary_section": section,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    },
                )
            )
    return docs
