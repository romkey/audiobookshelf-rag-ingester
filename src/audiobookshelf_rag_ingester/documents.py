"""Pure functions that turn Audiobookshelf items and registry rows into text and payloads.

``point_id``, ``base_payload`` and ``catalog_text`` are shared with the
pipeline's other ingesters (calibre-web2rag writes the same ``hs_catalog``
points from the same registry rows); keep them identical there.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup

from audiobookshelf_rag_ingester.chunking import split_text

TIER = "member"
# Shared architecture: 5 policy/sign, 4 wiki, 3 asset/geowiki/event,
# 2 manual/repo/esphome, 1 slack fact. Library records sit with manuals.
AUTHORITY = 2
PODCAST_SOURCE = "podcast"
_BLANK_LINES = re.compile(r"\n\s*\n+")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLOCKS = ["p", "div", "li", "blockquote", "tr", "h1", "h2", "h3", "h4", "h5", "h6"]
_GENERIC_CHAPTER = re.compile(r"^\s*(chapter|track|part|ch\.?)?\s*[\divxlc]+\s*$", re.IGNORECASE)

# ISO 639-2/B codes, as Calibre stores languages, for the names and ISO 639-1
# codes Audiobookshelf metadata usually carries.
_LANGUAGE_CODES = {
    "english": "eng",
    "en": "eng",
    "spanish": "spa",
    "español": "spa",
    "es": "spa",
    "french": "fre",
    "français": "fre",
    "fr": "fre",
    "german": "ger",
    "deutsch": "ger",
    "de": "ger",
    "italian": "ita",
    "it": "ita",
    "portuguese": "por",
    "pt": "por",
    "dutch": "dut",
    "nl": "dut",
    "japanese": "jpn",
    "ja": "jpn",
    "chinese": "chi",
    "zh": "chi",
    "russian": "rus",
    "ru": "rus",
    "polish": "pol",
    "pl": "pol",
    "swedish": "swe",
    "sv": "swe",
}

# (text) -> {"asset_ids": [...], ...}: the linker pass.
LinkFn = Callable[[str], dict[str, list[str]]]


def point_id(source: str, doc_id: str, chunk_index: int) -> str:
    """uuid5(source:doc_id:chunk_index): stable, so a re-sync overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_id}:{chunk_index}"))


def text_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# Any: JSON-ready values.
def json_hash(value: Any) -> str:
    return text_hash(json.dumps(value, sort_keys=True, default=str))


# Payload fields that change on every run without the point changing.
_VOLATILE = frozenset({"indexed_at", "sync_run_id", "payload_hash"})


# Any: payload values are JSON.
def payload_hash(payload: dict[str, Any]) -> str:
    """Hash of everything in a payload but the per-run fields."""
    return json_hash({k: v for k, v in payload.items() if k not in _VOLATILE})


def html_to_text(html: str | None) -> str:
    """An HTML description or show notes as plain text with paragraph breaks."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(_BLOCKS):
        block.insert_after("\n\n")
    text = soup.get_text("")
    paragraphs = [_SPACES.sub(" ", p).strip() for p in _BLANK_LINES.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def ms_to_rfc3339(value: int | None) -> str | None:
    """Audiobookshelf's epoch milliseconds as an RFC 3339 UTC timestamp."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()
    except OverflowError, OSError, ValueError:
        return None


def now_rfc3339() -> str:
    return datetime.now(UTC).isoformat()


def year_of(value: str | None) -> int | None:
    match = re.search(r"\b(\d{4})\b", value or "")
    return int(match.group(1)) if match else None


def language_code(value: str | None) -> str | None:
    """A language as Calibre stores it ("eng"), so both sources filter alike."""
    if not value:
        return None
    text = value.strip().lower()
    return _LANGUAGE_CODES.get(text, text)


def format_duration(seconds: float | None) -> str:
    """'15 hr 30 min' or '42 min'; '' for no length."""
    if not seconds or seconds <= 0:
        return ""
    hours, minutes = divmod(round(seconds / 60), 60)
    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    return f"{max(minutes, 1)} min"


def meaningful_chapters(titles: list[str]) -> list[str]:
    """Chapter titles, or none when they are just "Chapter 1, 2, 3"."""
    if len(titles) < 2 or all(_GENERIC_CHAPTER.match(t) for t in titles):
        return []
    return titles


def _join(label: str, values: Iterable[str]) -> str | None:
    items = [v for v in values if v]
    return f"{label}: {', '.join(items)}" if items else None


# Any: registry rows and manifestation details are JSON.
def catalog_text(
    work: dict[str, Any], manifestations: list[dict[str, Any]], series: str | None
) -> str:
    """What a work's catalog point embeds: title, authors, series, formats, narrators,
    length, topics, description, contents and audiobook chapters."""
    details = [m.get("details") or {} for m in manifestations]
    # Prefer the library's own description; use the machine summary when the
    # description is missing or is marketing copy.
    description = ""
    for d in details:
        if d.get("description") and not d.get("description_is_marketing"):
            description = str(d["description"])
            break
    good_for = next((str(d["good_for"]) for d in details if d.get("good_for")), "")
    if not description:
        description = next(
            (str(d["enriched_description"]) for d in details if d.get("enriched_description")),
            "",
        )
    toc: list[str] = next((list(d["toc"]) for d in details if d.get("toc")), [])
    chapters: list[str] = next((list(d["chapters"]) for d in details if d.get("chapters")), [])
    subtitle = next((str(d["subtitle"]) for d in details if d.get("subtitle")), "")
    tags = sorted({t for d in details for t in d.get("tags") or []})
    subjects = sorted({s for d in details for s in d.get("enriched_subjects") or []})
    narrators = sorted({n for d in details for n in d.get("narrators") or []})
    length = next((d["duration_seconds"] for d in details if d.get("duration_seconds")), None)
    formats = sorted({str(m["format"]) for m in manifestations})

    title = str(work["title"]) + (
        f": {subtitle}" if subtitle and subtitle not in work["title"] else ""
    )
    lines = [title]
    if work.get("authors"):
        lines.append(f"by {', '.join(work['authors'])}")
    if series:
        index = work.get("series_index")
        number = f" #{index:g}" if isinstance(index, int | float) else ""
        lines.append(f"Series: {series}{number}")
    for line in (
        _join("Formats in the library", formats),
        _join("Narrated by", narrators),
        f"Audiobook length: {format_duration(length)}" if length else None,
        _join("Topics", work.get("topics") or []),
        _join("Tags", tags),
        _join("Subjects", subjects),
    ):
        if line:
            lines.append(line)
    if description:
        lines.extend(["", description])
    if good_for and good_for != description:
        lines.extend(["", f"What this book is good for (machine-generated summary): {good_for}"])
    if toc:
        lines.extend(["", "Contents:", *(str(t) for t in toc)])
    if chapters:
        lines.extend(["", "Audiobook chapters:", *(str(c) for c in chapters)])
    return "\n".join(lines)


# Any: payload values are JSON.
def base_payload(
    *,
    source: str,
    doc_id: str,
    chunk_index: int,
    title: str,
    text: str,
    context: str,
    url: str | None,
    run_id: str,
    created_at: str | None,
    updated_at: str | None,
    heading_path: list[str] | None = None,
    page: int | None = None,
    lang: str | None = None,
    topics: list[str] | None = None,
    work_ids: list[str] | None = None,
    links: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The payload every text collection shares (see the shared architecture plan)."""
    links = links or {}
    return {
        "source": source,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "title": title,
        "heading_path": heading_path or [],
        "text": text,
        "context": context,
        "url": url,
        "tier": TIER,
        "authority": AUTHORITY,
        "created_at": created_at,
        "updated_at": updated_at,
        "indexed_at": now_rfc3339(),
        "stale_after": None,
        "superseded_by": None,
        "content_hash": text_hash(context, text),
        "sync_run_id": run_id,
        "asset_ids": links.get("asset_ids", []),
        "location_ids": links.get("location_ids", []),
        "work_ids": sorted(set(work_ids or []) | set(links.get("work_ids", []))),
        "event_ids": links.get("event_ids", []),
        "training_ids": links.get("training_ids", []),
        "topics": topics or [],
        "section": None,
        "page": page,
        "lang": lang,
    }


# ---------------------------------------------------------------------------
# Library overview (hs_member)
# ---------------------------------------------------------------------------


def _approx(n: int) -> str:
    return f"about {round(n, -1):,}" if n >= 100 else f"{n}"


# Any: catalog rows from Postgres and staged podcast records.
def overview_sections(
    rows: list[dict[str, Any]],
    podcasts: list[dict[str, Any]],
    *,
    library_url: str,
    access_info: str | None,
    apps_info: str | None,
) -> list[tuple[str, str]]:
    """(heading, body) sections of the regenerated audiobook library overview."""
    books = [r for r in rows if r.get("format") == "audiobook"]
    authors = {a for r in books for a in r.get("authors") or []}
    narrators = {n for r in books for n in r.get("narrators") or []}
    hours = sum(float(r.get("duration_seconds") or 0) for r in books) / 3600
    topics: Counter[str] = Counter(t for r in books for t in r.get("topics") or [])
    genres: Counter[str] = Counter(g for r in books for g in r.get("genres") or [])
    series = {r["series"] for r in books if r.get("series")}
    languages: Counter[str] = Counter(lang for r in books for lang in r.get("languages") or [])
    episodes = sum(len(p.get("episodes") or []) for p in podcasts)

    intro = (
        f"PDX Hackerspace members have access to an audiobook library of {_approx(len(books))} "
        f"audiobooks, about {round(hours):,} hours of listening, by {len(authors):,} authors "
        f"and {len(narrators):,} narrators. It runs on Audiobookshelf, a self-hosted audiobook "
        f"and podcast server, at {library_url}."
    )
    if podcasts:
        plural = "s" if len(podcasts) != 1 else ""
        intro += f" It also keeps {len(podcasts)} podcast{plural} with {episodes:,} episodes."
    sections = [("The PDX Hackerspace audiobook library", intro)]

    contents: list[str] = []
    if topics:
        contents.append(
            "Its strongest topics are "
            + ", ".join(f"{t} ({n})" for t, n in topics.most_common(12))
            + "."
        )
    if genres:
        contents.append(
            "Common genres: " + ", ".join(f"{g} ({n})" for g, n in genres.most_common(15)) + "."
        )
    if series:
        contents.append(f"It holds books from {len(series):,} series.")
    if languages:
        contents.append(
            "Languages: " + ", ".join(f"{lang} ({n})" for lang, n in languages.most_common(8)) + "."
        )
    contents.append(
        "For exact counts, lists, lengths and narrators, ask the catalog; it is answered "
        "from the live catalog, not from this summary."
    )
    sections.append(("What is in the audiobook library", " ".join(contents)))
    sections.append(
        (
            "How to get an account on the audiobook library",
            access_info
            or "Accounts on the audiobook library are set up by its maintainers; ask them in "
            "the hackerspace's usual channels.",
        )
    )
    sections.append(
        (
            "Apps that work with the audiobook library",
            apps_info
            or f"Members can listen in a web browser at {library_url}. Audiobookshelf has its "
            "own mobile apps for Android and iOS, and other apps that support Audiobookshelf "
            "servers can sign in with the same account.",
        )
    )
    return sections


# ---------------------------------------------------------------------------
# Podcasts (hs_member, source "podcast")
# ---------------------------------------------------------------------------


def podcast_doc_id(item_id: str) -> str:
    return f"audiobookshelf:podcast:{item_id}"


def episode_doc_id(item_id: str, episode_id: str) -> str:
    return f"audiobookshelf:episode:{item_id}:{episode_id}"


def _episode_date(published_at: int | None, pub_date: str | None) -> str | None:
    stamp = ms_to_rfc3339(published_at)
    if stamp:
        dt = datetime.fromisoformat(stamp)
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    return pub_date


# Any: the staged podcast record (LibraryItem.to_json plus topics) and payloads.
def podcast_points(
    record: dict[str, Any],
    *,
    url: str,
    run_id: str,
    chunk_size: int,
    chunk_overlap: int,
    link: LinkFn,
) -> list[tuple[str, dict[str, Any]]]:
    """(point id, payload) for a podcast: its summary point(s), then each episode."""
    item_id = str(record["id"])
    title = str(record["title"])
    hosts = ", ".join(record.get("authors") or [])
    topics = list(record.get("topics") or [])
    lang = language_code(record.get("language"))
    updated = ms_to_rfc3339(record.get("updated_at"))
    episodes = record.get("episodes") or []
    items: list[tuple[str, dict[str, Any]]] = []

    # Any: payload values are JSON.
    def add(doc_id: str, index: int, payload: dict[str, Any], **fields: Any) -> None:
        payload.update({"item_id": item_id, "podcast_title": title, **fields})
        items.append((point_id(PODCAST_SOURCE, doc_id, index), payload))

    # The podcast: chunk 0 is its summary point, then the rest of a long description.
    doc_id = podcast_doc_id(item_id)
    facts = [
        line
        for line in (
            f"Hosted by {hosts}" if hosts else None,
            _join("Genres", record.get("genres") or []),
            _join("Topics", topics),
            f"Episodes in the library: {len(episodes)}",
            "Explicit content" if record.get("explicit") else None,
        )
        if line
    ]
    header = f"Podcast: {title}\n" + ". ".join(facts) + "."
    pieces = split_text(str(record.get("description") or ""), chunk_size, chunk_overlap) or [""]
    for index, piece in enumerate(pieces):
        head = header if index == 0 else f"Podcast: {title} (description, continued)"
        text = head + (f"\n\n{piece}" if piece else "")
        add(
            doc_id,
            index,
            base_payload(
                source=PODCAST_SOURCE,
                doc_id=doc_id,
                chunk_index=index,
                title=title,
                text=text,
                context=f"The podcast {title!r} in the hackerspace's Audiobookshelf library.",
                url=url,
                run_id=run_id,
                created_at=ms_to_rfc3339(record.get("added_at")),
                updated_at=updated,
                heading_path=[title],
                lang=lang,
                topics=topics,
                links=link(text),
            ),
            kind="podcast",
        )

    for ep in episodes:
        ep_doc = episode_doc_id(item_id, str(ep["id"]))
        published = _episode_date(ep.get("published_at"), ep.get("pub_date"))
        lines = [f"Episode: {ep['title']}", f"Podcast: {title}"]
        if ep.get("subtitle") and ep["subtitle"] != ep["title"]:
            lines.append(str(ep["subtitle"]))
        numbering = [
            f"Season {ep['season']}" if ep.get("season") else "",
            f"Episode {ep['episode']}" if ep.get("episode") else "",
        ]
        if any(numbering):
            lines.append(", ".join(n for n in numbering if n))
        if published:
            lines.append(f"Published: {published}")
        if ep.get("duration"):
            lines.append(f"Length: {format_duration(ep['duration'])}")
        notes = str(ep.get("description") or "")
        ep_pieces = split_text(notes, chunk_size, chunk_overlap) or [""]
        context = (
            f"An episode of the podcast {title!r}"
            + (f", published {published}" if published else "")
            + "."
        )
        for index, piece in enumerate(ep_pieces):
            head = (
                "\n".join(lines)
                if index == 0
                else f"Episode: {ep['title']} (show notes, continued)"
            )
            text = head + (f"\n\n{piece}" if piece else "")
            add(
                ep_doc,
                index,
                base_payload(
                    source=PODCAST_SOURCE,
                    doc_id=ep_doc,
                    chunk_index=index,
                    title=str(ep["title"]),
                    text=text,
                    context=context,
                    url=url,
                    run_id=run_id,
                    created_at=ms_to_rfc3339(ep.get("published_at")),
                    updated_at=updated,
                    heading_path=[title, str(ep["title"])],
                    lang=lang,
                    topics=topics,
                    links=link(text),
                ),
                kind="episode",
                episode_id=str(ep["id"]),
                published_at=ms_to_rfc3339(ep.get("published_at")),
                duration_seconds=ep.get("duration"),
            )
    return items
