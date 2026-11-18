"""Work identity: one ``work_id`` for a book in any format (pure, no I/O).

Plan 02/03 correlate ebook, audiobook and hardcopy by ``work_id``: the ISBN
when there is one, else a normalized title plus first author. Audiobook titles
often carry "(Unabridged)" or the series; those are stripped before matching.
"""

import re
import unicodedata

from audiobookshelf_rag_ingester.models import LibraryItem

_NOISE = re.compile(
    r"[\(\[]\s*(?:un)?abridged(?:\s+edition)?\s*[\)\]]"
    r"|[\(\[]\s*(?:a\s+)?(?:novel|audiobook|audio\s+book|audible\s+audio\s+edition)\s*[\)\]]"
    r"|:\s*(?:a\s+)?novel\s*$",
    re.IGNORECASE,
)
_PAREN_SERIES = re.compile(
    r"[\(\[][^)\]]*(?:#|book|vol\.?|volume)\s*\d+[^)\]]*[\)\]]", re.I
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return text.strip()


def normalize_title(title: str, series: list[str] | None = None) -> str:
    """Lowercase, ASCII-folded title without format noise or series markers."""
    cleaned = _NOISE.sub(" ", title)
    cleaned = _PAREN_SERIES.sub(" ", cleaned)
    for name in series or []:
        if name:
            cleaned = re.sub(
                rf"[\(\[:\-,]?\s*{re.escape(name)}\s*(?:#|book|vol\.?)?\s*\d*\s*[\)\]]?",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )
    # "Title: Subtitle" matches "Title": only the main title is compared.
    # A series prefix ("Series 1: Title") leaves an empty first part: skip it.
    parts = [p for p in cleaned.split(":") if _fold(p)]
    main = parts[0] if parts else cleaned
    return _LEADING_ARTICLE.sub("", _fold(main)).strip()


def normalize_isbn(isbn: str | None) -> str | None:
    if not isbn:
        return None
    digits = re.sub(r"[^0-9Xx]", "", isbn).upper()
    return digits if len(digits) in (10, 13) else None


def work_id(
    title: str, authors: list[str], isbn: str | None, series: list[str] | None = None
) -> str:
    clean_isbn = normalize_isbn(isbn)
    if clean_isbn:
        return f"isbn:{clean_isbn}"
    author = _fold(authors[0]) if authors else "unknown"
    return f"title:{normalize_title(title, series)}|{author}"


def item_work_id(item: LibraryItem) -> str | None:
    """The work an audiobook belongs to. Podcasts are not works."""
    if item.media_type != "book":
        return None
    return work_id(item.title, item.authors, item.isbn, [s.name for s in item.series])


def series_sequence(item: LibraryItem) -> tuple[str, str | None] | None:
    """(series name, sequence) of the first series, the tiebreaker for matching."""
    if not item.series:
        return None
    first = item.series[0]
    return first.name, first.sequence
