import re
from datetime import UTC, datetime

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# HTML / formatting
# ---------------------------------------------------------------------------


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    if "<" not in html and "&" not in html:
        return " ".join(html.split())
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def format_duration(seconds: float | None) -> str | None:
    """Human-friendly duration, e.g. '12 hr 5 min'."""
    if seconds is None or seconds <= 0:
        return None
    total_minutes = round(seconds / 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} hr {minutes} min"
    if hours:
        return f"{hours} hr"
    if minutes:
        return f"{minutes} min"
    return f"{int(seconds)} sec"


def format_timestamp(seconds: float) -> str:
    """H:MM:SS timestamp for chapter markers."""
    total = int(max(0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()
    except OverflowError, OSError, ValueError:
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _joined_length(items: list[str]) -> int:
    if not items:
        return 0
    return sum(len(s) for s in items) + len(items) - 1


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    result: list[str] = []
    total = 0
    for s in reversed(sentences):
        added = len(s) + (1 if result else 0)
        if total + added > overlap:
            break
        result.insert(0, s)
        total += added
    return result


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentence-aware chunking for prose (descriptions, ebook text)."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _split_sentences(normalized):
        sent_len = len(sentence)
        if sent_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
            step = chunk_size - overlap
            for pos in range(0, sent_len, step):
                piece = sentence[pos : pos + chunk_size]
                if piece:
                    chunks.append(piece)
                if pos + chunk_size >= sent_len:
                    break
            current = []
            current_len = 0
            continue

        new_len = current_len + (1 if current else 0) + sent_len
        if new_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            current_len = _joined_length(current)

        current.append(sentence)
        current_len = _joined_length(current)

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_lines(lines: list[str], chunk_size: int, header: str = "") -> list[str]:
    """Pack whole lines into chunks (for listings like chapters or catalogs).

    ``header`` is repeated at the top of every chunk so each chunk stands alone.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = len(header)
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if len(line) > chunk_size:
            line = line[: chunk_size - 1] + "…"
        added = len(line) + 1
        if current and current_len + added > chunk_size:
            chunks.append("\n".join(([header] if header else []) + current))
            current = []
            current_len = len(header)
        current.append(line)
        current_len += added
    if current:
        chunks.append("\n".join(([header] if header else []) + current))
    return chunks
