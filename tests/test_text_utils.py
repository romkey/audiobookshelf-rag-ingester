import pytest

from audiobookshelf_rag_ingester.text_utils import (
    format_duration,
    format_timestamp,
    ms_to_iso,
    split_lines,
    split_text,
    strip_html,
)


def test_strip_html() -> None:
    assert strip_html("<p>Hello <b>world</b></p><p>Again</p>") == "Hello world Again"
    assert strip_html("plain   text") == "plain text"
    assert strip_html(None) == ""
    assert strip_html("Tom &amp; Jerry") == "Tom & Jerry"
    assert strip_html("line<br>break") == "line break"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, None),
        (0, None),
        (20, "20 sec"),
        (45, "1 min"),
        (3600, "1 hr"),
        (3600 * 12 + 300, "12 hr 5 min"),
    ],
)
def test_format_duration(seconds: float | None, expected: str | None) -> None:
    assert format_duration(seconds) == expected


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "0:00:00"
    assert format_timestamp(3725.4) == "1:02:05"


def test_ms_to_iso() -> None:
    assert ms_to_iso(None) is None
    assert ms_to_iso(0) == "1970-01-01T00:00:00+00:00"
    assert ms_to_iso(10**20) is None


def test_split_text_respects_size_and_overlap() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Overlap: the tail sentence of one chunk starts the next
    assert chunks[1].split(".")[0] in chunks[0]


def test_split_text_long_sentence_is_windowed() -> None:
    chunks = split_text("x" * 1000, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_split_text_validation() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("hi", chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="chunk_size"):
        split_text("hi", chunk_size=0, overlap=0)
    assert split_text("   ", 10, 0) == []


def test_split_lines_repeats_header() -> None:
    lines = [f"- item {i}" for i in range(50)]
    chunks = split_lines(lines, chunk_size=100, header="Catalog:")
    assert len(chunks) > 1
    assert all(c.startswith("Catalog:") for c in chunks)
    joined = "\n".join(chunks)
    assert all(line in joined for line in lines)


def test_split_lines_truncates_and_skips_blank() -> None:
    chunks = split_lines(["", "y" * 50], chunk_size=20)
    assert chunks == ["y" * 19 + "…"]
