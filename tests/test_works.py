import pytest

from audiobookshelf_rag_ingester.works import (
    author_surname,
    book_isbn13,
    clean_audiobook_title,
    normalize_isbn,
    normalize_title,
    series_id_for,
    similarity,
    work_id_for,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0-306-40615-2", "9780306406157"),
        ("978-0-306-40615-7", "9780306406157"),
        ("080442957X", "9780804429573"),
        ("0306406153", None),  # bad ISBN-10 check digit
        ("9780306406158", None),  # bad ISBN-13 check digit
        ("1234567890123", None),  # not a 978/979 prefix
        ("12345", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_isbn(value: str | None, expected: str | None) -> None:
    assert normalize_isbn(value) == expected


def test_book_isbn13_tries_identifier_keys() -> None:
    assert book_isbn13({"isbn": "junk", "isbn13": "9780306406157"}) == "9780306406157"
    assert book_isbn13({"asin": "B00"}) is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("The Hobbit (Unabridged)", "hobbit"),
        ("Python Crash Course, 2nd Edition", "python crash course"),
        ("Clean Code: A Handbook of Agile Software Craftsmanship", "clean code"),
        ("An Élan Vital — Revised Edition", "elan vital"),
        ("Second Edition of Nothing", "of nothing"),
    ],
)
def test_normalize_title(title: str, expected: str) -> None:
    assert normalize_title(title) == expected


def test_author_surname() -> None:
    assert author_surname("Ursula K. Le Guin") == "guin"
    assert author_surname("Tolkien, J. R. R.") == "tolkien"
    assert author_surname("") == ""
    assert author_surname(None) == ""
    assert author_surname("   ") == ""


def test_work_id_for() -> None:
    assert work_id_for(title="X", authors=[], isbn13="9780306406157") == "isbn:9780306406157"
    assert work_id_for(title="The Hobbit, 2nd edition", authors=["J. Tolkien"], isbn13=None) == (
        "title:hobbit--tolkien"
    )
    assert work_id_for(title="(Untitled)", authors=[], isbn13=None) == "title:untitled--unknown"


def test_series_and_similarity() -> None:
    assert series_id_for("The Expanse") == "series:the-expanse"
    assert series_id_for("!!!") == "series:unnamed"
    assert similarity("The Hobbit", "Hobbit (Unabridged)") == 1.0
    assert similarity("", "x") == 0.0
    assert similarity("Dune", "Cooking") < 0.5


@pytest.mark.parametrize(
    ("title", "series", "expected"),
    [
        ("Wayfarers 1: The Long Way", ["Wayfarers"], "The Long Way"),
        ("The Expanse #1 - Leviathan Wakes", ["The Expanse"], "Leviathan Wakes"),
        ("Leviathan Wakes: The Expanse, Book 1", ["The Expanse"], "Leviathan Wakes"),
        ("Leviathan Wakes (Expanse 1)", ["Expanse"], "Leviathan Wakes"),
        ("Book 3 - The Dragon Reborn", [], "The Dragon Reborn"),
        ("Discworld", ["Discworld"], "Discworld"),
        ("Dune", [], "Dune"),
    ],
)
def test_clean_audiobook_title(title: str, series: list[str], expected: str) -> None:
    assert clean_audiobook_title(title, series) == expected


def test_audiobook_and_ebook_get_the_same_work_id() -> None:
    audio = clean_audiobook_title("Wayfarers 1: The Long Way (Unabridged)", ["Wayfarers"])
    assert work_id_for(title=audio, authors=["Becky Chambers"], isbn13=None) == work_id_for(
        title="The Long Way", authors=["Chambers, Becky"], isbn13=None
    )
