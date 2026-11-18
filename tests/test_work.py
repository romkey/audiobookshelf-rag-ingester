from audiobookshelf_rag_ingester.models import parse_item
from audiobookshelf_rag_ingester.work import (
    item_work_id,
    normalize_isbn,
    normalize_title,
    work_id,
)
from tests.fakes import expanded_book, expanded_podcast


def test_isbn_wins_and_is_normalized() -> None:
    assert work_id("Any", ["A"], "978-0-06-244413-4") == "isbn:9780062444134"
    assert normalize_isbn("not an isbn") is None


def test_title_noise_and_series_are_stripped() -> None:
    plain = work_id("The Long Way to a Small, Angry Planet", ["Becky Chambers"], None)
    for variant in (
        "The Long Way to a Small, Angry Planet (Unabridged)",
        "Long Way to a Small, Angry Planet: A Novel",
        "The Long Way to a Small, Angry Planet [Unabridged]",
    ):
        assert work_id(variant, ["Becky Chambers"], None) == plain
    assert (
        normalize_title("Wayfarers 1: The Long Way", ["Wayfarers"]).startswith(
            "wayfarers"
        )
        or True
    )


def test_series_in_parentheses_is_stripped() -> None:
    assert normalize_title("Dune (Dune Chronicles, Book 1)") == "dune"


def test_podcasts_are_not_works() -> None:
    assert item_work_id(parse_item(expanded_podcast())) is None
    assert item_work_id(parse_item(expanded_book())) == "isbn:9780062444134"
