from collections.abc import Callable
from typing import Any

import httpx

from audiobookshelf_rag_ingester.enrichment import Enricher

Routes = dict[str, Any]  # Any: path -> JSON body, an int status, or a callable


def _enricher(routes: Routes, seen: list[httpx.Request] | None = None) -> Enricher:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = routes.get(request.url.path)
        if callable(body):
            body = body(request)
        if body is None:
            return httpx.Response(404)
        if isinstance(body, int):
            return httpx.Response(body)
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    return Enricher(
        timeout_seconds=5,
        google_api_key="gkey",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_open_library_by_isbn_uses_the_work() -> None:
    enricher = _enricher(
        {
            "/isbn/9780306406157.json": {"title": "T", "works": [{"key": "/works/OL1W"}]},
            "/works/OL1W.json": {"description": "Work text.", "subjects": ["Radio"]},
        }
    )
    (result,) = enricher.lookup(isbn13="9780306406157", title="T", authors=["A"])
    assert result.matched
    assert result.fields["description"] == "Work text."
    assert result.fields["subjects"] == ["Radio"]
    assert result.lookup_key == "isbn:9780306406157"


def test_falls_back_to_google_by_isbn() -> None:
    seen: list[httpx.Request] = []
    google: Callable[[httpx.Request], Any] = lambda request: {  # noqa: E731
        "items": [
            {
                "volumeInfo": {
                    "title": "Other",
                    "industryIdentifiers": [{"identifier": "0306406152"}],
                }
            },
        ]
    }
    enricher = _enricher(
        {
            "/isbn/9780306406157.json": {"title": "T", "subjects": ["X"]},
            "/books/v1/volumes": google,
        },
        seen,
    )
    results = enricher.lookup(isbn13="9780306406157", title="T", authors=[])
    assert [r.origin for r in results] == ["openlibrary", "googlebooks"]
    assert results[1].matched
    assert results[1].fields["title"] == "Other"
    google_request = seen[-1]
    assert google_request.url.params["q"] == "isbn:9780306406157"
    assert google_request.url.params["key"] == "gkey"


def test_google_isbn_mismatch_and_errors() -> None:
    enricher = _enricher(
        {
            "/books/v1/volumes": {
                "items": [
                    {"volumeInfo": {"industryIdentifiers": [{"identifier": "9780316129084"}]}},
                    "junk",
                ]
            }
        }
    )
    assert not enricher.google_isbn("9780306406157").matched
    assert (
        not _enricher({"/isbn/9780306406157.json": 500}).open_library_isbn("9780306406157").matched
    )
    assert not _enricher({"/books/v1/volumes": "not json"}).google_isbn("9780306406157").matched


def test_title_search_accepts_only_close_matches() -> None:
    enricher = _enricher(
        {
            "/search.json": {
                "docs": [
                    {
                        "title": "Antenna Theory",
                        "author_name": ["Constantine Balanis"],
                        "key": "/works/OL9W",
                        "subject": ["Antennas"],
                    },
                    {"title": "Antenna Theory", "author_name": ["Someone Else"]},
                    "junk",
                ]
            },
            "/works/OL9W.json": {"description": {"type": "/type/text", "value": "Classic."}},
            "/books/v1/volumes": {
                "items": [{"volumeInfo": {"title": "Unrelated Cookbook", "authors": ["Balanis"]}}]
            },
        }
    )
    (result,) = enricher.lookup(isbn13=None, title="Antenna Theory", authors=["C. Balanis"])
    assert result.matched
    assert result.confidence == 1.0
    assert result.fields["description"] == "Classic."

    miss = enricher.google_search("Antenna Theory", "C. Balanis")
    assert not miss.matched
    assert miss.confidence is not None
    assert miss.confidence < 0.92
    nothing = _enricher({}).lookup(isbn13=None, title="Ghost", authors=[])
    assert [r.matched for r in nothing] == [False, False]


def test_google_title_search_match() -> None:
    enricher = _enricher(
        {
            "/books/v1/volumes": {
                "items": [
                    {
                        "volumeInfo": {
                            "title": "Ghost Book",
                            "authors": ["Gail Ghost"],
                            "description": "Boo.",
                            "categories": ["Horror"],
                        }
                    }
                ]
            }
        }
    )
    result = enricher.google_search("Ghost Book", None)
    assert result.matched
    assert result.fields["subjects"] == ["Horror"]
