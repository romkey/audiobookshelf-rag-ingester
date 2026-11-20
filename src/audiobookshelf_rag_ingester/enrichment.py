"""Fills in missing descriptions and subjects from Open Library, then Google Books.

Used only when Calibre lacks a description or tags. Results are stored apart
from library data, with their origin, and never overwrite it.

APIs:
- Open Library: /isbn/{isbn}.json (redirects to the edition), the edition's
  work at /works/{id}.json, and /search.json. https://openlibrary.org/developers/api
- Google Books: /books/v1/volumes?q=isbn:... or intitle:...+inauthor:...
  https://developers.google.com/books/docs/v1/using
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from audiobookshelf_rag_ingester import __version__
from audiobookshelf_rag_ingester.works import author_surname, normalize_isbn, similarity

logger = logging.getLogger(__name__)

OPEN_LIBRARY = "https://openlibrary.org"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1"
# A title-and-author match is accepted only when it is this close.
TITLE_MATCH_THRESHOLD = 0.92
MAX_SUBJECTS = 25


@dataclass(frozen=True)
class Enrichment:
    origin: str
    lookup_key: str
    matched: bool
    confidence: float | None = None
    # Any: description (str), subjects (list[str]) and the matched title/authors.
    fields: dict[str, Any] = field(default_factory=dict)


def _description(value: object) -> str | None:
    # Open Library descriptions are a string or {"type": ..., "value": "..."}.
    if isinstance(value, dict):
        value = value.get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _subjects(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if isinstance(v, str)][:MAX_SUBJECTS]


def _title_confidence(
    want_title: str, want_author: str | None, got_title: str, got_authors: list[str]
) -> float:
    surname = author_surname(want_author)
    if surname and surname not in {author_surname(a) for a in got_authors}:
        return 0.0
    return similarity(want_title, got_title)


class Enricher:
    def __init__(
        self,
        *,
        timeout_seconds: int,
        google_api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._google_key = google_api_key
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": f"audiobookshelf-rag-ingester/{__version__}"},
        )

    def close(self) -> None:
        self._client.close()

    # Any: decoded JSON.
    def _get(self, url: str, params: dict[str, str] | None = None) -> Any:
        try:
            response = self._client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Enrichment lookup failed for %s: %s", url, e)
            return None

    def lookup(self, *, isbn13: str | None, title: str, authors: list[str]) -> list[Enrichment]:
        """Open Library first; Google Books only if Open Library found no description."""
        results: list[Enrichment] = []
        first_author = authors[0] if authors else None
        open_library = (
            self.open_library_isbn(isbn13)
            if isbn13
            else self.open_library_search(title, first_author)
        )
        results.append(open_library)
        if open_library.matched and open_library.fields.get("description"):
            return results
        results.append(
            self.google_isbn(isbn13) if isbn13 else self.google_search(title, first_author)
        )
        return results

    def open_library_isbn(self, isbn13: str) -> Enrichment:
        key = f"isbn:{isbn13}"
        edition = self._get(f"{OPEN_LIBRARY}/isbn/{isbn13}.json")
        if not isinstance(edition, dict):
            return Enrichment("openlibrary", key, matched=False)
        fields: dict[str, Any] = {
            "title": edition.get("title"),
            "subjects": _subjects(edition.get("subjects")),
            "description": _description(edition.get("description")),
        }
        works = edition.get("works")
        if isinstance(works, list) and works and isinstance(works[0], dict):
            work_key = str(works[0].get("key") or "")
            work = self._get(f"{OPEN_LIBRARY}{work_key}.json") if work_key else None
            if isinstance(work, dict):
                fields["description"] = fields["description"] or _description(
                    work.get("description")
                )
                fields["subjects"] = fields["subjects"] or _subjects(work.get("subjects"))
                fields["work_key"] = work_key
        return Enrichment("openlibrary", key, matched=True, confidence=1.0, fields=fields)

    def open_library_search(self, title: str, author: str | None) -> Enrichment:
        key = f"title:{title}|author:{author or ''}"
        params = {"title": title, "limit": "5", "fields": "key,title,author_name,subject,isbn"}
        if author:
            params["author"] = author
        data = self._get(f"{OPEN_LIBRARY}/search.json", params)
        docs = data.get("docs") if isinstance(data, dict) else None
        best: tuple[float, dict[str, Any]] | None = None
        for doc in docs if isinstance(docs, list) else []:
            if not isinstance(doc, dict):
                continue
            score = _title_confidence(
                title, author, str(doc.get("title") or ""), _subjects(doc.get("author_name"))
            )
            if best is None or score > best[0]:
                best = (score, doc)
        if best is None or best[0] < TITLE_MATCH_THRESHOLD:
            return Enrichment(
                "openlibrary", key, matched=False, confidence=best[0] if best else None
            )
        score, doc = best
        fields: dict[str, Any] = {
            "title": doc.get("title"),
            "authors": _subjects(doc.get("author_name")),
            "subjects": _subjects(doc.get("subject")),
            "description": None,
        }
        work_key = str(doc.get("key") or "")
        if work_key:
            work = self._get(f"{OPEN_LIBRARY}{work_key}.json")
            if isinstance(work, dict):
                fields["description"] = _description(work.get("description"))
            fields["work_key"] = work_key
        return Enrichment("openlibrary", key, matched=True, confidence=score, fields=fields)

    def _google(self, query: str) -> list[dict[str, Any]]:
        params = {"q": query, "maxResults": "5"}
        if self._google_key:
            params["key"] = self._google_key
        data = self._get(f"{GOOGLE_BOOKS}/volumes", params)
        items = data.get("items") if isinstance(data, dict) else None
        volumes: list[dict[str, Any]] = []
        for item in items if isinstance(items, list) else []:
            info = item.get("volumeInfo") if isinstance(item, dict) else None
            if isinstance(info, dict):
                volumes.append(info)
        return volumes

    @staticmethod
    def _google_fields(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": info.get("title"),
            "authors": _subjects(info.get("authors")),
            "subjects": _subjects(info.get("categories")),
            "description": _description(info.get("description")),
        }

    def google_isbn(self, isbn13: str) -> Enrichment:
        key = f"isbn:{isbn13}"
        for info in self._google(f"isbn:{isbn13}"):
            identifiers = info.get("industryIdentifiers")
            isbns = {
                normalize_isbn(str(i.get("identifier")))
                for i in (identifiers if isinstance(identifiers, list) else [])
                if isinstance(i, dict)
            }
            if not isbns or isbn13 in isbns:
                return Enrichment(
                    "googlebooks",
                    key,
                    matched=True,
                    confidence=1.0,
                    fields=self._google_fields(info),
                )
        return Enrichment("googlebooks", key, matched=False)

    def google_search(self, title: str, author: str | None) -> Enrichment:
        key = f"title:{title}|author:{author or ''}"
        query = f"intitle:{title}" + (f" inauthor:{author_surname(author)}" if author else "")
        best: tuple[float, dict[str, Any]] | None = None
        for info in self._google(query):
            score = _title_confidence(
                title, author, str(info.get("title") or ""), _subjects(info.get("authors"))
            )
            if best is None or score > best[0]:
                best = (score, info)
        if best is None or best[0] < TITLE_MATCH_THRESHOLD:
            return Enrichment(
                "googlebooks", key, matched=False, confidence=best[0] if best else None
            )
        return Enrichment(
            "googlebooks",
            key,
            matched=True,
            confidence=best[0],
            fields=self._google_fields(best[1]),
        )
