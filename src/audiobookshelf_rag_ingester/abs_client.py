"""Thin client for the Audiobookshelf REST API.

Uses only read-only endpoints. Authenticate with an Audiobookshelf API key
(Settings -> API Keys, ABS >= 2.26) or a user token, sent as a Bearer token.
"""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any  # Any: decoded JSON from the Audiobookshelf API

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class AbsClientError(RuntimeError):
    pass


class AbsClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: int = 60,
        page_size: int = 100,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sleep = sleep
        self._page_size = max(1, page_size)
        self._max_retries = max(1, max_retries)
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AbsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self, path: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if response.status_code in (401, 403):
                    raise AbsClientError(
                        f"Audiobookshelf rejected the API token "
                        f"({response.status_code}) "
                        f"for {path}. Check ABS_API_TOKEN."
                    )
                if response.status_code not in _RETRY_STATUSES:
                    response.raise_for_status()
                    return response
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries:
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "Request %s failed (%s); retry %d/%d in %ss",
                    path,
                    last_exc,
                    attempt,
                    self._max_retries - 1,
                    delay,
                )
                self._sleep(delay)
        raise AbsClientError(f"Request to {path} failed: {last_exc}") from last_exc

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request(path, params).json()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def list_libraries(self) -> list[dict[str, Any]]:
        data = self._get_json("/libraries")
        return list(data.get("libraries") or [])

    def iter_library_items(self, library_id: str) -> Iterator[dict[str, Any]]:
        """Yield every (minified) item in a library, following pagination."""
        page = 0
        seen = 0
        while True:
            data = self._get_json(
                f"/libraries/{library_id}/items",
                params={
                    "limit": self._page_size,
                    "page": page,
                    "minified": 1,
                    "sort": "addedAt",
                },
            )
            results = data.get("results") or []
            total = data.get("total")
            yield from results
            seen += len(results)
            if not results or len(results) < self._page_size:
                break
            if isinstance(total, int) and seen >= total:
                break
            page += 1

    def list_library_items(self, library_id: str) -> list[dict[str, Any]]:
        return list(self.iter_library_items(library_id))

    def get_item(self, item_id: str) -> dict[str, Any]:
        data = self._get_json(f"/items/{item_id}", params={"expanded": 1})
        if not isinstance(data, dict):
            raise AbsClientError(f"Unexpected response for item {item_id}")
        return data

    def list_authors(self, library_id: str) -> list[dict[str, Any]]:
        data = self._get_json(f"/libraries/{library_id}/authors")
        if isinstance(data, dict):
            return list(data.get("authors") or data.get("results") or [])
        return []

    def download_ebook(self, item_id: str, max_bytes: int) -> bytes | None:
        """Download an item's primary ebook file. Returns None if it is too large."""
        with self._client.stream("GET", f"/items/{item_id}/ebook") as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                logger.info(
                    "Skipping ebook for item %s: %s bytes exceeds limit",
                    item_id,
                    declared,
                )
                return None
            buf = bytearray()
            for chunk in response.iter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    logger.info(
                        "Skipping ebook for item %s: exceeds size limit", item_id
                    )
                    return None
            return bytes(buf)
