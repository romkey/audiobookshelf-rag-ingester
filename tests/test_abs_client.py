from collections.abc import Callable

import httpx
import pytest

from audiobookshelf_rag_ingester.abs_client import AbsClient, AbsClientError


def _client(
    handler: Callable[[httpx.Request], httpx.Response], page_size: int = 2
) -> AbsClient:
    return AbsClient(
        base_url="http://abs.local/",
        token="secret",  # noqa: S106 - dummy test token
        page_size=page_size,
        max_retries=1,
        transport=httpx.MockTransport(handler),
    )


def test_auth_header_and_libraries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.url.path == "/api/libraries"
        return httpx.Response(200, json={"libraries": [{"id": "l1", "name": "Books"}]})

    assert _client(handler).list_libraries() == [{"id": "l1", "name": "Books"}]


def test_items_pagination() -> None:
    data = [{"id": f"i{n}"} for n in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/libraries/l1/items"
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        assert request.url.params["minified"] == "1"
        chunk = data[page * limit : (page + 1) * limit]
        return httpx.Response(200, json={"results": chunk, "total": len(data)})

    ids = [i["id"] for i in _client(handler).iter_library_items("l1")]
    assert ids == ["i0", "i1", "i2", "i3", "i4"]


def test_get_item_expanded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/items/abc"
        assert request.url.params["expanded"] == "1"
        return httpx.Response(200, json={"id": "abc"})

    assert _client(handler).get_item("abc") == {"id": "abc"}


def test_authors_both_shapes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"authors": [{"id": "a"}]})

    assert _client(handler).list_authors("l1") == [{"id": "a"}]


def test_unauthorized_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(AbsClientError, match="ABS_API_TOKEN"):
        _client(handler).list_libraries()


def test_server_error_raises_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(AbsClientError):
        _client(handler).list_libraries()


def test_download_ebook_respects_size_limit() -> None:
    payload = b"x" * 5000

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/items/i1/ebook"
        return httpx.Response(200, content=payload)

    client = _client(handler)
    assert client.download_ebook("i1", max_bytes=10_000) == payload
    assert client.download_ebook("i1", max_bytes=100) is None


def test_download_ebook_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    assert _client(handler).download_ebook("i1", max_bytes=100) is None


def test_retries_then_succeeds() -> None:
    responses = [httpx.Response(503), httpx.Response(200, json={"libraries": []})]
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    client = AbsClient(
        base_url="http://abs.local",
        token="secret",  # noqa: S106 - dummy test token
        max_retries=3,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    assert client.list_libraries() == []
    assert sleeps == [1]


def test_transport_errors_are_retried() -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = AbsClient(
        base_url="http://abs.local",
        token="secret",  # noqa: S106 - dummy test token
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    with client, pytest.raises(AbsClientError, match="refused"):
        client.list_libraries()
    assert sleeps == [1]
