import json
from typing import Any

import httpx
import pytest

from audiobookshelf_rag_ingester.llm import LLM, LLMError, input_hash


class DictCache:
    def __init__(self) -> None:
        # Any: cached JSON.
        self.data: dict[tuple[str, str, str], Any] = {}

    # Any: cached JSON.
    def llm_cache_get(self, task: str, model: str, key: str) -> Any:
        return self.data.get((task, model, key))

    # Any: cached JSON.
    def llm_cache_put(self, task: str, model: str, key: str, output: Any) -> None:
        self.data[(task, model, key)] = output


def _llm(reply: str | httpx.Response, cache: DictCache | None = None) -> tuple[LLM, list[Any]]:
    # Any: the request bodies sent.
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer key"
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    llm = LLM(
        model="m",
        base_url="http://gw/v1/",
        api_key="key",
        timeout_seconds=5,
        cache=cache or DictCache(),
        transport=httpx.MockTransport(handler),
    )
    return llm, sent


def test_analyze_book_filters_topics_and_caches() -> None:
    reply = 'Sure! {"topics": ["Antennas", "made-up", 3], "description_is_marketing": true}'
    cache = DictCache()
    llm, sent = _llm(reply, cache)
    args: dict[str, Any] = {
        "title": "T",
        "authors": [],
        "description": "",
        "tags": [],
        "toc": "",
        "taxonomy": ["antennas", "fantasy"],
    }
    first = llm.analyze_book(**args)
    second = llm.analyze_book(**args)
    assert first.topics == ["antennas"]
    assert first.description_is_marketing is True
    assert first.good_for == ""
    assert second == first
    assert len(sent) == 1
    assert sent[0]["model"] == "m"
    assert sent[0]["temperature"] == 0


def test_chunk_context_and_confirm_link() -> None:
    llm, _ = _llm("It is the intro.")
    assert llm.chunk_context(document_title="D", document="doc", chunk="c") == "It is the intro."
    yes, _ = _llm('{"match": true}')
    assert yes.confirm_link(text="t", alias="a", entity_type="asset", entity_id="x") is True


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        "not json at all",
        "[1, 2]",
    ],
)
def test_errors_raise_llm_error(reply: str | httpx.Response) -> None:
    llm, _ = _llm(reply)
    with pytest.raises(LLMError):
        llm.confirm_link(text="t", alias="a", entity_type="asset", entity_id="x")


def test_input_hash_separates_parts() -> None:
    assert input_hash("ab", "c") != input_hash("a", "bc")
