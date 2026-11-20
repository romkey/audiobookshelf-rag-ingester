"""Ingest-time LLM tasks through the OpenAI-compatible gateway.

Every result is cached in Postgres by task, model and a hash of the input,
so re-runs and re-indexes don't repeat the calls. With no ``LLM_MODEL`` set
the tasks are skipped and return None.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Cache(Protocol):
    # Any: cached outputs are JSON.
    def llm_cache_get(self, task: str, model: str, input_hash: str) -> Any: ...
    def llm_cache_put(self, task: str, model: str, input_hash: str, output: Any) -> None: ...


def input_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class BookAnalysis:
    topics: list[str]
    description_is_marketing: bool
    good_for: str


class LLMError(RuntimeError):
    """The gateway failed or returned something unusable."""


class LLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: int,
        cache: Cache,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._cache = cache
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def chat(self, system: str, user: str) -> str:
        """One chat completion (``POST {base_url}/chat/completions``)."""
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"chat completion failed: {e}") from e
        if not isinstance(content, str):
            raise LLMError("chat completion returned no text")
        return content.strip()

    # Any: the parsed JSON object the model returned.
    def _json(self, system: str, user: str) -> dict[str, Any]:
        text = self.chat(system, user)
        match = _JSON_OBJECT.search(text)
        try:
            value = json.loads(match.group() if match else text)
        except json.JSONDecodeError as e:
            raise LLMError(f"response is not JSON: {text[:200]!r}") from e
        if not isinstance(value, dict):
            raise LLMError("response JSON is not an object")
        return value

    # Any: whatever JSON the task stores.
    def _cached(self, task: str, key: str) -> Any:
        return self._cache.llm_cache_get(task, self.model, key)

    def analyze_book(
        self,
        *,
        title: str,
        authors: list[str],
        description: str,
        tags: list[str],
        toc: str,
        taxonomy: list[str],
    ) -> BookAnalysis:
        """Topics from the taxonomy, a marketing-copy check and a "good for" summary."""
        prompt = (
            f"Title: {title}\nAuthors: {', '.join(authors) or 'unknown'}\n"
            f"Tags: {', '.join(tags) or 'none'}\n\nDescription:\n{description or '(none)'}\n\n"
            f"Table of contents:\n{toc[:6000] or '(none)'}"
        )
        key = input_hash(prompt, "\n".join(taxonomy))
        cached = self._cached("book_analysis", key)
        if cached is None:
            system = (
                "You catalog books for a hackerspace library. Reply with one JSON object: "
                '{"topics": [...], "description_is_marketing": true|false, "good_for": "..."}. '
                "topics: 1 to 5 slugs, chosen only from this list: "
                + ", ".join(taxonomy)
                + ". description_is_marketing: true if the description is mostly publisher "
                "praise or blurbs rather than what the book covers, or if there is none. "
                "good_for: one paragraph, plain and factual, saying what the book covers and "
                "who would find it useful. Use only the information given."
            )
            cached = self._json(system, prompt)
            self._cache.llm_cache_put("book_analysis", self.model, key, cached)
        allowed = set(taxonomy)
        raw_topics = cached.get("topics") or []
        topics = sorted({t.strip().lower() for t in raw_topics if isinstance(t, str)} & allowed)
        return BookAnalysis(
            topics=topics,
            description_is_marketing=bool(cached.get("description_is_marketing")),
            good_for=str(cached.get("good_for") or "").strip(),
        )

    def podcast_topics(
        self,
        *,
        title: str,
        hosts: list[str],
        description: str,
        genres: list[str],
        episode_titles: list[str],
        taxonomy: list[str],
    ) -> list[str]:
        """1 to 5 taxonomy slugs for a podcast."""
        prompt = (
            f"Podcast: {title}\nHosts: {', '.join(hosts) or 'unknown'}\n"
            f"Genres: {', '.join(genres) or 'none'}\n\nDescription:\n{description or '(none)'}"
            "\n\nRecent episodes:\n" + ("\n".join(episode_titles[:40]) or "(none)")
        )
        key = input_hash(prompt, "\n".join(taxonomy))
        cached = self._cached("podcast_topics", key)
        if cached is None:
            system = (
                "You catalog podcasts for a hackerspace library. Reply with one JSON object: "
                '{"topics": [...]}. topics: 1 to 5 slugs, chosen only from this list: '
                + ", ".join(taxonomy)
                + ". Use only the information given."
            )
            cached = self._json(system, prompt)
            self._cache.llm_cache_put("podcast_topics", self.model, key, cached)
        allowed = set(taxonomy)
        raw_topics = cached.get("topics") or []
        return sorted({t.strip().lower() for t in raw_topics if isinstance(t, str)} & allowed)

    def chunk_context(self, *, document_title: str, document: str, chunk: str) -> str:
        """One or two sentences placing a chunk within its document."""
        key = input_hash(document_title, document[:8000], chunk)
        cached = self._cached("chunk_context", key)
        if cached is None:
            text = self.chat(
                "Write one or two sentences that place the excerpt within the document, "
                "for search indexing. Reply with the sentences only.",
                f"Document: {document_title}\n\n{document[:8000]}\n\nExcerpt:\n{chunk}",
            )
            cached = {"context": text}
            self._cache.llm_cache_put("chunk_context", self.model, key, cached)
        return str(cached.get("context") or "")

    def confirm_link(self, *, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        """Does ``alias`` in ``text`` refer to this entity? For ambiguous linker hits."""
        key = input_hash(text[:4000], alias, entity_type, entity_id)
        cached = self._cached("confirm_link", key)
        if cached is None:
            cached = self._json(
                'Reply with one JSON object: {"match": true|false}.',
                f"Does the phrase {alias!r} in this text refer to the {entity_type} "
                f"{entity_id!r}?\n\n{text[:4000]}",
            )
            self._cache.llm_cache_put("confirm_link", self.model, key, cached)
        return bool(cached.get("match"))
