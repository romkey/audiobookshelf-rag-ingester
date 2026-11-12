"""`audiobookshelf-rag-ingester query`: semantic search over the collection.

audiobookshelf-rag-ingester query "space opera with a sarcastic AI"
audiobookshelf-rag-ingester query --type episode --limit 10 "3D printing"
"""

import textwrap
from dataclasses import dataclass
from typing import Any, TextIO

from audiobookshelf_rag_ingester.embeddings import Embedder
from audiobookshelf_rag_ingester.qdrant_store import QdrantStore


@dataclass(frozen=True)
class Hit:
    score: float
    # Any: point payloads are free-form JSON stored by this app.
    payload: dict[str, Any]


def search(
    store: QdrantStore,
    embedder: Embedder,
    text: str,
    *,
    limit: int = 5,
    chunk_type: str | None = None,
    library_id: str | None = None,
    min_score: float | None = None,
) -> list[Hit]:
    vector = embedder.encode([text])[0]
    return [
        Hit(score, payload)
        for score, payload in store.search(
            vector,
            limit=limit,
            chunk_type=chunk_type,
            library_id=library_id,
            min_score=min_score,
        )
    ]


def format_hits(
    hits: list[Hit], out: TextIO, *, show_text: bool = False, width: int = 100
) -> None:
    if not hits:
        out.write("No results.\n")
        return
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        label = str(p.get("title") or p.get("author_name") or "?")
        if p.get("episode_title"):
            label += f" — {p['episode_title']}"
        if p.get("section_title"):
            label += f" § {p['section_title']}"
        out.write(f"{rank:>2}. {hit.score:.3f}  [{p.get('chunk_type')}] {label}\n")
        if p.get("url"):
            out.write(f"      {p['url']}\n")
        if show_text:
            body = textwrap.fill(
                str(p.get("text") or ""),
                width=width,
                initial_indent="      ",
                subsequent_indent="      ",
            )
            out.write(body + "\n")
        out.write("\n")
