from __future__ import annotations

import re


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries (.!?) followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _joined_length(items: list[str]) -> int:
    if not items:
        return 0
    return sum(len(s) for s in items) + len(items) - 1


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Return trailing sentences from the list that fit within the overlap character budget."""
    if overlap <= 0:
        return []
    result: list[str] = []
    total = 0
    for s in reversed(sentences):
        added = len(s) + (1 if result else 0)
        if total + added > overlap:
            break
        result.insert(0, s)
        total += added
    return result


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    sentences = _split_sentences(normalized)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if sent_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
            step = chunk_size - overlap
            for pos in range(0, sent_len, step):
                piece = sentence[pos : pos + chunk_size]
                if piece:
                    chunks.append(piece)
            current = []
            current_len = 0
            continue

        new_len = current_len + (1 if current else 0) + sent_len
        if new_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            current_len = _joined_length(current)

        current.append(sentence)
        current_len = _joined_length(current)

    if current:
        chunks.append(" ".join(current))

    return chunks
