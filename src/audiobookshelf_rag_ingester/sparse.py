"""Keyword (sparse) vectors for hybrid search.

Each distinct word becomes one dimension: its index is the CRC-32 of the
lowercased word, its value a BM25-style saturated term frequency. The
collections use Qdrant's IDF modifier, so Qdrant supplies the inverse
document frequency at query time. Clients must build query vectors with
this same function (``from audiobookshelf_rag_ingester.sparse import sparse_vector``).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

# Recorded in collection metadata so clients can tell which encoder to use.
# Shared with the other ingesters: the same encoder writes every collection.
SPARSE_ENCODER = "calibre_web2rag.sparse/crc32-bm25tf-v1"

_WORD = re.compile(r"\w+", re.UNICODE)
_K1 = 1.2
# Common English words that carry no keyword signal.
_STOPWORD_TEXT = """
a an and are as at be but by for from has have he her his i if in into is it its
of on or our she that the their them there these they this to was we were what
when which who will with you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in (w.lower() for w in _WORD.findall(text))
        if word not in _STOPWORDS and len(word) > 1
    ]


def sparse_vector(text: str) -> SparseVector:
    counts: Counter[int] = Counter()
    for word in tokenize(text):
        counts[zlib.crc32(word.encode("utf-8"))] += 1
    indices = sorted(counts)
    values = [counts[i] * (_K1 + 1) / (counts[i] + _K1) for i in indices]
    return SparseVector(indices=indices, values=values)
