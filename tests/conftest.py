import pytest
from qdrant_client import QdrantClient

from audiobookshelf_rag_ingester.qdrant_store import QdrantStore


@pytest.fixture
def store() -> QdrantStore:
    """An empty in-memory Qdrant collection with 8-dimensional vectors."""
    s = QdrantStore(collection="test", client=QdrantClient(":memory:"))
    s.ensure_collection(vector_size=8, distance="cosine")
    return s
