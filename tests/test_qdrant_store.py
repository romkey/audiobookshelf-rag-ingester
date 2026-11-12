import pytest
from qdrant_client import QdrantClient

from audiobookshelf_rag_ingester.documents import Document
from audiobookshelf_rag_ingester.qdrant_store import QdrantStore, item_scope


def _doc(key: str, item_id: str = "i1", chunk_type: str = "item_metadata") -> Document:
    payload = {"source": "audiobookshelf", "library_id": "lib", "item_id": item_id}
    return Document(key, chunk_type, f"text {key}", f"context {key}", payload)


def test_upsert_search_and_delete(store: QdrantStore) -> None:
    docs = [_doc("a"), _doc("b", chunk_type="episode"), _doc("c", item_id="i2")]
    vectors = [[1.0] * 8, [0.5] * 4 + [1.0] * 4, [1.0] * 4 + [0.1] * 4]
    assert store.upsert(docs, vectors) == 3
    assert store.count() == 3
    assert store.item_ids_in_library("lib") == {"i1", "i2"}
    assert store.library_ids() == {"lib"}

    hits = store.search([1.0] * 8, limit=5, chunk_type="episode", library_id="lib")
    assert [p["doc_key"] for _score, p in hits] == ["b"]

    assert store.delete_stale(item_scope("i1"), keep_ids={docs[0].point_id}) == 1
    assert store.delete_stale(item_scope("i1"), keep_ids={docs[0].point_id}) == 0
    store.delete_items(["i2"])
    store.delete_items([])
    assert store.count() == 1
    store.delete_library("lib")
    assert store.count() == 0


def test_upsert_length_mismatch(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert([_doc("a")], [])
    assert store.upsert([], []) == 0
    assert store.get_payloads([], ["x"]) == {}


def test_vector_size_mismatch() -> None:
    store = QdrantStore(collection="c", client=QdrantClient(":memory:"))
    store.ensure_collection(vector_size=8, distance="euclid")
    store.ensure_collection(vector_size=8, distance="euclid")  # idempotent
    with pytest.raises(ValueError, match="vector size 8"):
        store.ensure_collection(vector_size=4, distance="euclid")


def test_unknown_distance() -> None:
    store = QdrantStore(collection="c", client=QdrantClient(":memory:"))
    with pytest.raises(ValueError, match="distance"):
        store.ensure_collection(vector_size=8, distance="hamming")
