import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    ExtendedPointId,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from audiobookshelf_rag_ingester.documents import Document

logger = logging.getLogger(__name__)

# Any: point payloads are free-form JSON that this app writes.
type Payload = dict[str, Any]

_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "chunk_type": PayloadSchemaType.KEYWORD,
    "library_id": PayloadSchemaType.KEYWORD,
    "media_type": PayloadSchemaType.KEYWORD,
    "item_id": PayloadSchemaType.KEYWORD,
    "episode_id": PayloadSchemaType.KEYWORD,
    "author_id": PayloadSchemaType.KEYWORD,
    "authors": PayloadSchemaType.KEYWORD,
    "narrators": PayloadSchemaType.KEYWORD,
    "series": PayloadSchemaType.KEYWORD,
    "genres": PayloadSchemaType.KEYWORD,
    "tags": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    "published_year": PayloadSchemaType.KEYWORD,
    "episode_published_at": PayloadSchemaType.DATETIME,
}


def _distance(name: str) -> Distance:
    match name.lower():
        case "cosine":
            return Distance.COSINE
        case "dot":
            return Distance.DOT
        case "euclid":
            return Distance.EUCLID
        case "manhattan":
            return Distance.MANHATTAN
        case _:
            raise ValueError(f"Unsupported Qdrant distance metric: {name}")


def _match(key: str, value: Any) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


class QdrantStore:
    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self.client = client or QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    def ensure_collection(self, *, vector_size: int, distance: str) -> None:
        if not self.client.collection_exists(self.collection):
            logger.info(
                "Creating Qdrant collection %r (size=%d, distance=%s)",
                self.collection,
                vector_size,
                distance,
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size, distance=_distance(distance)
                ),
            )
        else:
            existing = self._existing_vector_size()
            if existing is not None and existing != vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' has vector size "
                    f"{existing}, but the "
                    f"embedding model produces {vector_size}. "
                    "Use a new QDRANT_COLLECTION name "
                    "or delete the collection."
                )
        for field_name, schema in _PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                logger.debug(
                    "Payload index %r not created (may already exist)", field_name
                )

    def _existing_vector_size(self) -> int | None:
        info = self.client.get_collection(self.collection)
        vectors: Any = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            if hasattr(first, "size"):
                return int(first.size)
        return None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_payloads(self, ids: Sequence[str], fields: list[str]) -> dict[str, Payload]:
        if not ids:
            return {}
        result: dict[str, Payload] = {}
        for start in range(0, len(ids), 256):
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=list(ids[start : start + 256]),
                with_payload=fields,
                with_vectors=False,
            )
            for p in points:
                result[str(p.id)] = p.payload or {}
        return result

    def scroll_ids(
        self, flt: Filter, fields: list[str] | None = None
    ) -> dict[str, Payload]:
        """Return {point_id: payload} for every point matching ``flt``."""
        out: dict[str, Payload] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=fields or False,
                with_vectors=False,
            )
            for p in points:
                out[str(p.id)] = p.payload or {}
            if offset is None:
                break
        return out

    def item_ids_in_library(self, library_id: str) -> set[str]:
        flt = Filter(
            must=[
                _match("library_id", library_id),
                _match("chunk_type", "item_metadata"),
            ]
        )
        return {
            str(payload.get("item_id"))
            for payload in self.scroll_ids(flt, ["item_id"]).values()
            if payload.get("item_id")
        }

    def library_ids(self) -> set[str]:
        flt = Filter(must=[_match("source", "audiobookshelf")])
        return {
            str(p.get("library_id"))
            for p in self.scroll_ids(flt, ["library_id"]).values()
            if p.get("library_id")
        }

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        docs: Sequence[Document],
        vectors: Sequence[list[float]],
        extra_payload: dict[str, Any] | None = None,
    ) -> int:
        if len(docs) != len(vectors):
            raise ValueError("docs and vectors must have equal lengths")
        if not docs:
            return 0
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=doc.point_id,
                vector=vector,
                payload={
                    **doc.payload,
                    **(extra_payload or {}),
                    "chunk_type": doc.chunk_type,
                    "doc_key": doc.key,
                    "text": doc.text,
                    "context": doc.context,
                    "content_hash": doc.content_hash,
                    "ingested_at": now,
                },
            )
            for doc, vector in zip(docs, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(
                collection_name=self.collection, points=points[start : start + 256]
            )
        return len(points)

    def set_payload(self, flt: Filter, payload: dict[str, Any]) -> None:
        self.client.set_payload(
            collection_name=self.collection, payload=payload, points=flt
        )

    def delete_where(self, flt: Filter) -> None:
        self.client.delete(
            collection_name=self.collection, points_selector=FilterSelector(filter=flt)
        )

    def delete_stale(self, scope: Filter, keep_ids: set[str]) -> int:
        """Delete points inside ``scope`` whose id is not in ``keep_ids``."""
        existing = set(self.scroll_ids(scope).keys())
        stale = existing - keep_ids
        if not stale:
            return 0
        ids: list[ExtendedPointId] = sorted(stale)
        for start in range(0, len(ids), 256):
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=ids[start : start + 256]),
            )
        return len(stale)

    def delete_items(self, item_ids: Sequence[str]) -> None:
        if not item_ids:
            return
        self.delete_where(
            Filter(
                must=[FieldCondition(key="item_id", match=MatchAny(any=list(item_ids)))]
            )
        )

    def delete_library(self, library_id: str) -> None:
        self.delete_where(Filter(must=[_match("library_id", library_id)]))

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        chunk_type: str | None = None,
        library_id: str | None = None,
        min_score: float | None = None,
    ) -> list[tuple[float, Payload]]:
        must: list[Condition] = []
        if chunk_type:
            must.append(_match("chunk_type", chunk_type))
        if library_id:
            must.append(_match("library_id", library_id))
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=Filter(must=must) if must else None,
            limit=limit,
            with_payload=True,
            score_threshold=min_score,
        ).points
        return [(hit.score, hit.payload or {}) for hit in hits]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count


def item_scope(item_id: str) -> Filter:
    return Filter(must=[_match("item_id", item_id)])


def item_chunk_scope(item_id: str, chunk_type: str) -> Filter:
    return Filter(must=[_match("item_id", item_id), _match("chunk_type", chunk_type)])


def library_chunk_scope(library_id: str, chunk_type: str) -> Filter:
    return Filter(
        must=[_match("library_id", library_id), _match("chunk_type", chunk_type)]
    )
