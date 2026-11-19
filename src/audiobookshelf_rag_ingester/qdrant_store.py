"""Qdrant collections with a dense and a sparse (keyword) vector per point."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

# Payload indexes every text collection gets (shared architecture), plus
# the catalog's own filter fields.
COMMON_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "source": qm.PayloadSchemaType.KEYWORD,
    "tier": qm.PayloadSchemaType.KEYWORD,
    "doc_id": qm.PayloadSchemaType.KEYWORD,
    "asset_ids": qm.PayloadSchemaType.KEYWORD,
    "location_ids": qm.PayloadSchemaType.KEYWORD,
    "topics": qm.PayloadSchemaType.KEYWORD,
    "section": qm.PayloadSchemaType.KEYWORD,
    "updated_at": qm.PayloadSchemaType.DATETIME,
    "sync_run_id": qm.PayloadSchemaType.KEYWORD,
}
CATALOG_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "work_id": qm.PayloadSchemaType.KEYWORD,
    "authors": qm.PayloadSchemaType.KEYWORD,
    "formats": qm.PayloadSchemaType.KEYWORD,
    "languages": qm.PayloadSchemaType.KEYWORD,
    "series": qm.PayloadSchemaType.KEYWORD,
    "tags": qm.PayloadSchemaType.KEYWORD,
    "year": qm.PayloadSchemaType.INTEGER,
    "narrators": qm.PayloadSchemaType.KEYWORD,
}
# Podcast points in hs_member: per-podcast cleanup and episode filters.
PODCAST_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "item_id": qm.PayloadSchemaType.KEYWORD,
    "kind": qm.PayloadSchemaType.KEYWORD,
    "published_at": qm.PayloadSchemaType.DATETIME,
}


class CollectionMismatchError(RuntimeError):
    """An existing collection was built with a different model or layout."""


def _distance(name: str) -> qm.Distance:
    match name.lower():
        case "cosine":
            return qm.Distance.COSINE
        case "dot":
            return qm.Distance.DOT
        case "euclid":
            return qm.Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported Qdrant distance metric: {name}")


def _match(key: str, value: str) -> qm.FieldCondition:
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


class QdrantStore:
    """One collection. ``metadata`` records the embedding model and encoders."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        distance: str,
        # Any: collection metadata is free-form JSON.
        metadata: Mapping[str, Any],
        indexes: Mapping[str, qm.PayloadSchemaType] = COMMON_INDEXES,
    ) -> None:
        self.client = client
        self.collection = collection
        self._ensure(vector_size, distance, dict(metadata), indexes)

    def _ensure(
        self,
        vector_size: int,
        distance: str,
        # Any: collection metadata is free-form JSON.
        metadata: dict[str, Any],
        indexes: Mapping[str, qm.PayloadSchemaType],
    ) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: qm.VectorParams(size=vector_size, distance=_distance(distance))
                },
                sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
                metadata=metadata,
            )
            logger.info("Created Qdrant collection %s", self.collection)
        else:
            self._check_existing(vector_size, metadata)
        for field_name, schema in indexes.items():
            # Creating an index that exists is a no-op in Qdrant.
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field_name, field_schema=schema
            )

    # Any: collection metadata is free-form JSON.
    def _check_existing(self, vector_size: int, metadata: dict[str, Any]) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        if not isinstance(vectors, dict) or DENSE not in vectors or SPARSE not in sparse:
            raise CollectionMismatchError(
                f"Qdrant collection {self.collection!r} exists without the {DENSE!r} and "
                f"{SPARSE!r} vectors; delete it (indexes are rebuilt from staging) or pick "
                "another collection name"
            )
        if vectors[DENSE].size != vector_size:
            raise CollectionMismatchError(
                f"Qdrant collection {self.collection!r} has {vectors[DENSE].size}-dimension "
                f"vectors but VECTOR_SIZE is {vector_size}"
            )
        existing = info.config.metadata or {}
        model = existing.get("embedding_model")
        if model is None:
            self.client.update_collection(collection_name=self.collection, metadata=metadata)
        elif model != metadata.get("embedding_model"):
            raise CollectionMismatchError(
                f"Qdrant collection {self.collection!r} was embedded with {model!r}, not "
                f"{metadata.get('embedding_model')!r}. Changing the model means a full "
                "re-embed: recreate the collection, then run every source with --mode reindex"
            )

    def upsert(self, points: list[qm.PointStruct]) -> None:
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def existing_hashes(self, point_ids: Iterable[str]) -> dict[str, str | None]:
        """point id -> payload content_hash, for the points that exist."""
        ids = list(point_ids)
        found: dict[str, str | None] = {}
        for start in range(0, len(ids), 256):
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=ids[start : start + 256],
                with_payload=["content_hash"],
            )
            for record in records:
                found[str(record.id)] = (record.payload or {}).get("content_hash")
        return found

    def existing_state(self, point_ids: Iterable[str]) -> dict[str, tuple[str | None, str | None]]:
        """point id -> (content_hash, payload_hash), for the points that exist."""
        ids = list(point_ids)
        found: dict[str, tuple[str | None, str | None]] = {}
        for start in range(0, len(ids), 256):
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=ids[start : start + 256],
                with_payload=["content_hash", "payload_hash"],
            )
            for record in records:
                payload = record.payload or {}
                found[str(record.id)] = (payload.get("content_hash"), payload.get("payload_hash"))
        return found

    # Any: payload values are JSON.
    def replace_payloads(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        """Rewrite each point's payload without touching its vectors."""
        for start in range(0, len(items), 256):
            self.client.batch_update_points(
                collection_name=self.collection,
                update_operations=[
                    qm.OverwritePayloadOperation(
                        overwrite_payload=qm.SetPayload(payload=payload, points=[pid])
                    )
                    for pid, payload in items[start : start + 256]
                ],
            )

    # Any: payload values are JSON.
    def set_payload(self, point_ids: list[str], payload: dict[str, Any]) -> None:
        if point_ids:
            self.client.set_payload(
                collection_name=self.collection,
                payload=payload,
                points=qm.PointIdsList(points=list(point_ids)),
            )

    # Any: payload values are JSON.
    def set_payload_for_doc(self, source: str, doc_id: str, payload: dict[str, Any]) -> None:
        self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=qm.Filter(must=[_match("source", source), _match("doc_id", doc_id)]),
        )

    def delete_ids(self, point_ids: list[str]) -> None:
        if point_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(points=list(point_ids)),
            )

    def delete_docs(self, source: str, doc_ids: list[str]) -> None:
        if doc_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            _match("source", source),
                            qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)),
                        ]
                    )
                ),
            )

    def delete_item_points(self, source: str, item_ids: list[str], keep: set[str]) -> int:
        """Delete points of these items (payload ``item_id``) whose id isn't in ``keep``."""
        if not item_ids:
            return 0
        selector = qm.Filter(
            must=[
                _match("source", source),
                qm.FieldCondition(key="item_id", match=qm.MatchAny(any=item_ids)),
            ]
        )
        stale: list[qm.ExtendedPointId] = []
        offset: Any = None  # Any: Qdrant's opaque scroll offset
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=selector,
                limit=512,
                offset=offset,
                with_payload=False,
            )
            stale.extend(r.id for r in records if str(r.id) not in keep)
            if offset is None:
                break
        if stale:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(points=stale),
            )
        return len(stale)

    def delete_stale(self, source: str, run_id: str) -> int:
        """Delete this source's points not stamped with ``run_id``."""
        stale = qm.Filter(must=[_match("source", source)], must_not=[_match("sync_run_id", run_id)])
        count = self.client.count(collection_name=self.collection, count_filter=stale).count
        if count:
            self.client.delete(
                collection_name=self.collection, points_selector=qm.FilterSelector(filter=stale)
            )
        return count

    def count(self, source: str | None = None) -> int:
        selector = qm.Filter(must=[_match("source", source)]) if source else None
        return self.client.count(collection_name=self.collection, count_filter=selector).count

    # Any: payloads are JSON.
    def payloads(self, source: str | None = None) -> list[dict[str, Any]]:
        """Every payload in the collection (or one source's); for tests and checks."""
        selector = qm.Filter(must=[_match("source", source)]) if source else None
        out: list[dict[str, Any]] = []
        offset: Any = None  # Any: Qdrant's opaque scroll offset
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=selector,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            out.extend(r.payload or {} for r in records)
            if offset is None:
                return out
