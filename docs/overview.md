# Audiobookshelf RAG ingester overview

**What:** mirrors an Audiobookshelf server (audiobooks, podcasts) into the member-tier Qdrant collection `hs_member`, plus one `hs_catalog` point per book work, for RAG (PDX Hackerspace pipeline plans 00 and 03). Member tier only: use a dedicated non-admin ABS user; listening progress, sessions, bookmarks and user lists are never read.
**Runs as:** worker that syncs every `POLL_INTERVAL_SECONDS` (or once, `--once`). No port; liveness is a heartbeat file.
**Talks to:** Audiobookshelf REST API (read-only), OpenAI embeddings API, Qdrant.

## Files

- `src/audiobookshelf_rag_ingester/__main__.py`: command line (sync, `query`, `healthcheck`), logging, SIGTERM.
- `config.py`: `Settings` from environment variables.
- `abs_client.py`: ABS REST client (libraries, paginated items, expanded item, authors, ebook download), retries.
- `models.py`: normalizes minified/expanded ABS JSON into dataclasses; `LibraryItem.fingerprint()`.
- `documents.py`: pure functions that build chunks (`Document`: key → point id, content hash, payload).
- `work.py`: `work_id` (ISBN, else normalized title + first author) for cross-format matching.
- `text_utils.py`: HTML stripping, durations, sentence and line chunkers.
- `extractors.py`: DRM-free EPUB/PDF text extraction.
- `embeddings.py`: OpenAI and SentenceTransformers embedders.
- `qdrant_store.py`: collection setup, payload indexes, upsert, diff/delete helpers, search.
- `service.py`: the sync cycle (diff by fingerprint, then content hash; delete what vanished).
- `heartbeat.py`: heartbeat file for the production health check.
- `query.py`: search and result formatting for `query`.
- `tests/`: pytest; `fakes.py` has a fake ABS client and embedder, `conftest.py` an in-memory Qdrant.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Qdrant), `.test`, `.lint`, `.prod` (with Qdrant).

## Sync invariants

- Point id = uuid5 of a document key (`item:<id>:metadata`, `item:<id>:episode:<eid>:<n>`, …).
- Unchanged items are skipped by `item_fingerprint` and `pipeline_signature` payload fields.
- Changed items embed only chunks whose `content_hash` changed, then delete the item's other points.
- The fingerprint is written last, so a failed item is retried next cycle.
- A stopped (SIGTERM) cycle never deletes anything.
- Bump `DOCUMENT_SCHEMA_VERSION` in `service.py` when document text or payload layout changes.

- Payload follows the shared layout (`tier`, `authority`, `doc_id`, `work_ids`, `sync_run_id`, ...); `updated_at` is excluded from the content hash.
- Deletions need a complete listing: a library listing under half the indexed item count deletes nothing.
- Not here (other services): topic taxonomy mapping, ISBN/ASIN enrichment, the Postgres staging store and catalog table, merging formats into one work point.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
