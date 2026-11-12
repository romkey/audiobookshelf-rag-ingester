# Audiobookshelf RAG ingester overview

**What:** mirrors an Audiobookshelf server (audiobooks, podcasts, attached ebooks) into a Qdrant collection for RAG.
**Runs as:** worker that syncs every `POLL_INTERVAL_SECONDS` (or once, `--once`). No port; liveness is a heartbeat file.
**Talks to:** Audiobookshelf REST API (read-only), Ollama (embeddings), Qdrant.

## Files

- `src/audiobookshelf_rag_ingester/__main__.py`: command line (sync, `query`, `healthcheck`), logging, SIGTERM.
- `config.py`: `Settings` from environment variables.
- `abs_client.py`: ABS REST client (libraries, paginated items, expanded item, authors, ebook download), retries.
- `models.py`: normalizes minified/expanded ABS JSON into dataclasses; `LibraryItem.fingerprint()`.
- `documents.py`: pure functions that build chunks (`Document`: key → point id, content hash, payload).
- `text_utils.py`: HTML stripping, durations, sentence and line chunkers.
- `extractors.py`: DRM-free EPUB/PDF text extraction.
- `embeddings.py`: Ollama and SentenceTransformers embedders.
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

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
