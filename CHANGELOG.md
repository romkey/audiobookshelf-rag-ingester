# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Rebuilt as the audiobook and podcast source of the PDX Hackerspace RAG
  pipeline (plans 00 and 03), on the same schema, registry and shared code
  as the ebook source (calibre-web2rag), so an audiobook and an ebook of the
  same book are one work. Existing collections and settings don't carry
  over: recreate the Qdrant collections (the data is disposable) and set the
  new variables in `.env.example`.
- A batch CLI instead of a polling worker: `run --mode incremental|full|reindex`,
  `migrate` and `status`, scheduled from cron. The heartbeat health check,
  `query` command, `POLL_INTERVAL_SECONDS`, `FORCE_REINGEST` and the
  `INGEST_*` switches are gone; `reindex` replaces forced re-ingestion.
- Points use named `dense` and `sparse` (keyword) vectors and the shared
  payload; `source` is `catalog`, `podcast` or `audiobookshelf`.
- Attached ebook text, per-author documents and per-library tallies are no
  longer indexed.

### Added

- PostgreSQL staging store, entity registry (works, manifestations, series,
  merge decisions, review queue), flat `catalog_items` table with narrator,
  duration, genre and abridged columns, LLM cache, enrichment results,
  quarantine and run status (shared tables, created if missing). Requires
  `DATABASE_URL`.
- `work_id` identical to the ebook source's, after stripping "(Unabridged)"
  and series markers; fuzzy matching against other sources' works, with
  series and sequence as the tiebreaker.
- One `hs_catalog` point per work listing every format, with narrators,
  length and audiobook chapter titles in the embedded text.
- Podcasts and episodes in `hs_member` with `source: podcast`.
- Library overview document (size, hours, strongest topics, how to get an
  account, which apps work): `LIBRARY_ACCESS_INFO`, `LIBRARY_APPS_INFO`.
- Open Library and Google Books enrichment, LLM topic mapping to the shared
  taxonomy and "good for" summaries (`LLM_MODEL`), secret redaction with a
  quarantine log, alias linker.
- Run IDs, nightly reconcile, a safety valve that blocks deletions when a
  run sees under half the expected items, no deletions for a library whose
  listing failed, and an advisory lock against overlapping runs.

## [0.2.0] - 2026-11-18

### Changed

- PDX Hackerspace RAG pipeline alignment (member tier). Points now carry the
  shared payload (`tier: member`, `authority`, `doc_id`, `heading_path`,
  `work_ids`, `topics`, `created_at`/`updated_at`/`indexed_at`,
  `sync_run_id`, ...). `source` is `audiobook` for books and `podcast` for
  podcasts and episodes. Existing collections rebuild automatically
  (`DOCUMENT_SCHEMA_VERSION` 2); the data is disposable.
- `QDRANT_COLLECTION` defaults to `hs_member`.
- `INGEST_EBOOKS` defaults to `false`: book text is not searched by default.

### Added

- `work_id` per audiobook (ISBN, else normalized title plus first author;
  "(Unabridged)" and series markers stripped) plus a title-only
  `work_title_key` for cross-format matching. Podcasts are not works.
- `hs_catalog` (`QDRANT_CATALOG_COLLECTION`, empty disables): one point per
  work with this source's `manifestation_audiobook` (format, source id, url,
  duration, narrators), chapter titles in the embedded text. Other sources
  merge their own `manifestation_*` key into the same point.
- `LIBRARY_OVERVIEW_NOTE`: text (how to get an account, which apps work)
  added to the library overview document.
- Every sync run stamps the points it saw with `sync_run_id`.
- Safety valve: a library listing under half the size of what is indexed
  deletes nothing and counts as an error.

## [0.1.0] - 2026-11-12

### Added

- `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX` for models that
  need task prefixes; the default model is `nomic-ai/modernbert-embed-base`.
- Initial release: syncs Audiobookshelf libraries into Qdrant through the
  ABS REST API. Indexes item metadata, long descriptions, audiobook chapter
  lists, podcast episodes, DRM-free EPUB/PDF text, author documents and
  per-library summaries.
- Incremental sync: item fingerprints, per-chunk content hashes, ebook
  signatures, automatic rebuild when embedding or chunking settings change,
  and removal of deleted items and libraries.
- `query` command for searching the collection, and `healthcheck` command
  backed by a heartbeat file.
- OpenAI embeddings by default (optional `OPENAI_API_KEY`, `OPENAI_BASE_URL`
  for compatible servers); SentenceTransformers as an optional extra.
