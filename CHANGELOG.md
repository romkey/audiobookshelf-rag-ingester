# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
