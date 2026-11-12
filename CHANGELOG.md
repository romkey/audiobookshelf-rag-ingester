# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
