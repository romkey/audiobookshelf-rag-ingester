# Audiobookshelf RAG ingester

[![CI](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`audiobookshelf-rag-ingester` is the audiobook and podcast source of the
PDX Hackerspace RAG pipeline, feeding the member chatbot (member tier only).
It reads an [Audiobookshelf](https://www.audiobookshelf.org/) server through
its REST API with a non-admin token, stages every item in PostgreSQL, links
each audiobook to a cross-format *work* in the shared entity registry (so an
audiobook and an ebook of the same book are one record), fills gaps from
Open Library and Google Books, maps items to the shared topic taxonomy with
a local LLM, and indexes them into Qdrant: one hybrid (dense + keyword)
point per work in `hs_catalog`, each podcast and episode in `hs_member`,
and a library overview document in `hs_member`. It also writes the flat
catalog table that the realtime catalog tool counts and filters (narrator,
length, genre).

It follows the same pattern, schema and shared code as the ebook source,
[calibre-web2rag](https://github.com/romkey/pdxhackerspace-calibre-web-rag-injester);
run both against the same PostgreSQL and Qdrant.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/audiobookshelf_rag_ingester/__main__.py` | Command line: `run`, `migrate`, `status`; logging, SIGTERM, exit codes |
| `src/audiobookshelf_rag_ingester/config.py` | Settings from environment variables, validated at startup |
| `src/audiobookshelf_rag_ingester/pipeline.py` | One sync run: staging, registry, indexing, reconcile |
| `src/audiobookshelf_rag_ingester/abs_client.py`, `models.py` | Read-only Audiobookshelf client; normalized items |
| `src/audiobookshelf_rag_ingester/documents.py` | Point IDs, the shared payload, catalog, overview and podcast text |
| `src/audiobookshelf_rag_ingester/db.py`, `migrations/` | PostgreSQL queries; Alembic schema |
| `src/audiobookshelf_rag_ingester/works.py` | ISBN normalization, `work_id`, title similarity, audiobook title cleanup |
| `src/audiobookshelf_rag_ingester/sanitize.py` | Deterministic secret scanner and redaction |
| `src/audiobookshelf_rag_ingester/enrichment.py` | Open Library and Google Books lookups |
| `src/audiobookshelf_rag_ingester/llm.py` | Ingest-time LLM tasks through the gateway, cached by input hash |
| `src/audiobookshelf_rag_ingester/linker.py` | Links chunks to registry entities by alias |
| `src/audiobookshelf_rag_ingester/embeddings.py`, `sparse.py` | Dense embedders; keyword (sparse) vectors |
| `src/audiobookshelf_rag_ingester/qdrant_store.py` | Hybrid collections, payload indexes, upserts, reconcile |
| `src/audiobookshelf_rag_ingester/taxonomy.txt` | The shared topic taxonomy |
| `tests/` | pytest against real PostgreSQL, in-memory Qdrant and a fake Audiobookshelf |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | Short architecture summary and what is shared with the ebook source |

## Technical details

Command line:

```bash
audiobookshelf-rag-ingester run --mode incremental   # hourly: changed (updatedAt) and deleted items
audiobookshelf-rag-ingester run --mode full          # nightly: every item, then delete what wasn't seen
audiobookshelf-rag-ingester run --mode reindex       # rebuild every point from staging, no ABS read
audiobookshelf-rag-ingester status                   # recent runs, quarantine and catalog counts (JSON)
audiobookshelf-rag-ingester migrate                  # only apply schema migrations
audiobookshelf-rag-ingester --version
```

`audiobookshelf-rag-ingester` alone is `run --mode full`. Every command first
applies pending PostgreSQL migrations. Exit codes: 0 on success (including a
clean stop on SIGTERM, and a run skipped because another holds the lock), 1
if the run fails, the safety valve held back deletions or a library's
listing failed, 2 for invalid configuration.

**Access.** Use a dedicated Audiobookshelf user that is not an admin and
can see the libraries to index, and its API token. The ingester calls only
`GET /api/libraries`, `GET /api/libraries/:id/items` (minified, paginated)
and `GET /api/items/:id?expanded=1`. It never reads listening progress,
sessions, bookmarks or user lists, and none of those reach staging.

**What a run does.** For every item in the selected libraries it stages
title, subtitle, authors, narrators, series and sequence, publisher, year,
language, ISBN and ASIN, description (HTML cleaned to text), genres, tags,
explicit and abridged flags, duration, chapter titles, number of audio
files, cover path and date added; for podcasts, each episode's title, show
notes, date and duration. Secrets are redacted and logged to the
`quarantine` table. Missing descriptions or genres are looked up (Open
Library, then Google Books, by ISBN or title and author) and stored apart
from library data with their origin. With `LLM_MODEL` set, an LLM maps each
item to 1–5 slugs of the shared taxonomy and, when an audiobook's
description is missing or marketing copy, writes a "what this book is good
for" summary, marked machine-generated. LLM results are cached by input hash.

**Works.** An audiobook's `work_id` is computed exactly as the ebook source
computes it: `isbn:<ISBN-13>`, else `title:<normalized-title>--<surname>`.
"(Unabridged)" and series markers ("Wayfarers 1: …", "…: The Expanse, Book
1") are stripped first. If no work has that ID, the audiobook is matched
against other sources' works by the first author's surname and title
similarity: 0.93 or more merges, 0.80 or more merges when the series and
sequence also agree, otherwise it is queued in `registry_review_queue`.
Decisions are kept in `registry_merges`. Each audiobook is attached as a
manifestation `{format: audiobook, source: audiobookshelf, source_id, url,
duration, narrators}`; an Audiobookshelf item with no audio counts as an
`ebook` manifestation.

**What gets indexed.**

| Collection | `source` | One point per | Text |
|------------|----------|---------------|------|
| `hs_catalog` | `catalog` | work (every format) | Title, authors, series and number, formats, narrators, length, topics, tags, description or summary, contents, audiobook chapter titles |
| `hs_member` | `podcast` | podcast, plus long-description chunks | Title, hosts, genres, topics, episode count, description |
| `hs_member` | `podcast` | episode chunk | Episode title, podcast, season and number, date, length, show notes |
| `hs_member` | `audiobookshelf` | overview section | Size and hours, strongest topics and genres, how to get an account, which apps work |

Every point carries the shared payload (`id` = uuid5 of source, doc and
chunk index; `doc_id`, `chunk_index`, `title`, `heading_path`, `text`,
`context`, `url`, `tier: member`, `authority: 2`, timestamps,
`content_hash`, `sync_run_id`, `asset_ids`, `location_ids`, `work_ids`,
`event_ids`, `training_ids`, `topics`, `lang`). Catalog points add
`work_id`, `authors`, `series`, `series_index`, `formats`, `narrators`,
`duration_seconds` and `manifestations`, so series queries sort by
`series_index` across formats. Podcast points add `item_id`, `kind`
(`podcast` or `episode`), `episode_id`, `published_at` and
`duration_seconds`.

**Catalog table.** Each audiobook is a row in `catalog_items` (format
`audiobook`) with `narrators`, `duration_seconds`, `genres`, `abridged`,
`series` and `series_index` besides the shared columns; counts and filters
such as "audiobooks under five hours" are answered from it, never from the
index.

**Sync.** `incremental` fetches only items whose `updatedAt` changed;
`full` fetches every item and deletes `hs_member` points this source wrote
that the run didn't see. Unchanged text is never re-embedded. A run that
sees less than half the items of the last full run (or of staging) deletes
nothing and exits 1; nothing is deleted for a library whose listing failed.

## Running it

Requires Docker, an Audiobookshelf token for a non-admin user, PostgreSQL
and Qdrant (shared with the pipeline's other sources), and an
OpenAI-compatible embedding server. Configuration comes from `.env` (see
[docs/configuration.md](docs/configuration.md)).

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
```

That starts the latest release from
`ghcr.io/romkey/audiobookshelf-rag-ingester`, plus PostgreSQL and Qdrant.
It runs on its own: the first scan is a full one, then it sleeps
`SCAN_INTERVAL_MINUTES` (default 60) between incremental scans and does a
full reconcile once every `FULL_SCAN_INTERVAL_HOURS` (default 24). No cron
needed. To use the pipeline's existing PostgreSQL and Qdrant, delete those
services from the compose file and set `DATABASE_URL` and `QDRANT_URL`. Set
`APP_VERSION=X.Y.Z` to pin a release. For a one-off scan:

```bash
docker compose -f docker-compose.prod.yaml run --rm app audiobookshelf-rag-ingester run --mode reindex
```

### With pip

Needs Python 3.15 or newer:

```bash
python3 -m venv ~/.venvs/audiobookshelf-rag-ingester
~/.venvs/audiobookshelf-rag-ingester/bin/pip install https://github.com/romkey/audiobookshelf-rag-ingester/releases/download/vX.Y.Z/audiobookshelf_rag_ingester-X.Y.Z-py3-none-any.whl
~/.venvs/audiobookshelf-rag-ingester/bin/audiobookshelf-rag-ingester --version
```

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # one full sync from the working tree
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage, against PostgreSQL
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/) and a PostgreSQL you can reach:

```bash
uv sync
DATABASE_URL=postgresql://app:app@localhost:5432/app uv run pytest
uv run mypy
```

`uvx pre-commit install` runs the lint checks on every commit.

## License

[MIT](LICENSE)
