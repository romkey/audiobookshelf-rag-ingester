# audiobookshelf-rag-ingester overview

**What:** the audiobook and podcast source of the PDX Hackerspace RAG
pipeline (plans 00 "shared architecture" and 03 "audiobook library"),
following the pattern of the ebook source, calibre-web2rag (plan 02).
Member tier only.
**Runs as:** a long-running worker with no port. The default command,
`serve`, scans every `SCAN_INTERVAL_MINUTES` (default 60): incremental scans,
plus a full reconcile when none has finished in `FULL_SCAN_INTERVAL_HOURS`
(default 24). It touches a heartbeat file that `healthcheck` checks.
`run --mode incremental|full|reindex` does one scan and exits; `status`
prints recent runs as JSON. Every command except `healthcheck` first applies
pending schema migrations.
**Talks to:** the Audiobookshelf REST API (read-only, a non-admin token),
PostgreSQL and Qdrant (both shared with the pipeline's other sources), an
OpenAI-compatible gateway (embeddings and generation), Open Library and
Google Books.

## Flow

1. List the selected libraries' items (minified) with their `updatedAt`.
2. For each new or changed item, fetch it expanded. Audiobooks: clean the
   description and chapter titles, redact secrets (quarantine log), enrich
   from Open Library/Google Books if description or genres are missing, ask
   the LLM for topics and a "good for" summary, write the staging row.
   Podcasts: the same for the podcast and each episode, plus LLM topics.
3. Registry (audiobooks only): strip "(Unabridged)" and series markers from
   the title, compute `work_id` exactly as the ebook source does (ISBN-13,
   else title slug + first author's surname), follow merge decisions, fuzzy
   match against other sources' works (auto-merge, or merge when series and
   sequence agree, or queue for review), attach the `audiobook`
   manifestation; then the flat `catalog_items` row (with narrators,
   duration, genres).
4. Index: one `hs_catalog` point per touched work, built from every
   manifestation in the registry (all formats); each podcast and episode in
   `hs_member` (`source: podcast`); the regenerated library overview in
   `hs_member` (`source: audiobookshelf`). Dense + sparse vectors; unchanged
   text (by content hash) is not re-embedded.
5. Deletions propagate every run; `full` also deletes points not stamped
   with this run's ID. A run seeing < half the expected items deletes
   nothing and exits 1; nothing is deleted for a library whose listing
   failed. An advisory lock stops overlapping runs.

## Files

- `__main__.py`: CLI, logging, SIGTERM, exit codes.
- `config.py`: `Settings.from_env()`; validates every variable.
- `abs_client.py`: Audiobookshelf REST client with retries.
- `models.py`: normalizes minified and expanded ABS JSON into dataclasses.
- `pipeline.py`: `Pipeline.run(mode)`: staging, registry, indexing, reconcile.
- `documents.py`: point IDs, payloads, catalog, overview and podcast text (pure).
- `db.py`: psycopg queries; `migrate()`. `migrations/`: Alembic schema.
- `works.py`: ISBN normalization, work IDs, title similarity, audiobook title cleanup.
- `sanitize.py`: secret scanner. `linker.py`: alias matching.
- `llm.py`: gateway chat tasks, cached. `enrichment.py`: book lookups.
- `embeddings.py`: dense embedders. `sparse.py`: keyword vectors.
- `qdrant_store.py`: collections, payload indexes, upserts, reconcile.
- `taxonomy.txt`: the shared topic taxonomy.

## Shared with the ebook source (keep identical)

The SQL in `migrations/versions/v0001_*.py` (`CREATE ... IF NOT EXISTS` of
the shared tables), `works.py` up to the audiobook section, `sanitize.py`,
`linker.py`, `sparse.py` (and its `SPARSE_ENCODER` id), `chunking.py`,
`schedule.py` (the `serve` loop and heartbeat), `db.last_full_run_at`,
`taxonomy.txt`, and in `documents.py` `point_id`, `base_payload` and
`catalog_text`. Both sources rebuild the same `hs_catalog` point for a
work, so their catalog text and payload must match. This repo's Alembic
version table is `audiobookshelf_rag_ingester_alembic_version`; its
migration also adds the audiobook columns of `catalog_items`.

Shared tables: `staging_documents`, `registry_works`,
`registry_manifestations`, `registry_series`, `registry_merges`,
`registry_review_queue`, `entity_aliases`, `catalog_items`, `quarantine`,
`sync_runs`, `llm_cache`, `enrichment`.

## Conventions

- Configuration from environment variables; see docs/configuration.md.
- One bad item is logged and skipped; it never stops the run.
- No listening progress, sessions, bookmarks or users are read: only
  `/api/libraries`, `/api/libraries/:id/items` and `/api/items/:id?expanded=1`.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
