"""Staging store, entity registry, catalog table and run bookkeeping.

The tables other sources share (staging, registry, catalog, quarantine,
status) are created only if missing, so every ingester in the pipeline can
create them; their layout is the contract described in docs/overview.md.
The CREATE statements are identical to those of the pipeline's ebook source
(calibre-web2rag); keep them in step. The audiobook columns of
``catalog_items`` are added if missing.

Revision ID: 0001
Revises:
Create Date: 2026-11-18
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_UPGRADE = """
CREATE TABLE IF NOT EXISTS staging_documents (
    source            text        NOT NULL,
    source_doc_id     text        NOT NULL,
    title             text,
    url               text,
    tier              text        NOT NULL CHECK (tier IN ('public', 'member')),
    raw_content       jsonb       NOT NULL,
    extracted_text    text        NOT NULL,
    content_hash      text        NOT NULL,
    flags             jsonb       NOT NULL DEFAULT '{}'::jsonb,
    indexable         boolean     NOT NULL DEFAULT false,
    source_updated_at text,
    fetched_at        timestamptz NOT NULL,
    sync_run_id       text        NOT NULL,
    PRIMARY KEY (source, source_doc_id)
);

CREATE TABLE IF NOT EXISTS quarantine (
    id            bigserial   PRIMARY KEY,
    source        text        NOT NULL,
    source_doc_id text        NOT NULL,
    reason        text        NOT NULL,
    detail        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    sync_run_id   text,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS quarantine_source_idx ON quarantine (source, source_doc_id);

CREATE TABLE IF NOT EXISTS sync_runs (
    run_id      text        PRIMARY KEY,
    source      text        NOT NULL,
    mode        text        NOT NULL,
    started_at  timestamptz NOT NULL,
    finished_at timestamptz,
    ok          boolean,
    seen        integer     NOT NULL DEFAULT 0,
    added       integer     NOT NULL DEFAULT 0,
    updated     integer     NOT NULL DEFAULT 0,
    deleted     integer     NOT NULL DEFAULT 0,
    quarantined integer     NOT NULL DEFAULT 0,
    errors      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    message     text
);
CREATE INDEX IF NOT EXISTS sync_runs_source_idx ON sync_runs (source, mode, started_at DESC);

CREATE TABLE IF NOT EXISTS llm_cache (
    task       text        NOT NULL,
    model      text        NOT NULL,
    input_hash text        NOT NULL,
    output     jsonb       NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (task, model, input_hash)
);

CREATE TABLE IF NOT EXISTS enrichment (
    source        text        NOT NULL,
    source_doc_id text        NOT NULL,
    origin        text        NOT NULL,
    lookup_key    text        NOT NULL,
    matched       boolean     NOT NULL,
    confidence    real,
    fields        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    fetched_at    timestamptz NOT NULL,
    PRIMARY KEY (source, source_doc_id, origin)
);

CREATE TABLE IF NOT EXISTS registry_series (
    series_id  text        PRIMARY KEY,
    name       text        NOT NULL,
    aliases    text[]      NOT NULL DEFAULT '{}',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry_works (
    work_id      text        PRIMARY KEY,
    title        text        NOT NULL,
    authors      text[]      NOT NULL DEFAULT '{}',
    isbn13       text,
    series_id    text        REFERENCES registry_series (series_id) ON DELETE SET NULL,
    series_index real,
    aliases      text[]      NOT NULL DEFAULT '{}',
    topics       text[]      NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry_manifestations (
    source     text        NOT NULL,
    source_id  text        NOT NULL,
    work_id    text        NOT NULL REFERENCES registry_works (work_id),
    format     text        NOT NULL,
    url        text,
    details    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, source_id)
);
CREATE INDEX IF NOT EXISTS registry_manifestations_work_idx ON registry_manifestations (work_id);

-- Merge decisions survive re-indexing: from_work_id is the ID a source
-- computed; to_work_id the work it belongs to ('merge') or must stay apart
-- from ('distinct').
CREATE TABLE IF NOT EXISTS registry_merges (
    from_work_id text        NOT NULL,
    to_work_id   text        NOT NULL,
    decision     text        NOT NULL CHECK (decision IN ('merge', 'distinct')),
    decided_by   text        NOT NULL,
    similarity   real,
    decided_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (from_work_id, to_work_id)
);

CREATE TABLE IF NOT EXISTS registry_review_queue (
    id          bigserial   PRIMARY KEY,
    kind        text        NOT NULL,
    left_id     text        NOT NULL,
    right_id    text        NOT NULL,
    similarity  real,
    details     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz,
    UNIQUE (kind, left_id, right_id)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_type text NOT NULL,
    entity_id   text NOT NULL,
    alias       text NOT NULL,
    PRIMARY KEY (entity_type, entity_id, alias)
);

-- The flat catalog the realtime catalog tool queries: one row per
-- manifestation (an ebook, an audiobook, a hardcopy).
CREATE TABLE IF NOT EXISTS catalog_items (
    source         text        NOT NULL,
    source_id      text        NOT NULL,
    work_id        text        NOT NULL,
    format         text        NOT NULL,
    title          text        NOT NULL,
    sort_title     text,
    subtitle       text,
    authors        text[]      NOT NULL DEFAULT '{}',
    series         text,
    series_index   real,
    publisher      text,
    published_year integer,
    languages      text[]      NOT NULL DEFAULT '{}',
    tags           text[]      NOT NULL DEFAULT '{}',
    topics         text[]      NOT NULL DEFAULT '{}',
    file_formats   text[]      NOT NULL DEFAULT '{}',
    rating         integer,
    url            text,
    location       text,
    date_added     timestamptz,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, source_id)
);
CREATE INDEX IF NOT EXISTS catalog_items_work_idx ON catalog_items (work_id);
CREATE INDEX IF NOT EXISTS catalog_items_format_idx ON catalog_items (format);
CREATE INDEX IF NOT EXISTS catalog_items_topics_idx ON catalog_items USING gin (topics);
CREATE INDEX IF NOT EXISTS catalog_items_tags_idx ON catalog_items USING gin (tags);

-- Audiobook filters for the catalog tool: narrator, length, genre, abridged.
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS narrators text[] NOT NULL DEFAULT '{}';
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS duration_seconds real;
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS genres text[] NOT NULL DEFAULT '{}';
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS abridged boolean;
CREATE INDEX IF NOT EXISTS catalog_items_narrators_idx ON catalog_items USING gin (narrators);
CREATE INDEX IF NOT EXISTS catalog_items_genres_idx ON catalog_items USING gin (genres);
"""

# Only this source's rows: the shared tables may hold other sources' data.
_DOWNGRADE = """
DELETE FROM catalog_items WHERE source = 'audiobookshelf';
DELETE FROM registry_manifestations WHERE source = 'audiobookshelf';
DELETE FROM staging_documents WHERE source = 'audiobookshelf';
DELETE FROM enrichment WHERE source = 'audiobookshelf';
DELETE FROM sync_runs WHERE source = 'audiobookshelf';
DELETE FROM quarantine WHERE source = 'audiobookshelf';
"""


def upgrade() -> None:
    op.execute(_UPGRADE)


def downgrade() -> None:
    op.execute(_DOWNGRADE)
