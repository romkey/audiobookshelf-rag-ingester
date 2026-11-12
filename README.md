# Audiobookshelf RAG ingester

[![CI](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/audiobookshelf-rag-ingester/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Mirrors an [Audiobookshelf](https://www.audiobookshelf.org/) server (audiobooks, podcasts and the ebooks attached to them) into a [Qdrant](https://qdrant.tech/) vector store, so a chatbot or other RAG pipeline can answer questions such as "what's a cozy sci-fi audiobook narrated by Ray Porter?", "which podcast episode covered laser cutter safety?" or "how many Discworld books do we have?". It runs as a small worker that re-syncs every hour. Each sync is incremental: it re-embeds only what changed and removes whatever was deleted from Audiobookshelf. Embeddings come from [Ollama](https://ollama.com/).

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/audiobookshelf_rag_ingester/__main__.py` | Command line: sync (default), `query`, `healthcheck`; SIGTERM handling |
| `src/audiobookshelf_rag_ingester/config.py` | Settings from environment variables |
| `src/audiobookshelf_rag_ingester/abs_client.py` | Read-only Audiobookshelf REST client with retries |
| `src/audiobookshelf_rag_ingester/models.py` | Normalizes Audiobookshelf JSON (books, podcasts, episodes, authors) |
| `src/audiobookshelf_rag_ingester/documents.py` | Builds the chunks that get embedded, with stable ids and content hashes |
| `src/audiobookshelf_rag_ingester/extractors.py` | Text from DRM-free EPUB and PDF files |
| `src/audiobookshelf_rag_ingester/embeddings.py` | Ollama (default) and SentenceTransformers embedders |
| `src/audiobookshelf_rag_ingester/qdrant_store.py` | Qdrant collection, payload indexes, upsert, delete and search |
| `src/audiobookshelf_rag_ingester/service.py` | The sync loop |
| `src/audiobookshelf_rag_ingester/query.py` | Search and result formatting for `query` |
| `src/audiobookshelf_rag_ingester/heartbeat.py` | Heartbeat file behind the container health check |
| `tests/` | pytest; full sync runs against an in-memory Qdrant and a fake Audiobookshelf |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | Short map of the code and its invariants |

## Technical details

### How it talks to Audiobookshelf

It uses the REST API with a Bearer token, not the database. Audiobookshelf's SQLite schema is internal and changes between releases. The API is stable, works when Audiobookshelf runs on another host, and respects the token user's library permissions. The ingester calls only these read-only endpoints:

| Endpoint | Used for |
|----------|----------|
| `GET /api/libraries` | Libraries and their media type (book or podcast) |
| `GET /api/libraries/:id/items?minified=1` | Paginated item list: cheap change detection |
| `GET /api/items/:id?expanded=1` | Full item (chapters, episodes, ebook file), only for items that changed |
| `GET /api/libraries/:id/authors` | Author names and bios |
| `GET /api/items/:id/ebook` | The item's ebook file, only when it changed |

### What gets indexed

Each Qdrant point has a `chunk_type`:

| `chunk_type` | One per | Contents |
|--------------|---------|----------|
| `item_metadata` | book or podcast | Title, authors, narrators, series, genres, tags, year, publisher, length, chapter count, ISBN/ASIN, start of the description |
| `description` | book or podcast with a long blurb | The rest of the description, chunked |
| `chapters` | audiobook | Chapter titles with start times and lengths (skipped for "Chapter 1, 2, 3…") |
| `episode` | podcast episode | Title, season and number, date, length, chapter titles, show notes |
| `ebook_content` | chunk of an attached ebook | Text of DRM-free EPUB/PDF files, with section titles |
| `author` | author in a book library | The author's books in the library, then their bio |
| `library_summary` | library | Counts, author/narrator/series/genre tallies, full catalog |

The embedded text (`context`) of each chunk starts with a header such as `Audiobook: Title: Subtitle | By: Author | Narrated by: Narrator | Series: Name #2`, so every retrieved chunk carries its source.

### Incremental sync and reindexing

1. **Item fingerprint.** Each item's `updatedAt`, `mtimeMs`, size, chapter count, episode count and ebook format (all in the cheap list call) are compared with the values stored in Qdrant. If nothing changed, the item isn't fetched at all.
2. **Content hash.** For a changed item, only chunks whose text or payload changed are embedded. The item's other points (a shorter description, a removed episode) are deleted.
3. **Ebook signature.** An ebook is downloaded again only when its file (inode, size, mtime) or the item's title, authors or series change.
4. **Deletions.** Items and libraries that are gone from Audiobookshelf, or excluded by `ABS_LIBRARIES`, are removed. A sync stopped part-way (SIGTERM) deletes nothing.

You don't need an external reindex job:

| Change | What happens |
|--------|--------------|
| Edit metadata, add episodes, replace an ebook | Picked up on the next sync |
| Change `EMBEDDING_MODEL`, `EMBEDDING_PROVIDER`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `ABS_PUBLIC_URL` or `INGEST_EBOOKS`/`_CHAPTERS`/`_EPISODES` | Every item is rebuilt on the next sync |
| A model with a different vector size | Startup fails. Set a new `QDRANT_COLLECTION` or delete the old one |
| Anything else | Run once with `FORCE_REINGEST=true` |

Qdrant maintains its own vector and payload indexes, so there's nothing to rebuild there.

### Payload fields

Every point has `source` (`"audiobookshelf"`), `chunk_type`, `library_id`, `library_name`, `media_type`, `url` (a link into the Audiobookshelf web UI), `text` (the raw chunk), `context` (what was embedded; use this for LLM grounding), `chunk_index`, `total_chunks`, `content_hash`, `doc_key` and `ingested_at`.

The other fields depend on the chunk type:

- **Item chunks:** `item_id`, `title`, `subtitle`, `authors`, `narrators`, `series`, `series_labels`, `genres`, `tags`, `published_year`, `publisher`, `language`, `isbn`, `asin`, `explicit`, `duration_seconds`, `duration`, `has_ebook`, `added_at`, `item_fingerprint`, `pipeline_signature`.
- **Episodes:** also `episode_id`, `episode_title`, `episode_season`, `episode_number`, `episode_published_at`.
- **Ebook chunks:** also `section_title`, `ebook_format`, `ebook_filename`, `ebook_signature`.
- **Authors:** `author_id`, `author_name`, `book_count`.
- **Summaries:** `summary_section` (`overview`, `authors`, `narrators`, `series`, `genres`, `catalog`).

These fields have keyword indexes for filtering: `source`, `chunk_type`, `library_id`, `media_type`, `item_id`, `episode_id`, `author_id`, `authors`, `narrators`, `series`, `genres`, `tags`, `language` and `published_year`. `episode_published_at` has a datetime index.

### Command line

```bash
audiobookshelf-rag-ingester                 # sync every POLL_INTERVAL_SECONDS
audiobookshelf-rag-ingester --once          # sync once; exit 2 if any item failed
audiobookshelf-rag-ingester query "cozy science fiction about a ship crew"
audiobookshelf-rag-ingester query --type episode --show-text "laser cutter safety"
audiobookshelf-rag-ingester healthcheck     # exit 0 while the sync loop is alive
audiobookshelf-rag-ingester --version
```

### Querying from your own code

```python
import httpx
from qdrant_client import QdrantClient

vector = httpx.post(
    "http://localhost:11434/api/embed",
    json={"model": "nomic-embed-text", "input": ["books narrated by Ray Porter"]},
).json()["embeddings"][0]
for hit in (
    QdrantClient("http://localhost:6333")
    .query_points("audiobookshelf", query=vector, limit=5)
    .points
):
    print(hit.score, hit.payload["title"], hit.payload["url"])
```

## Running it

Requires Docker, plus an Ollama server with the embedding model pulled (`ollama pull nomic-embed-text`). Configuration comes from environment variables kept in `.env` (see [docs/configuration.md](docs/configuration.md)). At minimum, set `ABS_URL`, `ABS_API_TOKEN` and `OLLAMA_URL`:

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
docker compose -f docker-compose.prod.yaml logs -f app
```

That runs the latest release from `ghcr.io/romkey/audiobookshelf-rag-ingester` alongside Qdrant (on `127.0.0.1:6333`). Set `APP_VERSION=X.Y.Z` to pin a release. To use an existing Qdrant, remove the `qdrant` service from the compose file and set `QDRANT_URL`. For an Ollama on the Docker host, use `OLLAMA_URL=http://host.docker.internal:11434`.

To search from the container:

```bash
docker compose -f docker-compose.prod.yaml exec app audiobookshelf-rag-ingester query "space opera"
```

### With pip

Needs Python 3.15 or newer. Install into a virtual environment from a release's wheel (or `pip install audiobookshelf-rag-ingester` if it's published on PyPI):

```bash
python3 -m venv ~/.venvs/audiobookshelf-rag-ingester
~/.venvs/audiobookshelf-rag-ingester/bin/pip install https://github.com/romkey/audiobookshelf-rag-ingester/releases/download/vX.Y.Z/audiobookshelf_rag_ingester-X.Y.Z-py3-none-any.whl
~/.venvs/audiobookshelf-rag-ingester/bin/audiobookshelf-rag-ingester --version
```

Set the environment variables yourself (see [docs/configuration.md](docs/configuration.md)); the app doesn't read `.env`. To embed in-process instead of with Ollama, install the `sentence-transformers` extra (it pulls in PyTorch) and set `EMBEDDING_PROVIDER=sentence_transformers`.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # run from the working tree, with Qdrant
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions CI tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run audiobookshelf-rag-ingester --once
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
audiobookshelf-rag-ingester --once
```

Either way, `uvx pre-commit install` (or `pip install pre-commit && pre-commit install`) runs the lint checks on every commit.

The tests need no services: the sync loop runs against Qdrant's in-memory mode and a fake Audiobookshelf.

## License

[MIT](LICENSE)
