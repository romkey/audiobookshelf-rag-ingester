# Configuration

The app reads these environment variables at startup. Put them in `.env`
(copy `.env.example`); compose passes it to the container. An invalid
value stops the app with a message naming the variable.

## Audiobookshelf

| Variable | Default | Meaning |
|----------|---------|---------|
| `ABS_URL` | required | Base URL of the Audiobookshelf server, `http://` or `https://` |
| `ABS_API_TOKEN` | required | ABS API key (Settings → API Keys, ABS 2.26+) or a user's API token, sent as a Bearer token. The user needs to see the libraries to index; read access is enough |
| `ABS_PUBLIC_URL` | `ABS_URL` | Base for the `url` links stored with each chunk (`/item/<id>`, `/author/<id>`, `/library/<id>`) |
| `ABS_LIBRARIES` | all | Comma-separated library names or ids (case-insensitive). Points for libraries that aren't selected are **deleted** from the collection |
| `ABS_PAGE_SIZE` | `100` | Items per page when listing a library (≥ 1) |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Timeout for each Audiobookshelf request (≥ 1) |
| `ITEM_DELAY_SECONDS` | `0` | Pause after each item that is (re)ingested |

## What to index

| Variable | Default | Meaning |
|----------|---------|---------|
| `INGEST_EBOOKS` | `true` | Download and index the text of DRM-free EPUB and PDF files attached to items |
| `EBOOK_MAX_MB` | `100` | Skip ebook files larger than this (≥ 1) |
| `INGEST_CHAPTERS` | `true` | Index audiobook chapter lists (skipped when the titles are just "Chapter 1, 2, 3…") |
| `INGEST_EPISODES` | `true` | Index podcast episodes and show notes |
| `INGEST_AUTHORS` | `true` | Index an author document per author in book libraries |
| `INGEST_LIBRARY_SUMMARY` | `true` | Index per-library overview, tallies and catalog |

Booleans accept `true`/`false`, `yes`/`no`, `on`/`off`, `1`/`0`.

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint. The compose files set it to their `qdrant` service |
| `QDRANT_API_KEY` | none | Qdrant API key, if the server requires one |
| `QDRANT_COLLECTION` | `audiobookshelf` | Collection to write. It's created if missing |
| `QDRANT_DISTANCE` | `cosine` | `cosine`, `dot`, `euclid` or `manhattan`. Only used when the collection is created |

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`, or `sentence_transformers` (install the `sentence-transformers` extra; not in the image) |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model name for the provider. The vector size is detected at startup and must match an existing collection |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding request (≥ 1) |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Max tokens per text, including the prefix. Longer texts (at ~4 characters per token) are truncated. `0` disables truncation |
| `EMBEDDING_QUERY_PREFIX` | empty | Task prefix added to search queries (`query` command), e.g. `search_query:`. A space is added after it. See `.env.example` for common models |
| `EMBEDDING_DOCUMENT_PREFIX` | empty | Task prefix added to stored text, e.g. `search_document:`. Changing it changes every vector: run once with `FORCE_REINGEST=true`. Other apps querying Qdrant must add the same query prefix themselves |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API endpoint, or any OpenAI-compatible server (Ollama's `/v1`, LM Studio, vLLM…) |
| `OPENAI_API_KEY` | none | Optional bearer token. Needed for OpenAI itself, not for local servers without auth |
| `HF_CACHE_DIR` | none | Model cache for `sentence_transformers` |

## Chunking and scheduling

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHUNK_SIZE` | `1200` | Characters per chunk (≥ 100) |
| `CHUNK_OVERLAP` | `120` | Overlap between neighboring prose chunks. Must be smaller than `CHUNK_SIZE` |
| `POLL_INTERVAL_SECONDS` | `3600` | Seconds between syncs. `0` syncs once and exits (like `--once`) |
| `FORCE_REINGEST` | `false` | Re-embed every chunk on each sync, ignoring change detection. Set it for one run, then unset it |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

Changing `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `ABS_PUBLIC_URL`, `INGEST_EBOOKS`, `INGEST_CHAPTERS` or
`INGEST_EPISODES` makes the next sync rebuild every item automatically.

## Compose (not read by the app)

| Variable | Meaning |
|----------|---------|
| `APP_VERSION` | Image tag `docker-compose.prod.yaml` runs. Default `latest` |

## Publishing (not read by the app)

| Variable | Meaning |
|----------|---------|
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
