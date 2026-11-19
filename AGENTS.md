# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- Audiobookshelf API shapes come from its server source
  (`server/models/LibraryItem.js`, `Book.js`, `Podcast.js`,
  `PodcastEpisode.js`, `controllers/LibraryController.js`); check there
  rather than guessing. List endpoints return minified items; only
  `GET /api/items/:id?expanded=1` has chapters and episodes.
- This is one source of the PDX Hackerspace RAG pipeline (plans 00 and 03)
  and follows the ebook source, calibre-web2rag. The modules and SQL listed
  under "Shared with the ebook source" in docs/overview.md must stay
  identical in both repos: a change to one is a change to both.
- Keep `documents.py` pure (no I/O); it decides chunk text and payloads.
- Never delete anything in a run that was stopped (`stop`), for a library
  whose listing failed, or past the safety valve.
- Tests need PostgreSQL (`DATABASE_URL`); `docker-compose.test.yaml` and CI
  provide one.
