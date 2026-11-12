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
  `GET /api/items/:id?expanded=1` has chapters, episodes and the ebook file.
- Keep `documents.py` pure (no I/O); it is where chunk text and payloads are
  decided. Changing text or payload layout means bumping
  `DOCUMENT_SCHEMA_VERSION` in `service.py` so existing collections rebuild.
- Never delete points in a cycle that didn't finish (`stop_event`), or for a
  library whose item listing failed.
