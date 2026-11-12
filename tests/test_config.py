import pytest

from audiobookshelf_rag_ingester.config import ConfigError, Settings

REQUIRED = {"ABS_URL": "http://abs.local:13378/", "ABS_API_TOKEN": "tok"}


def test_defaults() -> None:
    s = Settings.from_env(REQUIRED)
    assert s.abs_url == "http://abs.local:13378"
    assert s.abs_public_url == s.abs_url
    assert s.abs_libraries == ()
    assert s.embedding_provider == "ollama"
    assert s.chunk_size == 1200
    assert s.ebook_max_bytes == 100 * 1024 * 1024
    assert s.poll_interval_seconds == 3600
    assert s.qdrant_api_key is None


def test_everything_set() -> None:
    env = {
        **REQUIRED,
        "ABS_PUBLIC_URL": "https://books.example.com/",
        "ABS_LIBRARIES": "Audiobooks, , Podcasts",
        "INGEST_EBOOKS": "no",
        "EBOOK_MAX_MB": "5",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "QDRANT_DISTANCE": "DOT",
        "QDRANT_API_KEY": "  ",
        "ITEM_DELAY_SECONDS": "0.5",
        "LOG_LEVEL": "debug",
        "FORCE_REINGEST": "TRUE",
    }
    s = Settings.from_env(env)
    assert s.abs_public_url == "https://books.example.com"
    assert s.abs_libraries == ("Audiobooks", "Podcasts")
    assert s.ingest_ebooks is False
    assert s.ebook_max_bytes == 5 * 1024 * 1024
    assert s.embedding_provider == "sentence_transformers"
    assert s.qdrant_distance == "dot"
    assert s.qdrant_api_key is None
    assert s.item_delay_seconds == 0.5
    assert s.log_level == "DEBUG"
    assert s.force_reingest is True


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "ABS_URL is required"),
        ({"ABS_URL": "http://x"}, "ABS_API_TOKEN is required"),
        ({**REQUIRED, "ABS_URL": "abs.local"}, "ABS_URL must start with http"),
        ({**REQUIRED, "CHUNK_SIZE": "big"}, "CHUNK_SIZE must be a whole number"),
        ({**REQUIRED, "CHUNK_SIZE": "10"}, "CHUNK_SIZE must be at least 100"),
        ({**REQUIRED, "CHUNK_OVERLAP": "1200"}, "CHUNK_OVERLAP .* smaller"),
        ({**REQUIRED, "INGEST_EBOOKS": "maybe"}, "INGEST_EBOOKS must be true or false"),
        ({**REQUIRED, "ITEM_DELAY_SECONDS": "soon"}, "ITEM_DELAY_SECONDS must be a"),
        ({**REQUIRED, "ITEM_DELAY_SECONDS": "-1"}, "must not be negative"),
        ({**REQUIRED, "EMBEDDING_PROVIDER": "openai"}, "EMBEDDING_PROVIDER must be"),
        ({**REQUIRED, "LOG_LEVEL": "LOUD"}, "LOG_LEVEL 'LOUD'"),
    ],
)
def test_invalid(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)
