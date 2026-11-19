import pytest

from audiobookshelf_rag_ingester.config import ConfigError, Settings

REQUIRED = {
    "ABS_URL": "http://abs.local:13378/",
    "ABS_API_TOKEN": "tok",
    "DATABASE_URL": "postgresql://u:p@db/app",  # not-a-secret
}


def test_defaults() -> None:
    s = Settings.from_env(REQUIRED)
    assert s.abs_url == "http://abs.local:13378"
    assert s.abs_public_url is None
    assert s.link_base_url == s.abs_url
    assert s.abs_libraries == ()
    assert s.member_collection == "hs_member"
    assert s.catalog_collection == "hs_catalog"
    assert s.embedding_model == "nomic-ai/modernbert-embed-base"
    assert s.chunk_size == 1800
    assert s.chunk_overlap == 150
    assert s.vector_size == 768
    assert s.llm_model is None
    assert s.enrichment_enabled is True
    assert s.gateway_base_url == s.openai_base_url
    assert s.gateway_api_key is None


def test_everything_set() -> None:
    env = {
        **REQUIRED,
        "ABS_PUBLIC_URL": "https://books.example.com/",
        "ABS_LIBRARIES": "Audiobooks, , Podcasts",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_QUERY_PREFIX": "search_query: ",
        "QDRANT_DISTANCE": "DOT",
        "QDRANT_API_KEY": "  ",
        "LOG_LEVEL": "debug",
        "LLM_MODEL": "qwen",
        "LLM_BASE_URL": "http://gw/v1",
        "LLM_API_KEY": "k",
        "ENRICHMENT_ENABLED": "no",
        "VECTOR_SIZE": "256",
        "EMBEDDING_DIMENSIONS": "256",
        "LIBRARY_ACCESS_INFO": "Ask in #library",
        "HF_HOME": "/hf",
    }
    s = Settings.from_env(env)
    assert s.link_base_url == "https://books.example.com"
    assert s.abs_libraries == ("Audiobooks", "Podcasts")
    assert s.embedding_provider == "sentence-transformers"
    assert s.embedding_query_prefix == "search_query: "
    assert s.distance == "dot"
    assert s.qdrant_api_key is None
    assert s.log_level == "DEBUG"
    assert s.gateway_base_url == "http://gw/v1"
    assert s.gateway_api_key == "k"
    assert s.enrichment_enabled is False
    assert s.embedding_dimensions == 256
    assert s.library_access_info == "Ask in #library"
    assert s.hf_cache_dir == "/hf"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "ABS_URL"),
        ({"ABS_URL": "http://x", "DATABASE_URL": "postgresql://x"}, "ABS_API_TOKEN"),
        ({"ABS_URL": "http://x", "ABS_API_TOKEN": "t"}, "DATABASE_URL"),
        ({**REQUIRED, "ABS_URL": "abs.local"}, "ABS_URL must start with http"),
        ({**REQUIRED, "CHUNK_SIZE": "big"}, "CHUNK_SIZE must be a whole number"),
        ({**REQUIRED, "CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, "CHUNK_OVERLAP"),
        ({**REQUIRED, "ABS_PAGE_SIZE": "0"}, "at least 1"),
        ({**REQUIRED, "EMBEDDING_PROVIDER": "nope"}, "EMBEDDING_PROVIDER"),
        ({**REQUIRED, "ENRICHMENT_ENABLED": "maybe"}, "true or false"),
        ({**REQUIRED, "LOG_LEVEL": "loud"}, "LOG_LEVEL"),
        ({**REQUIRED, "EMBEDDING_DIMENSIONS": "100"}, "EMBEDDING_DIMENSIONS"),
        ({**REQUIRED, "QDRANT_CATALOG_COLLECTION": "hs_member"}, "must be different"),
        ({**REQUIRED, "QDRANT_MEMBER_COLLECTION": "hs_public"}, "member tier only"),
    ],
)
def test_invalid(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)
