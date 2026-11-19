from audiobookshelf_rag_ingester.sparse import sparse_vector, tokenize


def test_tokenize_drops_stopwords_and_single_letters() -> None:
    assert tokenize("The Antenna, a Yagi & THE dipole!") == ["antenna", "yagi", "dipole"]


def test_sparse_vector_is_deterministic_and_saturates() -> None:
    once = sparse_vector("antenna")
    thrice = sparse_vector("antenna antenna antenna")
    assert once.indices == thrice.indices == sparse_vector("Antenna").indices
    assert once.values == [1.0]
    assert 1.0 < thrice.values[0] < 2.2
    assert sparse_vector("").indices == []
