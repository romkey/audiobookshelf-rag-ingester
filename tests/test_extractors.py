import io
import zipfile

from audiobookshelf_rag_ingester.extractors import epub_is_drm_free, extract_sections
from tests.fakes import build_epub_bytes


def _with_encryption_xml(data: bytes, algorithm: str) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info.filename))
        dst.writestr(
            "META-INF/encryption.xml",
            '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            'xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData>'
            f'<enc:EncryptionMethod Algorithm="{algorithm}"/>'
            "</enc:EncryptedData></encryption>",
        )
    return out.getvalue()


def test_extract_epub_sections_in_spine_order() -> None:
    data = build_epub_bytes(
        [("Chapter One", "It was a dark night."), ("Chapter Two", "Dawn.")]
    )
    sections = extract_sections(data, "epub")
    titles = [s.title for s in sections if s.title]
    assert titles[-2:] == ["Chapter One", "Chapter Two"]
    assert any("dark night" in s.text for s in sections)


def test_font_obfuscation_is_not_drm() -> None:
    data = build_epub_bytes([("One", "Text.")])
    assert epub_is_drm_free(
        _with_encryption_xml(data, "http://www.idpf.org/2008/embedding")
    )


def test_real_encryption_is_drm() -> None:
    data = build_epub_bytes([("One", "Text.")])
    drm = _with_encryption_xml(data, "http://www.w3.org/2001/04/xmlenc#aes128-cbc")
    assert not epub_is_drm_free(drm)
    assert extract_sections(drm, "epub") == []


def test_unsupported_and_garbage() -> None:
    assert extract_sections(b"whatever", "mobi") == []
    assert extract_sections(b"not a zip", "epub") == []
    assert extract_sections(b"not a pdf", "pdf") == []
