"""Text extraction for ebook files attached to Audiobookshelf items.

Only DRM-free EPUB and PDF files are read. Anything else (or anything that
fails to parse) yields no sections and is logged and skipped.
"""

import io
import logging
import warnings
from dataclasses import dataclass
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Font obfuscation is not DRM; these algorithms may legitimately appear in
# META-INF/encryption.xml of an otherwise readable EPUB.
_FONT_OBFUSCATION_ALGORITHMS = {
    "http://www.idpf.org/2008/embedding",
    "http://ns.adobe.com/pdf/enc#RC",
}


@dataclass(frozen=True)
class TextSection:
    title: str | None
    text: str


def extract_sections(data: bytes, fmt: str | None) -> list[TextSection]:
    fmt = (fmt or "").lower().lstrip(".")
    try:
        if fmt == "epub":
            return extract_epub_sections(data)
        if fmt == "pdf":
            return extract_pdf_sections(data)
    except Exception:
        logger.exception("Failed to extract text from %s ebook", fmt)
        return []
    logger.info("Ebook format %r is not supported for text extraction", fmt)
    return []


def epub_is_drm_free(data: bytes) -> bool:
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            names = {n.lower(): n for n in archive.namelist()}
            if "meta-inf/rights.xml" in names:
                return False
            enc_name = names.get("meta-inf/encryption.xml")
            if not enc_name:
                return True
            xml = archive.read(enc_name)
    except BadZipFile:
        return False
    soup = BeautifulSoup(xml, "html.parser")
    algorithms = {
        str(m.get("algorithm") or "").strip()
        for name in ("enc:encryptionmethod", "encryptionmethod")
        for m in soup.find_all(name)
    }
    algorithms.discard("")
    return algorithms.issubset(_FONT_OBFUSCATION_ALGORITHMS)


def extract_epub_sections(data: bytes) -> list[TextSection]:
    if not epub_is_drm_free(data):
        logger.info("EPUB appears to be DRM-protected; skipping")
        return []

    from ebooklib import ITEM_DOCUMENT, epub

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ebooklib warns about deprecated options
        book = epub.read_epub(io.BytesIO(data), options={"ignore_ncx": True})

    # Prefer spine (reading) order; fall back to manifest order.
    ordered = []
    for idref, _linear in book.spine or []:
        item = book.get_item_with_id(idref)
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            ordered.append(item)
    if not ordered:
        ordered = [i for i in book.get_items() if i.get_type() == ITEM_DOCUMENT]

    sections: list[TextSection] = []
    for item in ordered:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        title = None
        for tag in ("h1", "h2", "h3"):
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(" ", strip=True) or None
                break
        text = soup.get_text(" ", strip=True)
        if text:
            sections.append(TextSection(title=title, text=text))
    return sections


def extract_pdf_sections(data: bytes) -> list[TextSection]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        logger.info("PDF is encrypted; skipping")
        return []
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        return []
    return [TextSection(title=None, text=text)]
