import pytest

from audiobookshelf_rag_ingester.sanitize import REDACTED, find_secrets, redact, shannon_entropy

# Fake secrets in the shapes the scanner looks for. gitleaks:allow
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
# Split so repository secret scanners don't flag this fake key block.
PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n-----END RSA " + "PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"key {AWS} here", "aws-access-key"),
        (f"token {GITHUB}", "github-token"),
        (PEM, "private-key"),
        ("password: correcthorse", "assigned-secret"),
        ("https://user:pa55word@host.example/x", "url-credentials"),
        ("blob Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MEFCQ0RFRkdISUpL done", "high-entropy"),
    ],
)
def test_finds_secrets(text: str, rule: str) -> None:
    assert rule in {f.rule for f in find_secrets(text)}


def test_redact_replaces_spans_and_keeps_the_rest() -> None:
    result = redact(f"Before {AWS} middle password=hunter2hunter2 after")
    assert result.text == f"Before {REDACTED} middle password={REDACTED} after"
    assert result.rules == ["assigned-secret", "aws-access-key"]


def test_clean_text_is_untouched() -> None:
    text = (
        "ISBN 9780306406157. See https://example.com/books/the-art-of-electronics-third-edition "
        "for the errata."
    )
    result = redact(text)
    assert result.text == text
    assert result.findings == []


def test_overlapping_findings_merge() -> None:
    text = f"api_key={GITHUB}"
    findings = find_secrets(text)
    assert len(findings) == 1
    assert redact(text).text == f"api_key={REDACTED}"


def test_entropy() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == 2.0
