"""Deterministic secret scanner, run on all text before it is staged.

The rules follow the shapes gitleaks and detect-secrets look for (provider
token prefixes, private key blocks, ``password = ...`` assignments) plus a
Shannon-entropy check for long random-looking tokens. A hit redacts the
span; the caller flags the document and writes a quarantine entry.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

REDACTED = "[REDACTED]"

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{40,})\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("stripe-key", re.compile(r"\b[rs]k_live_[0-9A-Za-z]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    (
        "url-credentials",
        re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s:/@]+@[^\s/]+", re.IGNORECASE),
    ),
    (
        "assigned-secret",
        re.compile(
            r"""(?ix)
            \b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|
                 client[_-]?secret|wifi[_-]?password|psk)
            \s*[:=]\s*
            ["']?(?P<value>[^\s"']{6,})["']?
            """
        ),
    ),
)

# Long tokens of letters, digits and base64/hex punctuation.
_TOKEN = re.compile(r"[A-Za-z0-9+/_\-]{32,}={0,2}")
_ENTROPY_THRESHOLD = 4.3


@dataclass(frozen=True)
class Finding:
    rule: str
    start: int
    end: int


@dataclass(frozen=True)
class ScanResult:
    text: str
    findings: list[Finding]

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.findings})


def shannon_entropy(token: str) -> float:
    if not token:
        return 0.0
    counts = Counter(token)
    length = len(token)
    return -sum(n / length * math.log2(n / length) for n in counts.values())


def _looks_random(token: str) -> bool:
    # Letters alone (slugs, words joined by hyphens) or digits alone aren't secrets.
    if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
        return False
    return shannon_entropy(token) >= _ENTROPY_THRESHOLD


def find_secrets(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for rule, pattern in _RULES:
        for match in pattern.finditer(text):
            if "value" in pattern.groupindex:
                findings.append(Finding(rule, match.start("value"), match.end("value")))
            else:
                findings.append(Finding(rule, match.start(), match.end()))
    for match in _TOKEN.finditer(text):
        if _looks_random(match.group()):
            findings.append(Finding("high-entropy", match.start(), match.end()))
    return _merge(findings)


def _merge(findings: list[Finding]) -> list[Finding]:
    merged: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (f.start, -f.end)):
        if merged and finding.start < merged[-1].end:
            last = merged[-1]
            merged[-1] = Finding(last.rule, last.start, max(last.end, finding.end))
        else:
            merged.append(finding)
    return merged


def redact(text: str) -> ScanResult:
    """Replace every secret-looking span with ``[REDACTED]``."""
    findings = find_secrets(text)
    if not findings:
        return ScanResult(text, [])
    parts: list[str] = []
    position = 0
    for finding in findings:
        parts.append(text[position : finding.start])
        parts.append(REDACTED)
        position = finding.end
    parts.append(text[position:])
    return ScanResult("".join(parts), findings)
