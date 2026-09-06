"""
Parse a report (template or reference or hypothesis) into structured fields.

A report looks like:

    FINDINGS:
    LABEL A: content ...
    LABEL B: content ...

    IMPRESSION:
    summary ...

parse_report(text) -> {"fields": [(label, content), ...], "impression": str}

Field order is preserved. Alignment during scoring is by label, but keeping the
order lets us reassemble output in template order.
"""

import re

# A field label: start of line, uppercase-ish run, ending in a colon.
# Matches SUPPORT DEVICES:, CARDIOMEDIASTINAL SILHOUETTE:, OTHER FINDINGS:, BONES:
_LABEL = re.compile(r"(?m)^[ \t]*([A-Z][A-Z0-9 /&()\-]*?):[ \t]*")
_FINDINGS_HDR = re.compile(r"(?im)^\s*FINDINGS\s*:\s*")
_IMPRESSION_HDR = re.compile(r"(?im)^\s*IMPRESSION\s*:\s*")


def split_sections(report: str):
    """Return (findings_text, impression_text). Robust to a missing header."""
    if not report:
        return "", ""
    m = _IMPRESSION_HDR.search(report)
    if m:
        head, impression = report[:m.start()], report[m.end():]
    else:
        head, impression = report, ""
    head = _FINDINGS_HDR.sub("", head, count=1)
    return head.strip(), impression.strip()


def parse_fields(findings_text: str):
    """Ordered [(label, content)] from a FINDINGS block."""
    if not findings_text:
        return []
    matches = list(_LABEL.finditer(findings_text))
    if not matches:
        return [("", findings_text.strip())]   # unlabelled content
    fields = []
    for i, mt in enumerate(matches):
        label = mt.group(1).strip()
        start = mt.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(findings_text)
        fields.append((label, findings_text[start:end].strip()))
    return fields


def parse_report(report: str):
    findings_text, impression = split_sections(report)
    return {"fields": parse_fields(findings_text), "impression": impression}


def render_report(fields, impression: str) -> str:
    """Reassemble in template order. Enforces the two-section structure."""
    lines = ["FINDINGS:"]
    for label, content in fields:
        lines.append(f"{label}: {content}" if label else content)
    lines.append("")
    lines.append("IMPRESSION:")
    lines.append(impression.strip())
    return "\n".join(lines)


def normalize_eq(a: str, b: str) -> bool:
    """True if two field contents are equal after light whitespace collapse.
    (Byte-level check; the scorer's own normalize() is stricter.)"""
    collapse = lambda s: re.sub(r"\s+", " ", (s or "").strip())
    return collapse(a) == collapse(b)
