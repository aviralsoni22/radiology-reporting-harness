"""
Validation + repair. Protects the human review (faithfulness, routing, no
fabrication) and guards RES against structural mistakes.

check(report, template, dictation) -> list[str] of issues. Empty list == clean.

Use it two ways:
  - as a gate in the runner (log/flag failures),
  - to drive a repair prompt (re-ask the model to fix flagged fields).
Any fix must be a pipeline/prompt change that REGENERATES the report. Never hand
edit a specific test case; that violates the rules.
"""
import re

from parse import parse_report
from scorer import normalize

_NUM = re.compile(r"\d")
LATERALITY = {"left", "right", "bilateral", "bilaterally", "unilateral"}


def _numbers(tokens):
    return {t for t in tokens if _NUM.search(t)}


def check(report, template, dictation):
    issues = []
    rep = parse_report(report)
    tmpl = parse_report(template)

    # 1. structure: two sections present
    if not rep["fields"]:
        issues.append("no FINDINGS fields parsed")
    if not rep["impression"]:
        issues.append("empty IMPRESSION")

    # 2. label set + order match the template exactly
    tmpl_labels = [l for l, _ in tmpl["fields"]]
    rep_labels = [l for l, _ in rep["fields"]]
    if rep_labels != tmpl_labels:
        issues.append(f"label set/order drift: {rep_labels} != {tmpl_labels}")

    # 3. anti-hallucination: numbers and laterality in output must be supported
    src = set(normalize(dictation)) | set(normalize(template))
    out = set(normalize(report))
    for n in _numbers(out) - _numbers(src):
        issues.append(f"unsupported number in output: {n}")
    for side in (out & LATERALITY) - (src & LATERALITY):
        issues.append(f"unsupported laterality in output: {side}")

    # 4. untouched fields must be verbatim (they should never have changed)
    tmpl_map = {l: c for l, c in tmpl["fields"]}
    dict_tok = set(normalize(dictation))
    for label, content in rep["fields"]:
        base = tmpl_map.get(label)
        if base is None:
            continue
        anat = (set(normalize(label)) | set(normalize(base)))
        looks_touched = bool(dict_tok & anat)
        if not looks_touched and normalize(content) != normalize(base):
            issues.append(f"untouched field '{label}' was modified")

    return issues
