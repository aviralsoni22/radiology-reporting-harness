"""
Local reimplementation of the Radiology Edit Score (RES).

This is a FAITHFUL PROXY of Natoe's scorer built from the public brief, not the
official implementation. Use it to optimize locally on held-out train rows. Small
gaps vs the real scorer are expected; a few points are underspecified in the brief
and are marked ASSUMPTION below. Verify each against train.csv in Phase 0.

RES_case = 0.65 * F + 0.35 * I   (lower is better; 0.0 == exact match)
  F = field-aware FINDINGS score
  I = weighted word edit over the full IMPRESSION block
"""

import re
import unicodedata

from parse import parse_report, normalize_eq  # noqa: F401  (normalize_eq re-exported)

# --------------------------------------------------------------------------
# Token classification lexicons.
# SEED values. Refine from train-token frequencies in Phase 0. The three weight
# classes come straight from the brief; only the word membership is tunable.
# --------------------------------------------------------------------------
NEGATION = {"no", "not", "without", "negative", "absent", "absence", "none",
            "non", "unremarkable", "denies", "free", "clear"}
LATERALITY = {"left", "right", "bilateral", "bilaterally", "unilateral"}
SEVERITY = {"mild", "mildly", "moderate", "moderately", "severe", "severely",
            "minimal", "minimally", "marked", "markedly", "trace", "small",
            "large", "tiny", "prominent", "slight", "slightly", "extensive",
            "subtle", "gross"}
ACUITY = {"acute", "chronic", "subacute", "old", "new", "recent", "healing",
          "healed", "interval", "stable"}
UNITS = {"mm", "cm"}
CRITICAL_WORDS = NEGATION | LATERALITY | SEVERITY | ACUITY | UNITS

FUNCTION = {"the", "a", "an", "and", "or", "of", "with", "to", "in", "on", "at",
            "for", "as", "is", "are", "was", "were", "be", "been", "this",
            "that", "these", "those", "there", "which", "from", "by"}

W_CRITICAL = 4.00
W_CONTENT = 2.00
W_FUNCTION = 0.25

# ASSUMPTION: keep intra-number decimals (3.5 stays 3.5, not 35). The brief says
# "punctuation is ignored", which would corrupt decimals; treating a decimal
# point as part of the measurement is the sensible reading. Toggle and re-check.
KEEP_DECIMALS = True


def token_weight(tok: str) -> float:
    if any(ch.isdigit() for ch in tok):
        return W_CRITICAL
    if tok in CRITICAL_WORDS:
        return W_CRITICAL
    if tok in FUNCTION:
        return W_FUNCTION
    return W_CONTENT


# --------------------------------------------------------------------------
# Normalization -> token list (per the brief's rules)
# --------------------------------------------------------------------------
_UNIT_SUBS = [
    (re.compile(r"\bmillimet(?:er|re)s?\b"), "mm"),
    (re.compile(r"\bcentimet(?:er|re)s?\b"), "cm"),
]
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:\d+[.)]|[-*\u2022])[ \t]+")
_LETTER_HYPHEN = re.compile(r"(?<=[a-z])-(?=[a-z])")
_TOKEN = re.compile(r"[+-]?\d+(?:\.\d+)?|[a-z]+" if KEEP_DECIMALS
                    else r"[+-]?\d+|[a-z]+")


def normalize(text) -> list:
    if not text:
        return []
    t = unicodedata.normalize("NFKC", str(text)).lower()
    for pat, rep in _UNIT_SUBS:
        t = pat.sub(rep, t)
    t = _LIST_MARKER.sub("", t)          # strip leading list markers
    t = _LETTER_HYPHEN.sub("", t)        # air-space -> airspace
    return _TOKEN.findall(t)             # signed decimals kept, punctuation dropped,
                                         # letter/number boundaries split


# --------------------------------------------------------------------------
# Weighted word-level Levenshtein
# --------------------------------------------------------------------------
def weighted_edit(ref_tokens, hyp_tokens) -> float:
    """Normalized weighted edit distance in [0, 1].

    ins(t) = del(t) = weight(t).
    ASSUMPTION: sub(a, b) = 0 if a == b else max(weight(a), weight(b)).
    Cost is divided by max(total ref weight, total hyp weight), capped at 1.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    rw = [token_weight(t) for t in ref_tokens]
    hw = [token_weight(t) for t in hyp_tokens]
    if n == 0 and m == 0:
        return 0.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + rw[i - 1]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + hw[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                sub = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + max(rw[i - 1], hw[j - 1])
            dele = dp[i - 1][j] + rw[i - 1]
            ins = dp[i][j - 1] + hw[j - 1]
            dp[i][j] = min(sub, dele, ins)
    denom = max(sum(rw), sum(hw))
    return min(1.0, dp[n][m] / denom) if denom else 0.0


def field_edit(ref_text, hyp_text) -> float:
    return weighted_edit(normalize(ref_text), normalize(hyp_text))


# --------------------------------------------------------------------------
# Field-aware FINDINGS score
# --------------------------------------------------------------------------
def findings_score(template_fields, ref_fields, hyp_fields) -> float:
    """F = sum(w_field * field_edit(ref, hyp)) / sum(w_field).

    w_field = 3 if the reference field differs from the template field, else 1.
    Missing expected field -> edit(ref, empty) (~1) at that field's weight.
    Unexpected hyp field (label not in reference) -> penalty at weight 1.
    """
    tmpl = {lbl: c for lbl, c in template_fields}
    hyp = {lbl: c for lbl, c in hyp_fields}
    ref_labels = {lbl for lbl, _ in ref_fields}
    num = den = 0.0

    for lbl, ref_c in ref_fields:
        tmpl_c = tmpl.get(lbl)
        changed = tmpl_c is None or normalize(tmpl_c) != normalize(ref_c)
        w = 3.0 if changed else 1.0
        hyp_c = hyp.get(lbl)
        fe = field_edit(ref_c, hyp_c if hyp_c is not None else "")
        num += w * fe
        den += w

    for lbl, hyp_c in hyp_fields:            # unexpected fields (documented penalty)
        if lbl not in ref_labels:
            num += 1.0 * weighted_edit([], normalize(hyp_c))
            den += 1.0

    return num / den if den else 0.0


def res_case(template_report, ref_report, hyp_report):
    """Return (RES_case, F, I) for one case. Needs the reference, so this is a
    LOCAL / validation-time metric over train rows."""
    t = parse_report(template_report)
    r = parse_report(ref_report)
    h = parse_report(hyp_report)
    F = findings_score(t["fields"], r["fields"], h["fields"])
    I = weighted_edit(normalize(r["impression"]), normalize(h["impression"]))
    return 0.65 * F + 0.35 * I, F, I


def res_mean(rows):
    """rows: iterable of (template, reference, hypothesis). Returns mean RES."""
    scores = [res_case(t, r, h)[0] for t, r, h in rows]
    return sum(scores) / len(scores) if scores else 0.0
