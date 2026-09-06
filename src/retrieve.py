"""
Retrieval of train exemplars to steer phrasing toward the reference house style.
This is Lever 2/3. Token-overlap ranking keeps it dependency-free for v1; swap in
embeddings later if it helps local RES.

  index = Retriever(train_rows); index.for_field(row, label); index.for_impression(row)
"""
from collections import defaultdict

from parse import parse_report
from scorer import normalize


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Retriever:
    def __init__(self, train_rows, k=3):
        self.k = k
        self.by_field = defaultdict(list)   # label -> list of exemplar dicts
        self.impressions = []               # list of exemplar dicts
        for r in train_rows:
            tmpl = {l: c for l, c in parse_report(r["template_content"])["fields"]}
            ref = parse_report(r["report"])
            dict_tok = set(normalize(r["dictation"]))
            for label, ref_c in ref["fields"]:
                tmpl_c = tmpl.get(label)
                if tmpl_c is None or normalize(tmpl_c) != normalize(ref_c):
                    self.by_field[label].append({
                        "label": label,
                        "template": tmpl_c or "",
                        "reference": ref_c,
                        "dictation": r["dictation"],
                        "modality": r.get("modality", ""),
                        "_dtok": dict_tok,
                    })
            self.impressions.append({
                "template": "",
                "reference": ref["impression"],
                "dictation": r["dictation"],
                "modality": r.get("modality", ""),
                "_dtok": dict_tok,
            })

    def _rank(self, pool, query_tokens, modality):
        scored = []
        for ex in pool:
            s = _jaccard(query_tokens, ex["_dtok"])
            if modality and ex.get("modality") == modality:
                s += 0.1                      # small same-modality boost
            scored.append((s, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:self.k]]

    def for_field(self, row, label):
        pool = self.by_field.get(label, [])
        q = set(normalize(row["dictation"]))
        return self._rank(pool, q, row.get("modality", ""))

    def for_impression(self, row):
        q = set(normalize(row["dictation"]))
        return self._rank(self.impressions, q, row.get("modality", ""))
