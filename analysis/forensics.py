"""
Phase 0 forensics. Run this the moment you have train.csv. It confirms or breaks
the strategy the whole pipeline rests on.

    python analysis/forensics.py --train data/train.csv

Answers:
  1. Are unchanged reference fields byte-identical to the template field?
     (If yes -> verbatim copy of untouched fields is a guaranteed 0. This is Lever 1.)
  2. How many fields change per case, and which ones?
  3. What do the field schemas look like per modality / body_part?
  4. What does the edit look like on changed fields? (dumps triples to inspect)
  5. IMPRESSION length + shape.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from src.parse import parse_report, normalize_eq          # noqa: E402
from src.scorer import normalize                          # noqa: E402


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--dump", default="analysis/changed_fields.tsv",
                    help="where to write changed-field triples for inspection")
    args = ap.parse_args()

    rows = load(args.train)
    print(f"loaded {len(rows)} rows; columns: {list(rows[0].keys())}\n")

    n_fields_changed = []
    changed_label_counter = Counter()
    unchanged_identical = 0
    unchanged_total = 0
    unchanged_norm_equal_raw_differ = 0
    schema_by_modality = defaultdict(Counter)
    impression_lens = []
    dump = []

    for r in rows:
        tmpl = parse_report(r["template_content"])
        ref = parse_report(r["report"])
        tmpl_map = {l: c for l, c in tmpl["fields"]}
        modality = r.get("modality", "?")

        for l, _ in ref["fields"]:
            schema_by_modality[modality][l] += 1

        changed_here = 0
        for label, ref_c in ref["fields"]:
            tmpl_c = tmpl_map.get(label)
            if tmpl_c is None:
                changed_here += 1
                changed_label_counter[label] += 1
                dump.append((r.get("case_id", ""), modality, label,
                             "<no template field>", ref_c, r["dictation"]))
                continue
            same_norm = normalize(tmpl_c) == normalize(ref_c)
            if same_norm:
                unchanged_total += 1
                if normalize_eq(tmpl_c, ref_c):
                    unchanged_identical += 1
                else:
                    unchanged_norm_equal_raw_differ += 1
            else:
                changed_here += 1
                changed_label_counter[label] += 1
                dump.append((r.get("case_id", ""), modality, label,
                             tmpl_c, ref_c, r["dictation"]))
        n_fields_changed.append(changed_here)
        impression_lens.append(len(normalize(ref["impression"])))

    # ---- report ----
    print("=" * 60)
    print("1. UNCHANGED FIELDS (Lever 1 check)")
    if unchanged_total:
        pct = 100 * unchanged_identical / unchanged_total
        print(f"   unchanged fields (by normalized text): {unchanged_total}")
        print(f"   of those, byte-identical to template : {unchanged_identical} "
              f"({pct:.1f}%)")
        print(f"   normalized-equal but raw-differ      : "
              f"{unchanged_norm_equal_raw_differ}")
        print("   -> if ~100% identical, COPY untouched fields verbatim.")
    print()
    print("2. FIELDS CHANGED PER CASE")
    if n_fields_changed:
        avg = sum(n_fields_changed) / len(n_fields_changed)
        print(f"   avg {avg:.2f}, min {min(n_fields_changed)}, "
              f"max {max(n_fields_changed)}")
        print(f"   most-changed labels: {changed_label_counter.most_common(12)}")
    print()
    print("3. FIELD SCHEMA BY MODALITY (top labels)")
    for mod, counter in schema_by_modality.items():
        print(f"   {mod}: {[l for l, _ in counter.most_common(10)]}")
    print()
    print("4. IMPRESSION LENGTH (tokens)")
    if impression_lens:
        impression_lens.sort()
        mid = impression_lens[len(impression_lens) // 2]
        print(f"   median {mid}, min {impression_lens[0]}, "
              f"max {impression_lens[-1]}")
    print()

    with open(args.dump, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["case_id", "modality", "label", "template_field",
                    "reference_field", "dictation"])
        w.writerows(dump)
    print(f"5. wrote {len(dump)} changed-field triples -> {args.dump}")
    print("   READ THIS FILE. It is your phrasebook for how the reference")
    print("   authors turn a dictated finding into report wording.")


if __name__ == "__main__":
    main()
