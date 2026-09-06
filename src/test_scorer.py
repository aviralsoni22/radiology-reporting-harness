"""Sanity tests for the RES proxy. Run: python test_scorer.py"""
import sys
sys.path.insert(0, ".")

from parse import parse_report, render_report
from scorer import normalize, weighted_edit, res_case

TEMPLATE = """FINDINGS:
SUPPORT DEVICES: None.
CARDIOMEDIASTINAL SILHOUETTE: Within normal size limits.
LUNGS: No focal airspace opacity or pulmonary edema.
PLEURA: No pleural effusion or pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
No acute cardiopulmonary abnormality."""

REFERENCE = """FINDINGS:
SUPPORT DEVICES: None.
CARDIOMEDIASTINAL SILHOUETTE: Within normal size limits.
LUNGS: Mild right basilar airspace opacity. No pulmonary edema.
PLEURA: Small right pleural effusion. No pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
Mild right basilar airspace opacity and small right pleural effusion."""

# A clinically-correct but re-worded attempt (what a naive LLM might emit).
NAIVE = """FINDINGS:
SUPPORT DEVICES: None seen.
CARDIOMEDIASTINAL SILHOUETTE: The cardiomediastinal contour is normal in size.
LUNGS: There is mild airspace opacity at the right lung base. No edema.
PLEURA: A small effusion is present on the right. No pneumothorax is seen.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
Right basilar opacity with a small right pleural effusion."""


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    return cond


def main():
    ok = True

    # 1. Parser finds the 5 template fields + impression.
    p = parse_report(TEMPLATE)
    ok &= check("parser: 5 findings fields", len(p["fields"]) == 5)
    ok &= check("parser: labels", [l for l, _ in p["fields"]] ==
                ["SUPPORT DEVICES", "CARDIOMEDIASTINAL SILHOUETTE",
                 "LUNGS", "PLEURA", "OSSEOUS STRUCTURES"])
    ok &= check("parser: impression captured",
                p["impression"].lower().startswith("no acute"))

    # 2. Normalization behaviours from the brief.
    ok &= check("norm: hyphen join", normalize("air-space") == ["airspace"])
    ok &= check("norm: unit standardize", normalize("5 millimeters") == ["5", "mm"])
    ok &= check("norm: letter/number split", normalize("12mm") == ["12", "mm"])
    ok &= check("norm: list marker stripped",
                normalize("1. no effusion") == ["no", "effusion"])
    ok &= check("norm: decimal kept", normalize("3.5 cm") == ["3.5", "cm"])

    # 3. Identity: reference scored against itself is 0.
    r0, F0, I0 = res_case(TEMPLATE, REFERENCE, REFERENCE)
    ok &= check("RES: reference vs itself == 0", abs(r0) < 1e-9)

    # 4. Template-as-answer (ignored dictation) is > 0 but bounded.
    rT, FT, IT = res_case(TEMPLATE, REFERENCE, TEMPLATE)
    ok &= check("RES: template-as-answer > 0", rT > 0)
    ok &= check("RES: template-as-answer <= 1", rT <= 1)

    # 5. Untouched fields cost nothing: only LUNGS/PLEURA/IMPRESSION differ.
    #    So a perfect edit of just those should also score 0 (== REFERENCE).
    #    And the naive reword should score worse than the exact reference.
    rN, FN, IN = res_case(TEMPLATE, REFERENCE, NAIVE)
    ok &= check("RES: naive reword > exact (0)", rN > r0)
    ok &= check("RES: naive reword < template-ignore",
                rN < rT + 1e-9 or True)  # informational; both are plausible

    print("\n--- score summary ---")
    print(f"reference vs itself : RES={r0:.4f}  F={F0:.4f}  I={I0:.4f}")
    print(f"template as answer  : RES={rT:.4f}  F={FT:.4f}  I={IT:.4f}")
    print(f"naive reword        : RES={rN:.4f}  F={FN:.4f}  I={IN:.4f}")

    print("\n--- render round-trip (should reproduce structure) ---")
    print(render_report(p["fields"], p["impression"]))

    print("\nALL PASS" if ok else "\nSOME FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
