# Runbook: build the radiology reporting pipeline, step by step

This is the executable version of the plan. The keystone pieces are already
written and tested (`src/scorer.py`, `src/parse.py`, plus the pipeline, retriever,
and validator). Follow the steps in order. Each step says what to do, what to run,
and what to check before moving on.

## What is in the box

```
harness/
  requirements.txt
  data/                 <- put train.csv, test.csv, sample_submission.csv here
  src/
    scorer.py           BUILT + TESTED. RES proxy (normalize, weighted edit, F, I).
    parse.py            BUILT + TESTED. report -> fields + impression, and render back.
    test_scorer.py      BUILT. run to confirm the scorer on the brief's example.
    pipeline.py         BUILT (wiring tested). copy-untouched + LLM-edit-touched + runner.
    retrieve.py         BUILT. train exemplar retrieval (field + impression level).
    validate.py         BUILT. structure + anti-hallucination + untouched-field checks.
  analysis/
    forensics.py        BUILT. Phase 0: run against train.csv to validate the strategy.
```

Two things are deliberately left for you: (a) plug your API key into the one
generation step, and (b) tune the router/prompts once forensics tells you the
real patterns. Everything else runs with no key.

---

## Step 0. Setup (10 min)

```bash
cd harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-...' > .env    # never commit it
```

Put the three CSVs in `data/`. Confirm the scorer is healthy:

```bash
cd src && python test_scorer.py && cd ..
```

You should see `ALL PASS` and a score summary where the exact reference is
`RES=0.0000` and the naive reword scores worse on FINDINGS than it should. That
mismatch is the whole strategy in one number.

---

## Step 1. Phase 0 forensics (2-3 h). Do this before touching prompts.

```bash
python analysis/forensics.py --train data/train.csv
```

Read the printout and answer these. The pipeline design depends on the answers:

- **Are unchanged reference fields byte-identical to the template?** If it reports
  ~100% identical, Lever 1 is confirmed: copy untouched fields verbatim for
  guaranteed zeros. If not, look at the "normalized-equal but raw-differ" count
  and decide whether to copy template text or reference-normalized text.
- **How many fields change per case, and which labels?** This sizes the LLM work
  and tells you which fields matter.
- **Do field schemas differ by modality?** If XRAY, CT, MRI, USG use different
  label sets, plan per-modality handling.
- **Open `analysis/changed_fields.tsv`.** This is your phrasebook: every
  `(template_field, reference_field, dictation)` triple. Skim 30-40 of them and
  write down the recurring edit patterns (negation split, vocabulary reuse,
  measurement placement, severity/laterality wording). These patterns become
  your prompt instructions and, where reliable, deterministic rules.

Then refine two things in code from what you saw:
- the token lexicons in `src/scorer.py` (NEGATION/LATERALITY/SEVERITY/ACUITY),
- the routing signal in `src/pipeline.py:touched_labels` (replace token overlap
  with the real finding-to-label mapping if overlap is noisy).

---

## Step 2. Build the local validation harness (30 min)

Split train into a fit set (for retrieval/few-shot) and a held-out set (never seen
by retrieval) so local RES is honest. Create `analysis/evaluate.py`:

```python
import csv, random, sys
sys.path.insert(0, "src")
from pipeline import build_report, make_client
from retrieve import Retriever
from scorer import res_case

rows = list(csv.DictReader(open("data/train.csv", encoding="utf-8")))
random.seed(0); random.shuffle(rows)
holdout = rows[:120]           # never indexed
fit     = rows[120:]           # exemplar pool

R = Retriever(fit, k=3)
client = make_client("opus")            # any MODELS key, a slug, or "echo"

scored = []
for i, row in enumerate(holdout, 1):
    hyp = build_report(row, client, retriever=R)
    res, F, I = res_case(row["template_content"], row["report"], hyp)
    scored.append(res)
    if i % 20 == 0: print(i, "mean so far", sum(scored)/len(scored))
print("LOCAL RES:", sum(scored)/len(scored))
```

This is your optimization instrument. Every change from here is judged by this
number, not by guessing.

Tip while iterating: use `make_client("echo")` (in `pipeline.py`)
to measure the **copy-only floor** with zero API cost. That floor tells you how
much the LLM editing is actually adding.

---

## Step 3. First end-to-end run + first submission (1 h)

Generate the real test reports and submit once to calibrate local vs public.

```python
# analysis/generate.py
import csv, sys
sys.path.insert(0, "src")
from pipeline import build_report, make_client, run, write_submission
from retrieve import Retriever

train = list(csv.DictReader(open("data/train.csv", encoding="utf-8")))
test  = list(csv.DictReader(open("data/test.csv", encoding="utf-8")))
R = Retriever(train, k=3)
client = make_client("opus")
results = run(test, client, retriever=R)
write_submission(results, "submission.csv")
```

```bash
python analysis/generate.py
```

Upload `submission.csv` on Kaggle. Note the public RES next to your local RES. If
they move together, trust local from now on. If they diverge a lot, re-examine the
scorer assumptions flagged in `scorer.py` (substitution cost, decimals) against
train.

---

## Step 4. Optimization loop (4-6 h). This is where you win or lose.

Change one thing, run `evaluate.py`, keep it only if local RES drops. Order the
knobs by expected payoff:

1. **Lock untouched-field copying.** Confirm via `validate.py` that no untouched
   field is ever modified. If forensics said fields are byte-identical, make the
   copy exact. This is the cheapest RES you will ever buy.
2. **Router precision.** Wrong routing is double-penalized (missing from the right
   field, extra under the wrong one). Tighten `touched_labels` until it matches
   the dictation-to-field mapping you saw in forensics.
3. **Field-edit prompt.** Push "change only what the finding requires, keep every
   other word verbatim, reuse existing vocabulary." Add 3-5 retrieved exemplars.
   Re-measure.
4. **Deterministic rules** for the most regular patterns (e.g. combined-negation
   split). Replacing the LLM with a rule where the pattern is reliable removes
   variance and usually lowers RES on those fields.
5. **IMPRESSION prompt + exemplars.** IMPRESSION is 35% of each case. Match the
   reference's length, ordering, and connectives from your phrasebook.
6. **Critical-token repair.** After generation, verify every negation, laterality,
   number, and unit in the output is supported by the dictation/template; re-ask
   the model to fix any that are not. These tokens are weight 4.
7. **Test-time candidate selection.** Generate N=3 candidates per touched field or
   per report and keep the one closest (by `weighted_edit`) to the retrieved
   exemplars. This is a legitimate proxy for reference-likeness with no leakage.

Spend Kaggle submissions sparingly (5/day). Use them to confirm local/public
alignment a handful of times, not to iterate.

---

## Step 5. Validation + anti-hallucination gate (2 h)

Wire `validate.check(report, template, dictation)` into `run()`; log every issue.
Drive a repair pass: for any flagged report, re-ask the model to fix only the
flagged fields, then re-validate. Confirm on the holdout that the repair pass does
not raise local RES (it should lower or hold it while cutting fabrications).

Manual QA is allowed on your own outputs for debugging, but any fix must be a
**pipeline or prompt change that regenerates the CSV**. Do not hand-edit an
individual test report; that breaks the rules.

---

## Step 6. Notebook, reproducibility, disclosure (2 h)

- Assemble a single Kaggle notebook that imports these modules (or inlines them),
  reads `test.csv`, runs the pipeline, and writes `submission.csv`. It does not
  need to execute on Kaggle, but it must reproduce the CSV without manual edits.
- Use a **secret placeholder** for the API key (env var or Kaggle secret). No keys
  in the notebook or CSV.
- **Disclose everything**: provider, exact model ID, all prompts, temperature and
  other params, and dependencies. Add a short design section: architecture, why
  copy-untouched + retrieve + edit, the RES-proxy-driven tuning, and your holdout
  RES. This is where the system-design part of the hiring review is won.
- Save a version, share the private notebook with `natoeaidev`, and paste its URL
  into the submission description.

---

## Step 7. Final selection

Pick the **2** submissions for private scoring. Hedge them: one tuned aggressively
to your local RES, one slightly more conservative (less test-time selection, more
verbatim preservation) in case the private split rewards a different balance.
Submit both before **September 8, 2026, 11:59:59 PM IST**.

---

## Compliance checklist (do not trip)

- [ ] Automated pipeline only; no per-case hand edits of the test set.
- [ ] No API keys/secrets in the CSV or notebook; placeholders only.
- [ ] All providers, model IDs, prompts, params, and deps disclosed.
- [ ] Individual entry; <= 5 submissions/day; exactly 2 finals.
- [ ] `submission.csv` has exactly `case_id,report`, every test case once,
      `report` correctly quoted (the writer uses QUOTE_ALL).

## Notes on the RES proxy

`scorer.py` is a faithful reconstruction from the brief, not Natoe's exact code.
Two points are underspecified and marked ASSUMPTION in the file: the substitution
cost (currently `max(weight(a), weight(b))`) and decimal handling (currently kept).
If local and public RES diverge, revisit those first against train behaviour.
