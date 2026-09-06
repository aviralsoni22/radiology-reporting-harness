"""
End-to-end pipeline: template + dictation -> report.

Architecture (surgical by design):
  1. parse template into ordered fields
  2. copy every field the dictation does not touch  -> verbatim (free zeros)
  3. for touched fields: LLM edits ONLY those, with retrieved train exemplars
  4. IMPRESSION: LLM generates, with retrieved impression exemplars
  5. reassemble in template order; enforce structure
  6. validate + repair (see validate.py)

The LLM sits behind LLMClient so everything except steps 3-4 runs with no key.
OpenRouter is the only provider; key comes from OPENROUTER_API_KEY (.env).
"""
import os
import re

from parse import parse_report, render_report
from scorer import normalize

try:  # load OPENROUTER_API_KEY from .env if python-dotenv is installed
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(usecwd=True))  # type: ignore
except ImportError:
    pass

# --------------------------------------------------------------------------
# LLM client (OpenRouter only; key from environment)
# --------------------------------------------------------------------------
class LLMClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(LLMClient):
    """One key, many models. Swap `model` to A/B test.
    Model IDs use provider/model form (e.g. 'anthropic/claude-opus-4.8').
    Check https://openrouter.ai/models for current IDs and live pricing.
    """
    def __init__(self, model="anthropic/claude-opus-4.8",
                 max_tokens=1024, temperature=0.0, timeout=60):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env in the project root:\n"
                "  OPENROUTER_API_KEY=sk-or-...\n"
                "Get a key at https://openrouter.ai/keys")
        self._headers = {"Authorization": f"Bearer {key}"}
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.timeout = timeout

    def complete(self, system, user):
        import requests
        r = requests.post(API_URL, headers=self._headers, timeout=self.timeout,
                          json={"model": self.model,
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

class EchoClient(LLMClient):
    """No-op client for wiring/tests: returns the template field unchanged."""
    def complete(self, system, user):
        m = re.search(r"CURRENT NORMAL FIELD:\s*(.*)", user)
        return m.group(1).strip() if m else ""


# --------------------------------------------------------------------------
# CHOOSE YOUR MODEL HERE
# Short name -> OpenRouter model slug. Verify/extend slugs and see live
# pricing at https://openrouter.ai/models.
# Pick with make_client("opus") in code, or set OPENROUTER_MODEL in .env
# (accepts a short name OR a full slug). "echo" runs the pipeline with no key.
# --------------------------------------------------------------------------
MODELS = {
    "opus":     "anthropic/claude-opus-4.8",
    "sonnet":   "anthropic/claude-sonnet-4.5",
    "haiku":    "anthropic/claude-3.5-haiku",
    "gpt":      "openai/gpt-4o",
    "gpt-mini": "openai/gpt-4o-mini",
    "gemini":   "google/gemini-2.0-flash-001",
    "deepseek": "deepseek/deepseek-chat",
    "llama":    "meta-llama/llama-3.1-70b-instruct",
}

DEFAULT_MODEL = "opus"   # <- change this to switch the pipeline's default


def make_client(model=None, **kw):
    """Return an LLMClient for the chosen model.
    `model`: a MODELS short name, a full OpenRouter slug, or "echo".
    Falls back to OPENROUTER_MODEL env, then DEFAULT_MODEL.
    Extra kwargs (max_tokens, temperature, timeout) pass to the client.
    """
    name = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
    if name == "echo":
        return EchoClient()
    slug = MODELS.get(name, name)   # unknown name -> treat as a raw slug
    return OpenRouterClient(model=slug, **kw)


# --------------------------------------------------------------------------
# Which fields does the dictation touch?
# Start simple: token overlap between dictation and each field's anatomy.
# Refine in Phase 0 with the real routing patterns.
# --------------------------------------------------------------------------
def touched_labels(dictation, template_fields, router=None):
    if router is not None:
        return router(dictation, template_fields)
    dict_tokens = set(normalize(dictation))
    hits = []
    for label, content in template_fields:
        anat = set(normalize(label)) | set(normalize(content))
        # crude signal: any shared content word -> candidate. Replace with a
        # learned router (label keywords per finding) after forensics.
        if dict_tokens & (anat - _STOP):
            hits.append(label)
    return hits or [l for l, _ in template_fields]  # fail open: let LLM decide


_STOP = set(normalize("no not of or the and with within normal identified "
                      "these views abnormality"))


# --------------------------------------------------------------------------
# Prompts (tune these against local RES in Phase 3)
# --------------------------------------------------------------------------
FIELD_SYSTEM = (
    "You edit one field of a radiology report. Rewrite the NORMAL field so it "
    "reflects the dictated finding for THIS field only. Rules: change only what "
    "the finding requires; keep every other word of the normal field verbatim; "
    "reuse the field's existing vocabulary; keep negations for anything not "
    "mentioned; match the phrasing style of the EXAMPLES. Output the field text "
    "only, no label, no commentary.")

IMPRESSION_SYSTEM = (
    "You write the IMPRESSION section of a radiology report: a concise summary of "
    "the abnormal findings only, in the style of the EXAMPLES. Do not restate "
    "normal fields. Do not add anything not supported by the findings. Output the "
    "impression text only.")


def _exemplar_block(exemplars):
    out = []
    for ex in exemplars:
        out.append(f"NORMAL: {ex['template']}\nFINDING: {ex['dictation']}\n"
                   f"RESULT: {ex['reference']}")
    return "\n---\n".join(out)


def edit_field(client, label, normal_text, dictation, exemplars):
    user = (f"EXAMPLES:\n{_exemplar_block(exemplars)}\n\n"
            f"FIELD LABEL: {label}\n"
            f"DICTATION (full): {dictation}\n"
            f"CURRENT NORMAL FIELD: {normal_text}")
    return client.complete(FIELD_SYSTEM, user)


def generate_impression(client, changed_summary, dictation, exemplars):
    user = (f"EXAMPLES (findings -> impression):\n{_exemplar_block(exemplars)}\n\n"
            f"ABNORMAL FINDINGS IN THIS REPORT:\n{changed_summary}\n"
            f"DICTATION (full): {dictation}")
    return client.complete(IMPRESSION_SYSTEM, user)


# --------------------------------------------------------------------------
# Per-case pipeline
# --------------------------------------------------------------------------
def build_report(row, client, retriever=None, router=None):
    tmpl = parse_report(row["template_content"])
    fields = tmpl["fields"]
    dictation = row["dictation"]
    to_edit = set(touched_labels(dictation, fields, router))

    out_fields, changed_bits = [], []
    for label, normal_text in fields:
        if label in to_edit:
            ex = retriever.for_field(row, label) if retriever else []
            new_text = edit_field(client, label, normal_text, dictation, ex)
            new_text = new_text or normal_text          # never blank a field
            out_fields.append((label, new_text))
            if normalize(new_text) != normalize(normal_text):
                changed_bits.append(f"{label}: {new_text}")
        else:
            out_fields.append((label, normal_text))       # verbatim (free zero)

    imp_ex = retriever.for_impression(row) if retriever else []
    impression = generate_impression(
        client, "\n".join(changed_bits) or "None.", dictation, imp_ex)
    return render_report(out_fields, impression)


# --------------------------------------------------------------------------
# Batch runner -> submission.csv
# --------------------------------------------------------------------------
def run(test_rows, client, retriever=None, router=None):
    results = []
    for i, row in enumerate(test_rows, 1):
        try:
            report = build_report(row, client, retriever, router)
        except Exception as e:                              # never drop a case
            report = row["template_content"]
            print(f"[warn] case {row.get('case_id')} fell back to template: {e}")
        results.append({"case_id": row["case_id"], "report": report})
        if i % 20 == 0:
            print(f"  {i}/{len(test_rows)}")
    return results


def write_submission(results, path="submission.csv"):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["case_id", "report"])
        for r in results:
            w.writerow([r["case_id"], r["report"]])
    print(f"wrote {len(results)} rows -> {path}")
