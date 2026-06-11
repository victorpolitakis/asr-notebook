"""
Post-Processing Normalization Benchmark

Tests 4 transcript normalization variants to determine whether
LLM-based cleanup improves entity extraction, topic recall, and
summarization quality without changing ASR output.

Variants:
  A — Raw (no modification)
  B — Rule-Based Normalization (deterministic, no LLM)
  C — LLM Transcript Cleanup (strong model)
  D — Rule-Based + LLM (B → C)

Usage:
  python Politakis/benchmark_normalize.py <transcript.txt> <ground_truth.json>
"""

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Pipeline"))

from openai import OpenAI
from evaluate import _jiwer_transform, normalize_for_eval

# ── Stronger model than the current normalization default ───────────
CLEANUP_MODEL = os.getenv("BENCHMARK_CLEANUP_MODEL", "gpt-5.4-mini-2026-03-17")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# ── Cleanup prompt (stricter than normalization prompt) ─────────────
CLEANUP_PROMPT = """\
You are repairing a multilingual ASR transcript.

Do ONLY these:
1. Restore corrupted person names (e.g. "yan lecun" → "Yann LeCun",
   "Ιαν Λε Κων" → "Yann LeCun", "jeffrey hinton" → "Geoffrey Hinton").
2. Restore corrupted organization names (e.g. "openai" → "OpenAI",
   "google deepmind" → "Google DeepMind").
3. Restore corrupted technical terms (e.g. "api silicon" → "Apple Silicon",
   "gpt 4" → "GPT-4", "metal performance aders" → "Metal Performance Shaders").
4. Fix capitalization of known entities when confidence is high.
5. Fix obvious tokenization issues (e.g. "CTranslate2" → "CTranslate2").

Do NOT:
- Summarize, paraphrase, reorder, translate, or rewrite sentences.
- Add or remove any information.
- Improve grammar, style, or wording.
- Modify anything other than proper nouns, entities, and technical terms.

If confidence in a correction is low, leave the text unchanged.

Return ONLY the repaired transcript — no markdown, no explanations.
"""

ENTITIES = [
    "Yann LeCun", "Fei-Fei Li", "Geoffrey Hinton", "Sam Altman",
    "Chin-Yew Lin", "Herve Bredin", "OpenAI", "Google DeepMind",
    "Apple Silicon", "Metal Performance Shaders", "GPT-4", "Gemini",
    "Claude", "Whisper", "NVIDIA", "Meta", "Anthropic", "Carnegie Mellon",
    "MIT", "Khan Academy", "Duolingo", "NLP", "NER", "ROUGE",
    "Apple", "EU AI Act", "DeepMind", "Facebook AI Research",
]


def load_raw_transcript(path: str) -> str:
    """Load transcript.txt — tries UTF-8, falls back to latin-1."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript not found: {path}. Run run_pipeline.py first.")
    raw_bytes = p.read_bytes()
    if len(raw_bytes) == 0:
        raise ValueError(f"Transcript is empty: {path}")
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(encoding).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot decode {path} with any encoding")


def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Variant A — Raw (no modification)
# ═══════════════════════════════════════════════════════════════════════════

def variant_a(raw: str) -> str:
    return raw


# ═══════════════════════════════════════════════════════════════════════════
# Variant B — Rule-Based Normalization
# ═══════════════════════════════════════════════════════════════════════════

def variant_b(raw: str) -> str:
    text = raw
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Line ending normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip speaker labels
    text = re.sub(r"\[Speaker [A-Z]+\]:?\s*", "", text)
    # Collapse repeated punctuation (3+ → 1)
    text = re.sub(r"([.!?])\1{2,}", r"\1", text)
    # Strip leading/trailing whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Final strip
    text = text.strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Variant C — LLM Transcript Cleanup
# ═══════════════════════════════════════════════════════════════════════════

def variant_c(raw: str) -> str:
    if not LLM_API_KEY:
        print("  ⚠️  No OPENAI_API_KEY — returning raw transcript.")
        return raw

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=120.0)
        # Split long transcripts into chunks
        max_chunk = 6000
        chunks = [raw]
        if len(raw) > max_chunk:
            # Split on double newline
            paragraphs = re.split(r"\n\n+", raw)
            chunks = []
            current = ""
            for p in paragraphs:
                if len(current) + len(p) > max_chunk and current:
                    chunks.append(current.strip())
                    current = p
                else:
                    current += "\n\n" + p if current else p
            if current.strip():
                chunks.append(current.strip())

        cleaned_parts = []
        for i, chunk in enumerate(chunks):
            print(f"  LLM cleanup chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            response = client.chat.completions.create(
                model=CLEANUP_MODEL,
                temperature=0,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": CLEANUP_PROMPT},
                    {"role": "user", "content": chunk},
                ],
            )
            result = response.choices[0].message.content or ""
            cleaned_parts.append(result)

        cleaned = "\n\n".join(cleaned_parts)

        # Anti-hallucination: length must stay within 85–115% of original
        ratio = len(cleaned) / max(len(raw), 1)
        if ratio < 0.85 or ratio > 1.15:
            print(f"  ⚠️  Length ratio {ratio:.2f} out of bounds — falling back to raw.")
            return raw

        return cleaned
    except Exception as e:
        print(f"  ⚠️  LLM cleanup failed: {e} — falling back to raw.")
        return raw


# ═══════════════════════════════════════════════════════════════════════════
# Variant D — Rule-Based + LLM
# ═══════════════════════════════════════════════════════════════════════════

def variant_d(raw: str) -> str:
    return variant_c(variant_b(raw))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_variant(name: str, hyp: str, ref: str, raw: str) -> dict:
    """Run full evaluation suite on one variant."""
    transform = _jiwer_transform()

    # WER against raw reference
    wer = round(jiwer.wer(ref, hyp, reference_transform=transform, hypothesis_transform=transform), 4)

    # Normalized WER
    norm_ref = normalize_for_eval(ref)
    norm_hyp = normalize_for_eval(hyp)
    norm_wer = round(jiwer.wer(norm_ref, norm_hyp), 4)

    # ROUGE-1
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge1 = round(scorer.score(ref, hyp)["rouge1"].fmeasure, 4)
    except Exception:
        rouge1 = -1.0

    # Entity detection
    entity_check = {}
    for e in ENTITIES:
        entity_check[e] = e.lower() in hyp.lower()

    found = sum(1 for v in entity_check.values() if v)

    # Diff summary
    import difflib
    changes = list(difflib.SequenceMatcher(None, raw, hyp).get_opcodes())
    edit_regions = sum(1 for tag, *_ in changes if tag != "equal")

    return {
        "variant": name,
        "transcript_len": len(hyp),
        "ref_len": len(ref),
        "raw_len": len(raw),
        "edit_regions": edit_regions,
        "wer": wer,
        "normalized_wer": norm_wer,
        "rouge1_f1": rouge1,
        "entity_check": entity_check,
        "entities_found": found,
        "entities_total": len(ENTITIES),
        "first_200": hyp[:200],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "results/transcript.txt"
    gt_path  = sys.argv[2] if len(sys.argv) > 2 else "Samples/sample_podcasts/bilingual_long_gt.json"

    raw_text = load_raw_transcript(raw_path)
    gt = load_ground_truth(gt_path)
    ref = gt["transcript"]

    print(f"Raw transcript: {len(raw_text)} chars")
    print(f"Ground truth:   {len(ref)} chars")
    print(f"Model: {CLEANUP_MODEL}")
    print()

    results = []

    # Variant A — Raw
    print("── Variant A: Raw ──")
    t0 = time.time()
    va = variant_a(raw_text)
    r = evaluate_variant("A — Raw", va, ref, raw_text)
    r["runtime_sec"] = round(time.time() - t0, 2)
    results.append(r)
    print(f"  {r['runtime_sec']}s | {r['transcript_len']} chars | WER={r['wer']} | ROUGE1={r['rouge1_f1']}")

    # Variant B — Rule-Based
    print("── Variant B: Rule-Based ──")
    t0 = time.time()
    vb = variant_b(raw_text)
    r = evaluate_variant("B — Rule-Based", vb, ref, raw_text)
    r["runtime_sec"] = round(time.time() - t0, 2)
    results.append(r)
    print(f"  {r['runtime_sec']}s | {r['transcript_len']} chars | WER={r['wer']} | ROUGE1={r['rouge1_f1']}")

    # Variant C — LLM Cleanup
    print("── Variant C: LLM Cleanup ──")
    t0 = time.time()
    vc = variant_c(raw_text)
    r = evaluate_variant("C — LLM Cleanup", vc, ref, raw_text)
    r["runtime_sec"] = round(time.time() - t0, 2)
    results.append(r)
    print(f"  {r['runtime_sec']}s | {r['transcript_len']} chars | WER={r['wer']} | ROUGE1={r['rouge1_f1']}")

    # Variant D — Rule-Based + LLM
    print("── Variant D: Rule-Based + LLM ──")
    t0 = time.time()
    vd = variant_d(raw_text)
    r = evaluate_variant("D — Rule-Based + LLM", vd, ref, raw_text)
    r["runtime_sec"] = round(time.time() - t0, 2)
    results.append(r)
    print(f"  {r['runtime_sec']}s | {r['transcript_len']} chars | WER={r['wer']} | ROUGE1={r['rouge1_f1']}")

    # ── Print summary table ───────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"  {'Variant':25s} | {'Runtime':>7s} | {'WER':>7s} | {'nWER':>7s} | {'ROUGE-1':>7s} | {'Entities':>8s}")
    print(f"  {'-'*25} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*8}")
    for r in results:
        print(f"  {r['variant']:25s} | {r['runtime_sec']:6.1f}s | {r['wer']:7.4f} | "
              f"{r['normalized_wer']:7.4f} | {r['rouge1_f1']:7.4f} | "
              f"{r['entities_found']:3d}/{r['entities_total']}")

    print()
    print("─ Entity Detection Details ─")
    print(f"  {'Entity':30s} | {'A':^3} | {'B':^3} | {'C':^3} | {'D':^3}")
    for e in ENTITIES:
        marks = "".join(
            " ✅" if r["entity_check"].get(e) else " ❌"
            for r in results
        )
        print(f"  {e:30s} |{marks}")

    # Save
    out_path = Path(__file__).parent.parent / "Results" / "benchmark_normalize.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
