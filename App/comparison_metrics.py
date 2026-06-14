"""
comparison_metrics.py — Interactive transcript comparison metrics

Computes WER, CER, ROUGE-1/2/L, BLEU, readability stats, and
generates rich diff views + downloadable reports for the
Accuracy Check page.
"""

from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Readability Statistics
# ═══════════════════════════════════════════════════════════════

def compute_readability(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "estimated_reading_time_sec": 0.0,
        }
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    avg_word_length = (
        round(sum(len(w) for w in words) / word_count, 1) if word_count else 0.0
    )
    reading_time_sec = round((word_count / 150) * 60, 1)
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "estimated_reading_time_sec": reading_time_sec,
    }


# ═══════════════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════════════

def _jiwer_transform():
    import jiwer

    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def _aggressive_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[?speaker\s*[a-z]+\]?:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # Replace hyphens/dashes with spaces
    text = text.replace("-", " ").replace("—", " ").replace("–", " ")
    # Remove common English and Greek punctuation, quotes, brackets, and symbols
    text = re.sub(r'[.,;:!?!"\'()«»“”‘’·]', "", text)
    # Remove multiple spaces and strip
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# WER / CER
# ═══════════════════════════════════════════════════════════════

def compute_wer(hypothesis: str, reference: str) -> dict[str, Any]:
    if not reference.strip():
        return {
            "wer": 1.0,
            "cer": 1.0,
            "normalized_wer": 1.0,
            "error": "empty_reference",
        }
    import jiwer

    transform = _jiwer_transform()
    wer_val = round(
        jiwer.wer(
            reference,
            hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        ),
        4,
    )
    cer_val = round(
        jiwer.cer(
            reference,
            hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        ),
        4,
    )

    norm_ref = _aggressive_normalize(reference)
    norm_hyp = _aggressive_normalize(hypothesis)
    norm_wer = round(jiwer.wer(norm_ref, norm_hyp), 4)

    return {"wer": wer_val, "cer": cer_val, "normalized_wer": norm_wer}


# ═══════════════════════════════════════════════════════════════
# ROUGE
# ═══════════════════════════════════════════════════════════════

def compute_rouge(hypothesis: str, reference: str) -> dict[str, Any]:
    if not reference.strip():
        return {"rouge1": {}, "rouge2": {}, "rougeL": {}, "error": "empty_reference"}
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    def _pack(score):
        return {
            "f1": round(score.fmeasure, 4),
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
        }

    return {
        "rouge1": _pack(scores["rouge1"]),
        "rouge2": _pack(scores["rouge2"]),
        "rougeL": _pack(scores["rougeL"]),
    }


# ═══════════════════════════════════════════════════════════════
# BLEU
# ═══════════════════════════════════════════════════════════════

def compute_bleu(hypothesis: str, reference: str) -> dict[str, Any]:
    if not reference.strip():
        return {"bleu": 0.0, "error": "empty_reference"}
    try:
        from sacrebleu.metrics import BLEU

        bleu = BLEU()
        score = bleu.corpus_score([hypothesis], [[reference]])
        return {
            "bleu": round(score.score, 2),
            "precisions": [round(p, 1) for p in score.precisions],
            "brevity_penalty": round(score.bp, 2),
            "sys_len": score.sys_len,
            "ref_len": score.ref_len,
        }
    except ImportError:
        return {"bleu": 0.0, "error": "sacrebleu_not_installed"}
    except Exception as exc:
        return {"bleu": 0.0, "error": str(exc)[:120]}


# ═══════════════════════════════════════════════════════════════
# Diff Viewers
# ═══════════════════════════════════════════════════════════════

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_diff_html(
    reference: str, hypothesis: str, mode: str = "word"
) -> str:
    if mode == "char":
        ref_seq = list(reference)
        hyp_seq = list(hypothesis)
    else:
        ref_seq = reference.split()
        hyp_seq = hypothesis.split()

    matcher = difflib.SequenceMatcher(None, ref_seq, hyp_seq)
    parts: list[str] = []
    joiner = "" if mode == "char" else " "

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            chunk = joiner.join(ref_seq[i1:i2])
            if chunk:
                parts.append(f"<span>{_escape(chunk)}</span>")
        elif tag == "insert":
            chunk = joiner.join(hyp_seq[j1:j2])
            if chunk:
                parts.append(
                    f'<span style="background:#0e2a1a;color:#4aff9e;padding:1px 3px;border-radius:3px">'
                    f"{_escape(chunk)}</span>"
                )
        elif tag == "delete":
            chunk = joiner.join(ref_seq[i1:i2])
            if chunk:
                parts.append(
                    f'<span style="background:#2a0e0e;color:#ff5a5a;text-decoration:line-through;padding:1px 3px;border-radius:3px">'
                    f"{_escape(chunk)}</span>"
                )
        elif tag == "replace":
            old_chunk = joiner.join(ref_seq[i1:i2])
            new_chunk = joiner.join(hyp_seq[j1:j2])
            if old_chunk:
                parts.append(
                    f'<span style="background:#2a0e0e;color:#ff5a5a;text-decoration:line-through;padding:1px 3px;border-radius:3px">'
                    f"{_escape(old_chunk)}</span>"
                )
            if new_chunk:
                parts.append(
                    f'<span style="background:#0e2a1a;color:#4aff9e;padding:1px 3px;border-radius:3px">'
                    f"{_escape(new_chunk)}</span>"
                )

    return joiner.join(parts)


# ═══════════════════════════════════════════════════════════════
# Combined Metrics
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(
    hypothesis: str, reference: str, label: str = ""
) -> dict[str, Any]:
    norm_hyp = _normalize_for_comparison(hypothesis)
    norm_ref = _normalize_for_comparison(reference)

    hyp_read = compute_readability(norm_hyp)
    ref_read = compute_readability(norm_ref)
    wer_data = compute_wer(norm_hyp, norm_ref)
    rouge_data = compute_rouge(norm_hyp, norm_ref)
    bleu_data = compute_bleu(norm_hyp, norm_ref)
    diff_word = generate_diff_html(norm_ref, norm_hyp, mode="word")

    return {
        "label": label or "Comparison",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": {
            "text_preview": norm_hyp[:500] + ("…" if len(norm_hyp) > 500 else ""),
            "readability": hyp_read,
        },
        "reference": {
            "text_preview": norm_ref[:500] + ("…" if len(norm_ref) > 500 else ""),
            "readability": ref_read,
        },
        "wer": wer_data,
        "rouge": rouge_data,
        "bleu": bleu_data,
        "diff_word_html": diff_word,
    }


# ═══════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════

def generate_report_json(result: dict) -> str:
    clean = {k: v for k, v in result.items() if k not in ("diff_word_html", "diff_char_html")}
    return json.dumps(clean, indent=2, ensure_ascii=False)


def generate_report_txt(result: dict) -> str:
    lines = [
        "=" * 70,
        f"  TRANSCRIPT COMPARISON REPORT",
        f"  Label: {result.get('label', 'N/A')}",
        f"  Generated: {result.get('evaluated_at', 'N/A')}",
        "=" * 70,
        "",
        "── Readability ──────────────────────────────────────────────",
    ]
    for side, slabel in [("hypothesis", "Hypothesis"), ("reference", "Reference")]:
        r = result.get(side, {}).get("readability", {})
        lines.append(
            f"  {slabel}: "
            f"Words={r.get('word_count', 0)} "
            f"Chars={r.get('char_count', 0)} "
            f"Sentences={r.get('sentence_count', 0)} "
            f"AvgWordLen={r.get('avg_word_length', 0)} "
            f"ReadTime={r.get('estimated_reading_time_sec', 0):.1f}s"
        )
    lines.append("")

    lines.append("── ASR Accuracy ─────────────────────────────────────────────")
    w = result.get("wer", {})
    if w.get("error"):
        lines.append(f"  WER/CER: SKIPPED ({w['error']})")
    else:
        lines.append(f"  WER:             {w.get('wer', 'N/A')}")
        lines.append(f"  CER:             {w.get('cer', 'N/A')}")
        lines.append(f"  Norm. WER:       {w.get('normalized_wer', 'N/A')}")
    lines.append("")

    lines.append("── ROUGE Scores ──────────────────────────────────────────────")
    for rkey, rlabel in [
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
    ]:
        r = result.get("rouge", {}).get(rkey, {})
        lines.append(
            f"  {rlabel}: "
            f"F1={r.get('f1', 'N/A')} "
            f"P={r.get('precision', 'N/A')} "
            f"R={r.get('recall', 'N/A')}"
        )
    lines.append("")

    lines.append("── BLEU Score ───────────────────────────────────────────────")
    b = result.get("bleu", {})
    if b.get("error"):
        lines.append(f"  BLEU: N/A ({b['error']})")
    else:
        lines.append(f"  BLEU:            {b.get('bleu', 'N/A')}")
        lines.append(f"  n-gram prec:     {b.get('precisions', [])}")
        lines.append(f"  Brevity penalty: {b.get('brevity_penalty', 'N/A')}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Bulk Comparison
# ═══════════════════════════════════════════════════════════════

def compare_models(
    reference: str, hypotheses: list[dict[str, str]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in hypotheses:
        try:
            r = compute_all_metrics(entry["hypothesis"], reference, entry["label"])
            results.append(r)
        except Exception as exc:
            results.append(
                {
                    "label": entry.get("label", "Unknown"),
                    "error": str(exc)[:200],
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return results


def bulk_results_to_dataframe(results: list[dict[str, Any]]) -> "pd.DataFrame":
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"Model": r.get("label", "?")}
        if "error" in r:
            row["Error"] = r["error"]
        else:
            row["WER"] = r.get("wer", {}).get("wer")
            row["CER"] = r.get("wer", {}).get("cer")
            row["Norm WER"] = r.get("wer", {}).get("normalized_wer")
            r1 = r.get("rouge", {}).get("rouge1", {})
            r2 = r.get("rouge", {}).get("rouge2", {})
            rL = r.get("rouge", {}).get("rougeL", {})
            row["ROUGE-1 F1"] = r1.get("f1")
            row["ROUGE-2 F1"] = r2.get("f1")
            row["ROUGE-L F1"] = rL.get("f1")
            b = r.get("bleu", {})
            row["BLEU"] = b.get("bleu") if "error" not in b else None
            hyp_r = r.get("hypothesis", {}).get("readability", {})
            ref_r = r.get("reference", {}).get("readability", {})
            row["Hyp Words"] = hyp_r.get("word_count")
            row["Ref Words"] = ref_r.get("word_count")
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
