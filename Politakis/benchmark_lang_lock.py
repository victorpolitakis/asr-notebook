"""
Benchmark — Language Locking vs Auto-Detect

Hypothesis: Explicit per-chunk language locking (when confidence >= 0.90)
            improves multilingual entity preservation without degrading WER.

Experiment C:
  - For each VAD chunk, run faster-whisper detect_language() first.
  - If language_probability >= 0.90: transcribe with language=detected_language.
  - If language_probability <  0.90: transcribe with language=None (auto).
  - Same VAD, same chunk sizes, same evaluation.

Compare against baseline (language=None on all chunks).
"""

import json
import re
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import jiwer

sys.path.insert(0, str(Path(__file__).parent))

from faster_whisper import WhisperModel
from audio_processor import process_audio_file
from evaluate import _jiwer_transform, normalize_for_eval, apply_transcript_normalization

LANG_CONFIDENCE_THRESHOLD = 0.90
MODEL_SIZE = "turbo"
COMPUTE_TYPE = "int8"


def benchmark_language_locking(audio_path: str, gt_path: str, normalize: bool = False) -> dict:
    with open(gt_path) as f:
        gt = json.load(f)
    ref = gt["transcript"]

    print(f"Loading Whisper {MODEL_SIZE} ({COMPUTE_TYPE})...")
    model = WhisperModel(MODEL_SIZE, device="auto", compute_type=COMPUTE_TYPE)

    t0 = time.time()
    chunks_transcribed = 0
    chunks_locked = 0
    chunks_auto = 0
    lang_counts: Counter = Counter()
    all_texts: list[str] = []

    for chunk in process_audio_file(audio_path, min_chunk_sec=28, max_chunk_sec=30):
        if not chunk.get("is_speech"):
            continue

        audio = chunk["audio_data"]
        sample_rate = chunk["sample_rate"]
        chunk_id = chunk["chunk_id"]

        # Step 1: detect language
        detection_info = model.detect_language(audio)
        det_lang = detection_info[0] if isinstance(detection_info, tuple) else detection_info
        det_prob = detection_info[1] if isinstance(detection_info, tuple) and len(detection_info) > 1 else 1.0

        # Step 2: decide whether to lock
        if det_prob >= LANG_CONFIDENCE_THRESHOLD:
            lang_param, chunks_locked = det_lang, chunks_locked + 1
        else:
            lang_param, chunks_auto = None, chunks_auto + 1

        # Step 3: transcribe
        segments, info = model.transcribe(
            audio, language=lang_param, vad_filter=False,
            beam_size=3, word_timestamps=True, task="transcribe",
        )

        chunk_text = " ".join(seg.text.strip() for seg in segments)
        all_texts.append(chunk_text)
        lang_counts[info.language] += 1
        chunks_transcribed += 1

        print(f"  Chunk {chunk_id}: detect={det_lang}({det_prob:.2f}) → "
              f"{'LOCKED' if lang_param else 'AUTO'} → actual={info.language}({info.language_probability:.2f}) "
              f"| {len(chunk_text)} chars")

    elapsed = time.time() - t0
    hyp_raw = " ".join(all_texts)
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", hyp_raw)

    if normalize:
        hyp = apply_transcript_normalization(hyp_raw)
        print(f"  Normalization: raw={len(hyp_raw)} → norm={len(hyp)} chars")
    else:
        hyp = hyp_raw

    # WER
    transform = _jiwer_transform()
    wer_raw = round(jiwer.wer(ref, hyp_raw, reference_transform=transform, hypothesis_transform=transform), 4)

    # Normalized WER
    norm_ref = normalize_for_eval(ref)
    norm_hyp = normalize_for_eval(hyp)
    norm_wer = round(jiwer.wer(norm_ref, norm_hyp), 4)

    # Entity check
    entities = [
        "Yann LeCun", "Fei-Fei Li", "Geoffrey Hinton", "Sam Altman",
        "Chin-Yew Lin", "Herve Bredin", "OpenAI", "Google DeepMind",
        "Apple Silicon", "Metal Performance Shaders",
    ]
    entity_results = {}
    for e in entities:
        entity_results[e] = e.lower() in hyp.lower()

    return {
        "runtime_sec": round(elapsed, 1),
        "chunks": chunks_transcribed,
        "chunks_locked": chunks_locked,
        "chunks_auto": chunks_auto,
        "wer_raw": wer_raw,
        "wer": round(jiwer.wer(ref, hyp, reference_transform=transform, hypothesis_transform=transform), 4),
        "normalized_wer": norm_wer,
        "transcript_len": len(hyp),
        "ref_len": len(ref),
        "languages": dict(lang_counts.most_common()),
        "entity_check": entity_results,
        "first_300": hyp[:300],
        "normalized": normalize,
    }


def print_results(r: dict, label: str):
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Runtime:     {r['runtime_sec']:.1f}s")
    print(f"  Chunks:      {r['chunks']} ({r['chunks_locked']} locked, {r['chunks_auto']} auto)")
    print(f"  Languages:   {r['languages']}")
    print(f"  Transcript:  {r['transcript_len']} chars (ref: {r['ref_len']})")
    print(f"  WER:         {r['wer']:.4f}")
    print(f"  Normalized WER: {r['normalized_wer']:.4f}")
    print(f"  Entities:")
    for e, found in r["entity_check"].items():
        print(f"    {'✅' if found else '❌'} {e}")
    print(f"  First 300:")
    print(f"    {r['first_300']}")


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "sample_podcasts/bilingual_benchmark.wav"
    gt_file = sys.argv[2] if len(sys.argv) > 2 else "sample_podcasts/bilingual_benchmark_gt.json"
    normalize = "--normalize" in sys.argv

    result = benchmark_language_locking(audio, gt_file, normalize=normalize)
    print_results(result, "Experiment C — Language Locking (threshold=0.90)")

    out_path = Path(__file__).parent / "results" / "benchmark_lang_lock.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
