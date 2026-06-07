"""
Benchmark Experiment B — No VAD / No Chunking / No Diarization

Uses faster-whisper on the full audio in a single pass with beam_size=5.
No Silero VAD. No chunking. No pyannote. No pipeline overhead.
"""

import json
import re
import time
import jiwer
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("Install faster-whisper: pip install faster-whisper")

try:
    from evaluate import _jiwer_transform, normalize_for_eval, apply_transcript_normalization
except ImportError:
    sys.exit("Could not import evaluate module")


def transcribe_full(audio_path: str) -> tuple[str, list[str], float]:
    """Transcribe entire audio in one pass. No VAD, no chunking, no pyannote."""
    print(f"Loading Whisper turbo model (int8)...")
    model = WhisperModel("turbo", device="auto", compute_type="int8")

    print(f"Transcribing {audio_path} (full file, one pass, beam=5)...")
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=None,
        vad_filter=False,
        beam_size=5,
        word_timestamps=True,
        task="transcribe",
    )
    full_text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.time() - t0
    languages = [info.language] if info.language else []
    print(f"  Done in {elapsed:.1f}s | Language: {info.language} (prob={info.language_probability:.2f})")
    print(f"  Transcript: {len(full_text)} chars")
    return full_text, languages, elapsed


def run_benchmark(audio_path: str, gt_path: str, normalize: bool = False) -> dict:
    """Run Experiment B and return metrics dict."""
    with open(gt_path) as f:
        gt = json.load(f)
    ref = gt["transcript"]

    hyp_raw, langs, runtime = transcribe_full(audio_path)
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", hyp_raw)

    if normalize:
        hyp = apply_transcript_normalization(hyp_raw)
        print(f"  Normalization: raw={len(hyp_raw)} → norm={len(hyp)} chars")
    else:
        hyp = hyp_raw

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
        "source": audio_path,
        "ground_truth": gt_path,
        "runtime_sec": round(runtime, 1),
        "wer_raw": wer_raw,
        "wer": round(jiwer.wer(ref, hyp, reference_transform=transform, hypothesis_transform=transform), 4),
        "normalized_wer": norm_wer,
        "transcript_len": len(hyp),
        "ref_len": len(ref),
        "languages": langs,
        "first_300_chars": hyp[:300],
        "entity_check": entity_results,
        "normalized": normalize,
    }


def print_results(result: dict, label: str):
    """Pretty-print benchmark results."""
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Runtime:     {result['runtime_sec']:.1f}s")
    print(f"  Languages:   {result['languages']}")
    print(f"  Transcript:  {result['transcript_len']} chars (ref: {result['ref_len']})")
    print(f"  WER (raw):   {result.get('wer_raw', result['wer']):.4f}")
    if result.get('normalized'):
        print(f"  WER (norm):  {result['wer']:.4f}")
    print(f"  Normalized WER: {result['normalized_wer']:.4f}")
    print(f"  Entities:")
    for e, found in result["entity_check"].items():
        print(f"    {'✅' if found else '❌'} {e}")
    print(f"  First 300 chars:")
    print(f"    {result['first_300_chars']}")


if __name__ == "__main__":
    # Default to bilingual_benchmark.wav (180s clip)
    audio = sys.argv[1] if len(sys.argv) > 1 else "sample_podcasts/bilingual_benchmark.wav"
    gt_file = sys.argv[2] if len(sys.argv) > 2 else "sample_podcasts/bilingual_benchmark_gt.json"
    normalize = "--normalize" in sys.argv

    result = run_benchmark(audio, gt_file, normalize=normalize)
    print_results(result, "Experiment B — No VAD / No Chunking / No Pyannote")

    # Save to JSON
    out_path = Path(__file__).parent / "results" / "benchmark_no_vad.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
