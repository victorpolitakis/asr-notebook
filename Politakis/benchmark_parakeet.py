"""
Benchmark — NVIDIA Parakeet TDT 0.6B v3

Uses HuggingFace Transformers with AutoModelForTDT.
25 languages including Greek. Built-in punctuation + capitalization.
Native long-form support (up to 24 min with full attention).

Usage:
  python Politakis/benchmark_parakeet.py <audio.wav> <ground_truth.json>

Colab prerequisites:
  !pip install git+https://github.com/huggingface/transformers accelerate
  !pip install datasets[audio] jiwer rouge-score rapidfuzz

GPU runtime recommended (T4 is sufficient).
"""

import json
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import torch
from transformers import AutoModelForTDT, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import _jiwer_transform, normalize_for_eval, apply_transcript_normalization

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

ENTITIES = [
    "Yann LeCun", "Fei-Fei Li", "Geoffrey Hinton", "Sam Altman",
    "Chin-Yew Lin", "Herve Bredin", "OpenAI", "Google DeepMind",
    "Apple Silicon", "Metal Performance Shaders", "GPT-4", "Gemini",
    "Claude", "Whisper", "NVIDIA", "Meta", "Anthropic",
    "MIT", "Khan Academy", "Duolingo", "NLP", "NER", "ROUGE",
]


def transcribe_parakeet(audio_path: str) -> tuple[str, float]:
    """Transcribe full audio with Parakeet TDT. Returns (text, runtime_sec)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} on {device} ({torch_dtype})...")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except ImportError as e:
        if "Numba needs NumPy" in str(e) or "librosa" in str(e):
            sys.exit(
                "Parakeet needs NumPy 2.0.x. Run: pip install numpy==2.0.2\n"
                "Then restart the runtime: Runtime → Restart runtime"
            )
        raise
    model = AutoModelForTDT.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print(f"Processing audio: {audio_path}")
    t0 = time.time()

    # Load audio with correct sample rate
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    audio_array = waveform.squeeze().numpy()

    # Process in chunks if needed (supports up to 24 min natively)
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        truncation=False,
        padding="longest",
    )
    inputs = inputs.to(device, dtype=model.dtype)

    print(f"  Audio: {len(audio_array)/sample_rate:.1f}s — running inference...")
    with torch.no_grad():
        output = model.generate(**inputs, return_dict_in_generate=True)

    transcription = processor.decode(output.sequences, skip_special_tokens=True)
    elapsed = time.time() - t0

    # Handle tuple/list return from decode
    if isinstance(transcription, (list, tuple)):
        transcription = transcription[0] if transcription else ""

    print(f"  Done in {elapsed:.1f}s | Transcript: {len(transcription)} chars")
    return transcription, elapsed


def run_benchmark(audio_path: str, gt_path: str, normalize: bool = False) -> dict:
    with open(gt_path) as f:
        gt = json.load(f)
    ref = gt["transcript"]

    hyp_raw, runtime = transcribe_parakeet(audio_path)
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", hyp_raw)

    if normalize:
        hyp = apply_transcript_normalization(hyp_raw)
        print(f"  Normalization: raw={len(hyp_raw)} → norm={len(hyp)} chars")
    else:
        hyp = hyp_raw

    transform = _jiwer_transform()
    wer_raw = round(jiwer.wer(ref, hyp_raw, reference_transform=transform, hypothesis_transform=transform), 4)

    norm_ref = normalize_for_eval(ref)
    norm_hyp = normalize_for_eval(hyp)
    norm_wer = round(jiwer.wer(norm_ref, norm_hyp), 4)

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge1 = round(scorer.score(ref, hyp)["rouge1"].fmeasure, 4)
    except Exception:
        rouge1 = -1.0

    entity_check = {}
    for e in ENTITIES:
        entity_check[e] = e.lower() in hyp.lower()
    found = sum(1 for v in entity_check.values() if v)

    return {
        "model": MODEL_ID,
        "runtime_sec": round(runtime, 1),
        "wer_raw": wer_raw,
        "wer": round(jiwer.wer(ref, hyp, reference_transform=transform, hypothesis_transform=transform), 4),
        "normalized_wer": norm_wer,
        "rouge1_f1": rouge1,
        "transcript_len": len(hyp),
        "ref_len": len(ref),
        "entity_check": entity_check,
        "entities_found": found,
        "entities_total": len(ENTITIES),
        "first_400": hyp[:400],
        "normalized": normalize,
    }


def print_results(r: dict):
    print()
    print("=" * 60)
    print(f"  {r['model']}")
    print("=" * 60)
    print(f"  Runtime:     {r['runtime_sec']:.1f}s")
    print(f"  Transcript:  {r['transcript_len']} chars (ref: {r['ref_len']})")
    print(f"  WER (raw):   {r.get('wer_raw', r['wer']):.4f}")
    if r.get('normalized'):
        print(f"  WER (norm):  {r['wer']:.4f}")
    print(f"  Normalized WER: {r['normalized_wer']:.4f}")
    print(f"  ROUGE-1:     {r['rouge1_f1']:.4f}")
    print(f"  Entities:    {r['entities_found']}/{r['entities_total']}")
    print(f"  Entity check:")
    for e, found in r["entity_check"].items():
        print(f"    {'✅' if found else '❌'} {e}")
    print(f"  First 400:")
    print(f"    {r['first_400']}")


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "sample_podcasts/bilingual_benchmark.wav"
    gt_file = sys.argv[2] if len(sys.argv) > 2 else "sample_podcasts/bilingual_benchmark_gt.json"
    normalize = "--normalize" in sys.argv

    result = run_benchmark(audio, gt_file, normalize=normalize)
    print_results(result)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_parakeet.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
