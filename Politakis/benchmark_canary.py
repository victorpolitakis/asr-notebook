"""
Benchmark — NVIDIA Canary 1B V2 (25 languages, includes Greek)

Uses NeMo (nemo_toolkit[asr]) on GPU.
Canary supports audio up to 40s natively. For longer audio,
we manually chunk using simple 35s windows with small overlap.

Model: nvidia/canary-1b-v2
  - 25 European languages including Greek (el), English (en)
  - Built-in punctuation + capitalization

Colab prerequisites:
  !pip install nemo_toolkit[asr] jiwer rouge-score

Usage:
  python Politakis/benchmark_canary.py <audio.wav> <ground_truth.json>
"""

import json
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import _jiwer_transform, normalize_for_eval, apply_transcript_normalization

MODEL_ID = "nvidia/canary-1b-v2"
CHUNK_SEC = 35.0   # within Canary's 40s native limit
OVERLAP_SEC = 2.0  # small overlap to avoid cutting words

ENTITIES = [
    "Yann LeCun", "Fei-Fei Li", "Geoffrey Hinton", "Sam Altman",
    "Chin-Yew Lin", "Herve Bredin", "OpenAI", "Google DeepMind",
    "Apple Silicon", "Metal Performance Shaders", "GPT-4", "Gemini",
    "Claude", "Whisper", "NVIDIA", "Meta", "Anthropic",
    "MIT", "Khan Academy", "Duolingo", "NLP", "NER", "ROUGE",
]


def transcribe_canary(audio_path: str) -> tuple[str, float]:
    """Transcribe audio with Canary 1B V2. Chunks long audio manually."""
    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel
    except ImportError:
        sys.exit("NeMo not installed. Run: pip install nemo_toolkit[asr]")

    import torch
    device = "cuda"
    if not torch.cuda.is_available():
        sys.exit("Canary requires CUDA — enable GPU runtime (T4) in Colab.")

    print(f"Loading {MODEL_ID} on {device}...")
    t0 = time.time()

    model = EncDecMultiTaskModel.from_pretrained(MODEL_ID)
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    model = model.to(device)
    model.eval()

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Load audio as numpy array at 16 kHz mono
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze().numpy().astype(np.float32)
    duration = len(audio) / sample_rate
    print(f"  Audio: {duration:.1f}s — transcribing...")

    t1 = time.time()

    if duration <= 40.0:
        # Short audio: direct NeMo transcription
        output = model.transcribe([audio_path], batch_size=1, pnc='yes')
        transcription = output[0].text if output else ""
    else:
        # Long-form: manual chunking with overlap
        chunk_samples = int(CHUNK_SEC * sample_rate)
        overlap_samples = int(OVERLAP_SEC * sample_rate)
        step = chunk_samples - overlap_samples
        transcriptions = []
        n_chunks = max(1, (len(audio) - overlap_samples + step - 1) // step)

        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Save chunk to temp file (NeMo expects file paths)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import torchaudio as ta
                ta.save(f.name, torch.from_numpy(chunk).unsqueeze(0), sample_rate)
                chunk_path = f.name

            try:
                output = model.transcribe([chunk_path], batch_size=1, pnc='yes')
                text = output[0].text if output else ""
                transcriptions.append(text)
                print(f"    Chunk {i+1}/{n_chunks}: {len(text)} chars")
            finally:
                import os
                os.unlink(chunk_path)

        transcription = " ".join(transcriptions)

    elapsed = time.time() - t1
    if isinstance(transcription, (list, tuple)):
        transcription = " ".join(str(t) for t in transcription)

    print(f"  Inference: {elapsed:.1f}s | Transcript: {len(transcription)} chars")
    return transcription.strip(), elapsed


def run_benchmark(audio_path: str, gt_path: str, normalize: bool = False) -> dict:
    with open(gt_path) as f:
        gt = json.load(f)
    ref = gt["transcript"]

    hyp_raw, runtime = transcribe_canary(audio_path)
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
    print(f"  NVIDIA CANARY BENCHMARK — {r['model']}")
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
    out_path = out_dir / "benchmark_canary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
