"""
Unified ASR Benchmark — Run all models and compare.

Models:
  - faster-whisper turbo (production baseline)
  - faster-whisper large-v3
  - NVIDIA Parakeet TDT 0.6B v3
  - NVIDIA Canary 1B V2

Each model transcribes (never translates). Optional --normalize applies
the production LLM cleanup layer to all raw transcripts for fair comparison.

Usage:
  python Politakis/benchmark_all.py <audio.wav> <ground_truth.json> [--normalize]
"""

import json
import os
import re
import sys
import time
import tempfile
from collections import OrderedDict
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Pipeline"))

# Auto-load OPENAI_API_KEY from: 1) env, 2) .env file, 3) Colab secrets
if not os.environ.get("OPENAI_API_KEY"):
    # Try .env
    for env_path in (Path(__file__).parent / ".env", Path.home() / ".env", Path(".env")):
        try:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        os.environ.setdefault(key, val)
        except Exception:
            pass
    # Try Colab secrets
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            from google.colab import userdata
            os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
        except Exception:
            pass

from evaluate import _jiwer_transform, normalize_for_eval


def _apply_normalization(raw_text: str) -> str:
    """Apply LLM cleanup (self-contained, no evaluate.py dependency)."""
    try:
        from transcript_normalizer import normalize_transcript, ENABLE_NORMALIZATION
        if not ENABLE_NORMALIZATION:
            return raw_text
        result = normalize_transcript(raw_text)
        return result if result is not None else raw_text
    except Exception:
        return raw_text


ENTITIES = [
    "Yann LeCun", "Fei-Fei Li", "Geoffrey Hinton", "Sam Altman",
    "Chin-Yew Lin", "Herve Bredin", "OpenAI", "Google DeepMind",
    "Apple Silicon", "Metal Performance Shaders", "GPT-4", "Gemini",
    "Claude", "Whisper", "NVIDIA", "Meta", "Anthropic",
    "MIT", "Khan Academy", "Duolingo", "NLP", "NER", "ROUGE",
]


# ═══════════════════════════════════════════════════════════════════════════
# Whisper (faster-whisper via pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_whisper(audio: str, gt: dict, model_size: str) -> dict:
    from faster_whisper import WhisperModel
    from audio_processor import process_audio_file

    ref = gt["transcript"]
    print(f"  Loading faster-whisper {model_size}...")
    model = WhisperModel(model_size, device="auto", compute_type="int8")

    t0 = time.time()
    all_texts = []
    for chunk in process_audio_file(audio, min_chunk_sec=28, max_chunk_sec=30):
        if not chunk.get("is_speech"):
            continue
        segments, info = model.transcribe(
            chunk["audio_data"], language=None, vad_filter=False,
            beam_size=3, word_timestamps=True, task="transcribe",
        )
        all_texts.append(" ".join(s.text.strip() for s in segments))

    runtime = time.time() - t0
    hyp_raw = " ".join(all_texts)
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", hyp_raw)
    return _evaluate(hyp_raw, ref, runtime, f"faster-whisper {model_size}", "int8")


# ═══════════════════════════════════════════════════════════════════════════
# Parakeet TDT 0.6B v3 (Transformers)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_parakeet(audio: str, gt: dict) -> dict:
    import torch
    import torchaudio
    from transformers import AutoModelForTDT, AutoProcessor

    ref = gt["transcript"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("  Loading Parakeet TDT 0.6B...")
    processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = AutoModelForTDT.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    wf, sr = torchaudio.load(audio)
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    audio_arr = wf.squeeze().numpy().astype(np.float32)

    inputs = processor(audio_arr, sampling_rate=16000, return_tensors="pt", padding="longest")
    inputs = inputs.to(device, dtype=model.dtype)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(out.sequences, skip_special_tokens=True)
    runtime = time.time() - t0

    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", text.strip())
    return _evaluate(hyp_raw, ref, runtime, "Parakeet TDT 0.6B", "f16/cuda")


# ═══════════════════════════════════════════════════════════════════════════
# Canary 1B V2 (NeMo — transcribe only, never translate)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_canary(audio: str, gt: dict) -> dict:
    """
    Canary 1B V2 — multitask ASR/AST model. 25 languages.
    API: model.transcribe([path], source_lang='el', target_lang='el')
    ASR when source_lang == target_lang. Auto-chunks long audio.
    Detects language per chunk via faster-whisper tiny.
    """
    import torch
    import torchaudio
    from faster_whisper import WhisperModel
    from nemo.collections.asr.models import ASRModel

    ref = gt["transcript"]
    device = "cuda"
    if not torch.cuda.is_available():
        print("  ⚠️  Canary requires CUDA — skipping.")
        return None

    print("  Loading Canary 1B V2...")
    model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    model = model.to(device).eval()

    print("  Loading faster-whisper tiny for language detection...")
    whisper = WhisperModel("tiny", device="auto", compute_type="int8")

    wf, sr = torchaudio.load(audio)
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    audio_arr = wf.squeeze().numpy().astype(np.float32)

    CHUNK, OVERLAP = 35, 2
    chunk_samples = int(CHUNK * 16000)
    overlap_samples = int(OVERLAP * 16000)
    step = chunk_samples - overlap_samples
    n_chunks = max(1, (len(audio_arr) - overlap_samples + step - 1) // step)

    t0 = time.time()
    transcriptions = []
    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_samples, len(audio_arr))
        chunk = audio_arr[start:end]

        det = whisper.detect_language(chunk)
        lang = (det[0] if isinstance(det, tuple) else "en") or "en"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as cf:
            torchaudio.save(cf.name, torch.from_numpy(chunk).unsqueeze(0), 16000)
            chunk_path = cf.name
        try:
            # source_lang == target_lang → ASR (never translates)
            output = model.transcribe([chunk_path], source_lang=lang, target_lang=lang)
            text = output[0].text if output else ""
            transcriptions.append(text)
            if i < 3:
                print(f"    Chunk {i+1}: lang={lang} → {text[:80]}...")
        finally:
            os.unlink(chunk_path)

    runtime = time.time() - t0
    hyp_raw = re.sub(r"\[Speaker [A-Z]\]:\s*", "", " ".join(transcriptions).strip())
    return _evaluate(hyp_raw, ref, runtime, "Canary 1B V2", "f32/cuda")


# ═══════════════════════════════════════════════════════════════════════════
# Shared evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate(hyp_raw: str, ref: str, runtime: float, model: str, backend: str) -> dict:
    transform = _jiwer_transform()
    wer_raw = round(jiwer.wer(ref, hyp_raw, reference_transform=transform, hypothesis_transform=transform), 4)

    norm_ref = normalize_for_eval(ref)
    norm_hyp_raw = normalize_for_eval(hyp_raw)
    norm_wer_raw = round(jiwer.wer(norm_ref, norm_hyp_raw), 4)

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge_raw = round(scorer.score(ref, hyp_raw)["rouge1"].fmeasure, 4)
    except Exception:
        rouge_raw = -1.0

    entity_raw = {}
    for e in ENTITIES:
        entity_raw[e] = e.lower() in hyp_raw.lower()
    found_raw = sum(1 for v in entity_raw.values() if v)

    return {
        "model": model,
        "backend": backend,
        "runtime": round(runtime, 1),
        "transcript_len": len(hyp_raw),
        "ref_len": len(ref),
        "wer": wer_raw,
        "norm_wer": norm_wer_raw,
        "rouge1": rouge_raw,
        "entities": found_raw,
        "entity_total": len(ENTITIES),
        "entity_detail": entity_raw,
        "hyp_raw": hyp_raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "Samples/sample_podcasts/bilingual_benchmark.wav"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "Samples/sample_podcasts/bilingual_benchmark_gt.json"
    normalize = "--normalize" in sys.argv

    with open(gt_path) as f:
        gt = json.load(f)

    print(f"Audio: {audio}")
    print(f"Ground truth: {gt_path}")
    print(f"Normalize: {normalize}")
    print()

    results_models = OrderedDict()
    results_norm = OrderedDict()

    # ── Whisper Turbo ──
    print("═══ Whisper Turbo ═══")
    r = benchmark_whisper(audio, gt, "turbo")
    results_models[r["model"]] = r
    if normalize:
        print("  Applying normalization...")
        norm_text = _apply_normalization(r["hyp_raw"])
        results_norm[r["model"]] = _evaluate(norm_text, gt["transcript"], r["runtime"], r["model"] + " +norm", r["backend"])

    # ── Whisper Large-v3 ──
    print("═══ Whisper Large-v3 ═══")
    r = benchmark_whisper(audio, gt, "large-v3")
    results_models[r["model"]] = r
    if normalize:
        norm_text = _apply_normalization(r["hyp_raw"])
        results_norm[r["model"]] = _evaluate(norm_text, gt["transcript"], r["runtime"], r["model"] + " +norm", r["backend"])

    # ── Parakeet ──
    print("═══ Parakeet TDT ═══")
    try:
        r = benchmark_parakeet(audio, gt)
        if r:
            results_models[r["model"]] = r
            if normalize:
                norm_text = _apply_normalization(r["hyp_raw"])
                results_norm[r["model"]] = _evaluate(norm_text, gt["transcript"], r["runtime"], r["model"] + " +norm", r["backend"])
    except Exception as e:
        print(f"  Parakeet skipped: {e}")

    # ── Canary ──
    print("═══ Canary 1B V2 ═══")
    try:
        r = benchmark_canary(audio, gt)
        if r:
            results_models[r["model"]] = r
            if normalize:
                norm_text = _apply_normalization(r["hyp_raw"])
                results_norm[r["model"]] = _evaluate(norm_text, gt["transcript"], r["runtime"], r["model"] + " +norm", r["backend"])
    except Exception as e:
        print(f"  Canary skipped: {e}")

    # ── Print raw comparison table ───────────────────────────────────────
    print()
    print("=" * 95)
    print("  RAW COMPARISON TABLE")
    print("=" * 95)
    hdr = f"  {'Model':28s} | {'Runtime':>7s} | {'WER':>7s} | {'nWER':>7s} | {'R1':>7s} | {'Entities':>9s}"
    print(hdr)
    print(f"  {'-'*28} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*9}")
    for r in results_models.values():
        print(f"  {r['model']:28s} | {r['runtime']:6.1f}s | {r['wer']:7.4f} | {r['norm_wer']:7.4f} | {r['rouge1']:7.4f} | {r['entities']:3d}/{r['entity_total']}")

    if results_norm:
        print()
        print("=" * 95)
        print("  NORMALIZED COMPARISON TABLE (LLM cleanup applied)")
        print("=" * 95)
        print(hdr)
        print(f"  {'-'*28} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*9}")
        for r in results_norm.values():
            print(f"  {r['model']:28s} | {r['runtime']:6.1f}s | {r['wer']:7.4f} | {r['norm_wer']:7.4f} | {r['rouge1']:7.4f} | {r['entities']:3d}/{r['entity_total']}")

    # ── Entity detail table ──────────────────────────────────────────────
    print()
    print("=" * 95)
    print("  ENTITY DETAIL MATRIX")
    print("=" * 95)
    models = list(results_models.keys())
    width = max(len(m) for m in models + ENTITIES) + 2
    print(f"  {'Entity':{width}s}", end="")
    for m in models:
        print(f" | {m[:16]:^16s}", end="")
    print()
    for e in ENTITIES:
        print(f"  {e:{width}s}", end="")
        for m in models:
            found = results_models[m]["entity_detail"].get(e, False)
            print(f" | {' ✅' if found else ' ❌':^16s}", end="")
        print()

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_all.json"
    payload = {"raw": list(results_models.values()), "normalized": list(results_norm.values()) if results_norm else []}
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}")
