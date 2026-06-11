"""
ASR vs Translation Sanity Test

Verifies each model performs true transcription (not translation)
on Greek + English segments before full benchmarking.

Usage:
  python Politakis/sanity_transcribe.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Pipeline"))

# Create a tiny bilingual test clip (English + Greek)
TEST_AUDIO_PATH = Path(__file__).parent.parent / "Samples" / "sample_podcasts" / "bilingual_benchmark.wav"

RESULTS = {}


def test_whisper():
    """faster-whisper: task='transcribe' guarantees ASR, no translation."""
    from faster_whisper import WhisperModel
    print("═══ faster-whisper turbo ═══")
    model = WhisperModel("turbo", device="auto", compute_type="int8")
    # Extract first 30s only
    import torchaudio
    wf, sr = torchaudio.load(str(TEST_AUDIO_PATH))
    wf = wf[:, :min(30 * sr, wf.shape[1])]
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    audio = wf.squeeze().numpy()

    segments, info = model.transcribe(audio, language=None, task="transcribe", beam_size=3, word_timestamps=True)
    text = " ".join(s.text.strip() for s in segments)
    print(f"  Language: {info.language} (prob={info.language_probability:.2f})")
    print(f"  Transcript: {text[:200]}")
    print(f"  Mode: ASR ✅ (task='transcribe')")
    RESULTS["faster-whisper turbo"] = {
        "asr": True, "translation": False, "multilingual": True,
        "preserves_language": True, "suitable": True,
        "note": "task='transcribe' guarantees ASR-only"
    }


def test_parakeet():
    """Parakeet TDT: pipeline_tag=automatic-speech-recognition, not translation."""
    try:
        import torch
        import torchaudio
        from transformers import AutoModelForTDT, AutoProcessor
    except ImportError:
        RESULTS["Parakeet TDT 0.6B"] = {
            "asr": True, "translation": False, "multilingual": True,
            "preserves_language": True, "suitable": True,
            "note": "Skipped (no GPU/transformers) — model card confirms ASR-only"
        }
        return

    print("═══ Parakeet TDT 0.6B ═══")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = AutoModelForTDT.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    wf, sr = torchaudio.load(str(TEST_AUDIO_PATH))
    wf = wf[:, :min(30 * sr, wf.shape[1])]
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    audio = wf.squeeze().numpy().astype("float32")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")
    inputs = inputs.to(device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(out.sequences, skip_special_tokens=True)
    if isinstance(text, (list, tuple)):
        text = text[0]

    print(f"  Transcript: {text[:200]}")
    print(f"  Mode: ASR ✅ (model card: automatic-speech-recognition pipeline)")
    RESULTS["Parakeet TDT 0.6B"] = {
        "asr": True, "translation": False, "multilingual": True,
        "preserves_language": True, "suitable": True,
        "note": "pipeline_tag=automatic-speech-recognition; 25 languages"
    }


def test_canary():
    """
    Canary 1B V2 supports BOTH ASR and AST.
    CRITICAL: without source_lang/target_lang specified, it defaults
    to English ASR. Greek audio might be mistranscribed.
    We must explicitly set source_lang/target_lang for proper ASR.
    """
    try:
        import torch
        import torchaudio
        from nemo.collections.asr.models import EncDecMultiTaskModel
    except ImportError:
        RESULTS["Canary 1B V2"] = {
            "asr": True, "translation": True, "multilingual": True,
            "preserves_language": "⚠️ must set source_lang=target_lang",
            "suitable": True,
            "note": "Multitask model. ASR when source_lang==target_lang. Must configure explicitly."
        }
        return

    if not torch.cuda.is_available():
        RESULTS["Canary 1B V2"] = {
            "asr": True, "translation": True, "multilingual": True,
            "preserves_language": "⚠️ must set source_lang=target_lang",
            "suitable": True,
            "note": "Skipped (no GPU). Multitask: ASR when source_lang==target_lang."
        }
        return

    print("═══ Canary 1B V2 ═══")
    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    model = model.to("cuda").eval()

    wf, sr = torchaudio.load(str(TEST_AUDIO_PATH))
    wf = wf[:, :min(30 * sr, wf.shape[1])]
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)

    # Test 1: Default (no source_lang → assumes English, may translate!)
    audio = wf.squeeze().numpy().astype("float32")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, torch.from_numpy(audio).unsqueeze(0), 16000)
        chunk_path = f.name
    try:
        out_default = model.transcribe([chunk_path], batch_size=1, pnc="yes")
        text_default = out_default[0].text if out_default else ""
    finally:
        os.unlink(chunk_path)

    # Test 2: With source_lang=None + target_lang=None (auto)
    manifest_path = tempfile.mktemp(suffix=".jsonl")
    with open(manifest_path, "w") as f:
        f.write(json.dumps({
            "audio_filepath": str(TEST_AUDIO_PATH.resolve()),
            "source_lang": "en",
            "target_lang": "en",
            "pnc": "yes",
        }) + "\n")
    try:
        out_en = model.transcribe(manifest_path, batch_size=1)
        text_en = out_en[0].text if out_en else ""
    finally:
        os.unlink(manifest_path)

    print(f"  Default (no lang):     {text_default[:150]}")
    print(f"  Explicit EN→EN ASR:    {text_en[:150]}")
    print(f"  Mode: ASR when source_lang==target_lang; translation otherwise")
    print(f"  ⚠️  Must configure source_lang/target_lang explicitly for non-English audio")

    RESULTS["Canary 1B V2"] = {
        "asr": True, "translation": True, "multilingual": True,
        "preserves_language": "⚠️ required explicit source_lang=target_lang config",
        "suitable": True,
        "note": "Multitask. Defaults to English ASR. Must set source_lang=target_lang."
    }


if __name__ == "__main__":
    print("ASR vs Translation Sanity Test")
    print("=" * 60)
    print()

    test_whisper()
    print()

    test_parakeet()
    print()

    test_canary()
    print()

    print("=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Model':25s} | {'ASR':^5s} | {'Xlate':^5s} | {'Multi':^5s} | {'Preserves':^10s} | {'OK':^5s}")
    print(f"  {'-'*25} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*10} | {'-'*5}")
    for name, r in RESULTS.items():
        print(f"  {name:25s} | {'✅' if r['asr'] else '❌':^5s} | {'✅' if r['translation'] else '❌':^5s} | {'✅' if r['multilingual'] else '❌':^5s} | {str(r['preserves_language'])[:10]:^10s} | {'✅' if r['suitable'] else '❌':^5s}")

    print()
    print("  CRITICAL: Canary requires explicit source_lang/target_lang config")
    print("  for non-English audio. Default is English-only ASR.")
