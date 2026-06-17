"""
models_registry.py — Registry of ASR models for transcription and comparison.
Loads and runs models on GPU (Colab) or CPU.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import gc

import numpy as np
import torch


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Whisper (faster-whisper)
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_whisper(audio_path: str | Path, model_size: str = "turbo") -> str:
    """Transcribe audio using faster-whisper, passing the file path directly.

    Avoids process_audio_file / SileroVAD to prevent torch.hub.load conflicts
    when called from a background thread. Uses faster-whisper's built-in VAD.
    """
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Loading faster-whisper %s on %s...", model_size, device)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language=None,
            vad_filter=True,
            beam_size=3,
            task="transcribe",
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    finally:
        # unload_model() releases the CTranslate2 CUDA allocation.
        # del + empty_cache() alone does NOT free CTranslate2's own VRAM pool.
        try:
            model.model.unload_model()
        except Exception:
            pass
        del model
        _free_cuda()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Nvidia Parakeet
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_parakeet(audio_path: str | Path) -> str:
    """Transcribe audio using nvidia/parakeet-tdt-0.6b-v3 via Transformers."""
    import torchaudio
    from transformers import AutoModelForTDT, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading Parakeet TDT 0.6B on %s...", device)
    processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = AutoModelForTDT.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    wf, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    audio_arr = wf.squeeze().numpy().astype(np.float32)

    inputs = processor(audio_arr, sampling_rate=16000, return_tensors="pt", padding="longest")
    inputs = inputs.to(device, dtype=model.dtype)

    try:
        with torch.no_grad():
            out = model.generate(**inputs, return_dict_in_generate=True)
        text = processor.decode(out.sequences, skip_special_tokens=True)

        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return text.strip()
    finally:
        del model, processor
        _free_cuda()


# ═══════════════════════════════════════════════════════════════════════════
# Nvidia Canary
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_canary(audio_path: str | Path) -> str:
    """Transcribe audio using nvidia/canary-1b-v2 via NeMo."""
    import torchaudio
    from faster_whisper import WhisperModel
    from nemo.collections.asr.models import ASRModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("Canary requires CUDA. Running on CPU may be slow/fail.")

    logger.info("Loading Canary 1B V2 on %s...", device)
    model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    model = model.to(device).eval()

    # Whisper tiny for fast language detection per chunk
    whisper = WhisperModel("tiny", device=device, compute_type="int8")

    wf, sr = torchaudio.load(str(audio_path))
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

    transcriptions = []
    try:
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
                output = model.transcribe([chunk_path], source_lang=lang, target_lang=lang)
                text = output[0].text if output else ""
                transcriptions.append(text)
            finally:
                os.unlink(chunk_path)
    finally:
        del model, whisper
        _free_cuda()

    return " ".join(transcriptions).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Qwen-Audio
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_qwen(audio_path: str | Path) -> str:
    """Transcribe audio using Qwen/Qwen2-Audio-7B-Instruct via Transformers."""
    import torchaudio
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading Qwen2-Audio-7B on %s...", device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    ).eval()

    wf, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        torchaudio.save(tf.name, wf, 16000)
        temp_wav = tf.name

    try:
        prompt = "<|audio_preview|><|assistant|>Transcribe the audio:"
        inputs = processor(text=prompt, audios=temp_wav, return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_length=512)

        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response.strip()
    finally:
        os.unlink(temp_wav)
        del model, processor
        _free_cuda()


# ═══════════════════════════════════════════════════════════════════════════
# Nemotron
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_nemotron(audio_path: str | Path) -> str:
    """Transcribe audio using nvidia/stt_en_conformer_transducer_large_nemotron via NeMo."""
    from nemo.collections.asr.models import ASRModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Nemotron Large on %s...", device)
    model = ASRModel.from_pretrained("nvidia/stt_en_conformer_transducer_large_nemotron")
    model = model.to(device).eval()

    try:
        output = model.transcribe([str(audio_path)])
        if isinstance(output, list) and len(output) > 0:
            return output[0] if isinstance(output[0], str) else getattr(output[0], 'text', str(output[0]))
        return str(output).strip()
    finally:
        del model
        _free_cuda()


# ═══════════════════════════════════════════════════════════════════════════
# Model Runner
# ═══════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "whisper-turbo": lambda p: transcribe_whisper(p, "turbo"),
    "whisper-large-v3": lambda p: transcribe_whisper(p, "large-v3"),
    "parakeet": transcribe_parakeet,
    "canary": transcribe_canary,
    "qwen": transcribe_qwen,
    "nemotron": transcribe_nemotron,
}

def transcribe_with_model(model_name: str, audio_path: str | Path) -> str:
    """Transcribe an audio file using the specified model name."""
    fn = AVAILABLE_MODELS.get(model_name.lower())
    if not fn:
        raise ValueError(f"Model '{model_name}' is not supported. Supported: {list(AVAILABLE_MODELS.keys())}")
    return fn(audio_path)
