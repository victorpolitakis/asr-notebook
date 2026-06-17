"""
podcast_pipeline.py — Full podcast generation pipeline. Runs on Colab GPU.

Stages:
  1. Script generation: LLM creates a 2-speaker dialogue from transcript/summary
  2. TTS synthesis: per-speaker voice synthesis with configurable models
  3. Audio concatenation: merge segments with 300ms silence between turns

Model loading strategy (T4 GPU, ~16GB VRAM):
  - Models loaded lazily — only what the user selected
  - Sequential loading for comparison (load → synth → unload → next)
  - Drive caching: only Kokoro (~2GB) and F5-TTS (~4GB) cached to Drive.
    Dia (10GB) and Bark (8GB) re-download from HF CDN each session —
    Drive writes at 1-5 MB/s would take 30-90 min for these.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
import drive_bridge as db

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Audio constants
# ═══════════════════════════════════════════════════════════════════

SILENCE_SEC: float = 0.3
MP3_BITRATE: str = "128k"
WORDS_PER_MIN: int = 150
T4_VRAM_GB: float = 16.0

# ═══════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    "dia": {
        "name": "Dia-1.6B",
        "vram_gb": 10.0,
        "multi_speaker": True,
        "requires_hf": True,
        "cache_to_drive": False,
    },
    "kokoro": {
        "name": "Kokoro-82M",
        "vram_gb": 2.0,
        "multi_speaker": False,
        "requires_hf": True,
        "cache_to_drive": True,
        "pip_package": "kokoro",
    },
    "bark": {
        "name": "Bark",
        "vram_gb": 8.0,
        "multi_speaker": False,
        "requires_hf": True,
        "cache_to_drive": False,
    },
    "xtts_v2": {
        "name": "XTTS-v2",
        "vram_gb": 6.0,
        "multi_speaker": False,
        "requires_hf": False,
        "cache_to_drive": False,
        "pip_package": "TTS",
        "license_note": "CPML — non-commercial use only (this project qualifies)",
    },
    "f5_tts": {
        "name": "F5-TTS",
        "vram_gb": 4.0,
        "multi_speaker": False,
        "requires_hf": True,
        "cache_to_drive": True,
    },
}


def estimate_vram_gb(model_a: str, model_b: str) -> float:
    """Estimate total VRAM needed for two models (worst case if both loaded)."""
    info_a = MODEL_REGISTRY.get(model_a, {})
    info_b = MODEL_REGISTRY.get(model_b, {})
    return info_a.get("vram_gb", 8.0) + info_b.get("vram_gb", 8.0)


# ═══════════════════════════════════════════════════════════════════
# Script generation
# ═══════════════════════════════════════════════════════════════════

SCRIPT_SYSTEM_PROMPT = """\
You are a professional podcast scriptwriter. Given source material and
episode configuration, generate a natural 2-speaker dialogue script.

Format rules:
- Each line MUST start with exactly "Speaker A: " or "Speaker B: "
- Alternate turns between speakers
- Include natural conversation elements: interruptions, filler words
  ("hmm", "right"), pauses, follow-up questions
- NOT a robotic Q&A — this should sound like real people talking

Style: {tone}
Target duration: approximately {duration_words} words (~{duration_minutes} minutes)
Speaker A ({speaker_a_name}): {speaker_a_desc}
Speaker B ({speaker_b_name}): {speaker_b_desc}

Source material:
{source_text}

Generate the full dialogue script now. Start with Speaker A:"""


def _generate_script(job_config: dict) -> str:
    """Generate a dialogue script from the podcast config using the LLM."""
    tone = job_config.get("config", {}).get("tone", "casual")
    length = job_config.get("config", {}).get("length", "medium")
    speaker_a = job_config.get("speaker_a", {})
    speaker_b = job_config.get("speaker_b", {})
    source_text = job_config.get("source_text", "")

    if not source_text.strip():
        raise ValueError("No source text provided for script generation")

    duration_minutes = {"short": 3, "medium": 7, "long": 15}.get(length, 7)
    duration_words = duration_minutes * WORDS_PER_MIN

    prompt = SCRIPT_SYSTEM_PROMPT.format(
        tone=tone,
        duration_words=duration_words,
        duration_minutes=duration_minutes,
        speaker_a_name=speaker_a.get("name", "Alex"),
        speaker_a_desc=speaker_a.get("description", "curious interviewer"),
        speaker_b_name=speaker_b.get("name", "Sam"),
        speaker_b_desc=speaker_b.get("description", "domain expert"),
        source_text=source_text[:12000],
    )

    logger.info("Generating podcast script: ~%d words, tone=%s", duration_words, tone)

    # Use requests directly — the openai SDK's "sync" client still calls
    # asyncio internals which fail in Colab's always-running kernel loop.
    import requests as _req
    _base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    _resp = _req.post(
        f"{_base}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        json={
            "model":       os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "temperature": 0.7,
            "max_tokens":  2000,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Generate the dialogue script."},
            ],
        },
        timeout=120,
    )
    _resp.raise_for_status()
    raw = _resp.json()["choices"][0]["message"]["content"] or ""

    script = raw.strip() if raw else ""
    if not script:
        raise ValueError("LLM returned an empty script")

    # Validate at least some lines start with expected prefixes
    lines = [l for l in script.splitlines() if l.strip()]
    prefixed = sum(1 for l in lines if l.startswith("Speaker A:") or l.startswith("Speaker B:"))
    total = len(lines)
    if total == 0 or prefixed / total < 0.7:
        logger.warning(
            "Script format validation weak: %d/%d lines have expected prefix. "
            "Raw output may not be parseable.", prefixed, total,
        )

    word_count = len(script.split())
    logger.info("Script generated: %d words, %d lines", word_count, total)
    return script


# ═══════════════════════════════════════════════════════════════════
# Script parsing
# ═══════════════════════════════════════════════════════════════════

def _parse_script(script: str) -> list[dict]:
    """Parse a dialogue script into per-segment entries.

    Returns list of {speaker, text} dicts in speaking order.
    """
    segments: list[dict] = []
    current_speaker: str | None = None
    current_text: list[str] = []

    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for speaker prefix
        match_a = re.match(r"Speaker A:\s*(.*)", line, re.IGNORECASE)
        match_b = re.match(r"Speaker B:\s*(.*)", line, re.IGNORECASE)

        if match_a or match_b:
            # Flush previous segment
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                })
            current_speaker = "A" if match_a else "B"
            text = (match_a or match_b).group(1)
            current_text = [text] if text.strip() else []
        else:
            # Continuation of current speaker's turn
            if current_speaker:
                current_text.append(line)

    if current_speaker and current_text:
        segments.append({
            "speaker": current_speaker,
            "text": " ".join(current_text),
        })

    logger.info("Parsed script: %d segments (A=%d, B=%d)",
                len(segments),
                sum(1 for s in segments if s["speaker"] == "A"),
                sum(1 for s in segments if s["speaker"] == "B"))
    return segments


# ═══════════════════════════════════════════════════════════════════
# Drive caching helpers
# ═══════════════════════════════════════════════════════════════════

def _get_cache_dir(model_key: str) -> Path:
    """Return local cache path for a model, restoring from Drive if available."""
    cache_dir = Path(tempfile.gettempdir()) / "tts_models" / model_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = MODEL_REGISTRY.get(model_key, {})
    if not info.get("cache_to_drive"):
        return cache_dir

    drive_model_files = db.list_files(f"{config.DRIVE_MODELS_CACHE}/{model_key}")
    if drive_model_files:
        logger.info("Restoring %s from Drive cache (%d files)...", model_key, len(drive_model_files))
        for f in drive_model_files:
            local_path = cache_dir / f["name"]
            if not local_path.exists():
                db.download_file(f["id"], str(local_path))

    return cache_dir


def _save_to_drive_cache(model_key: str, cache_dir: Path) -> None:
    """Upload local model cache to Drive for future Colab sessions."""
    info = MODEL_REGISTRY.get(model_key, {})
    if not info.get("cache_to_drive"):
        return

    drive_cache = f"{config.DRIVE_MODELS_CACHE}/{model_key}"
    try:
        db.get_or_create_folder(drive_cache)
        for f in cache_dir.iterdir():
            if f.is_file():
                db.upload_file(str(f), drive_cache)
        logger.info("Saved %s to Drive cache", model_key)
    except Exception as e:
        logger.warning("Failed to save %s to Drive cache: %s — continuing", model_key, e)


# ═══════════════════════════════════════════════════════════════════
# Abstract TTS model
# ═══════════════════════════════════════════════════════════════════

class _BaseTTSModel:
    """Lazy-loading base for all TTS backends."""
    model_key: str = ""
    sample_rate: int = 24000

    def load(self) -> None:
        raise NotImplementedError

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        raise NotImplementedError

    def unload(self) -> None:
        import gc
        import torch
        del self._model
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Unloaded %s", self.model_key)


# ═══════════════════════════════════════════════════════════════════
# Kokoro
# ═══════════════════════════════════════════════════════════════════

class KokoroModel(_BaseTTSModel):
    """hexgrad/Kokoro-82M — 82M params, blazing fast, 54 voices"""
    model_key = "kokoro"
    sample_rate = 24000
    _loaded: bool = False
    _model: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from kokoro import KPipeline
        except ImportError:
            raise RuntimeError(
                "Kokoro not installed. Run: pip install kokoro>=0.9.4"
            )

        import torch
        cache_dir = _get_cache_dir("kokoro")
        self._model = KPipeline(lang_code="a")
        self._loaded = True
        logger.info("Kokoro loaded (lang=en, ~2GB VRAM)")

        # Cache to Drive after first load
        try:
            hf_cache = Path.home() / ".cache" / "huggingface"
            if hf_cache.exists():
                _save_to_drive_cache("kokoro", hf_cache)
        except Exception:
            pass

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        self.load()
        voice = voice or "af_heart"
        segments: list[np.ndarray] = []

        for _, _, audio in self._model(text, voice=voice, speed=1):
            segments.append(audio)

        if not segments:
            return np.array([], dtype=np.float32)
        return np.concatenate(segments).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Dia-1.6B
# ═══════════════════════════════════════════════════════════════════

class DiaModel(_BaseTTSModel):
    """nari-labs/Dia-1.6B — native multi-speaker dialogue synthesis"""
    model_key = "dia"
    sample_rate = 44100
    _loaded: bool = False
    _model: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from dia.model import Dia
        except ImportError:
            raise RuntimeError(
                "Dia not installed. Run: pip install git+https://github.com/nari-labs/dia.git"
            )

        import torch
        self._model = Dia.from_pretrained("nari-labs/Dia-1.6B")
        self._loaded = True
        logger.info("Dia-1.6B loaded (~10GB VRAM)")

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        # For single-speaker, just pass text directly
        self.load()
        output = self._model.generate(text)
        # output is already a numpy array at 44100 Hz
        return np.array(output, dtype=np.float32)

    def synthesize_multi_speaker(self, script: str) -> np.ndarray:
        """Native multi-speaker: convert Speaker tags to [S1]/[S2] and generate in one pass."""
        self.load()
        # Convert from our format to Dia's native format
        converted = script.replace("Speaker A:", "[S1]").replace("Speaker B:", "[S2]")
        logger.info("Dia multi-speaker: %d chars converted", len(converted))
        output = self._model.generate(converted)
        return np.array(output, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Bark
# ═══════════════════════════════════════════════════════════════════

class BarkModel(_BaseTTSModel):
    """suno-ai/bark — expressive with non-verbal sounds"""
    model_key = "bark"
    sample_rate = 24000
    _loaded: bool = False
    _model: Any = None

    def load(self) -> None:
        if self._loaded:
            return

        # Must be set BEFORE any bark import — else OOM on T4
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"

        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
        except ImportError:
            raise RuntimeError(
                "Bark not installed. Run: pip install git+https://github.com/suno-ai/bark.git"
            )

        preload_models()
        self._generate_fn = generate_audio
        self.sample_rate = SAMPLE_RATE
        self._loaded = True
        logger.info("Bark loaded (small models mode, ~8GB VRAM)")

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        self.load()
        # voice = history_prompt for bark, e.g. "v2/en_speaker_6"
        kwargs = {}
        if voice:
            kwargs["history_prompt"] = voice
        audio = self._generate_fn(text, **kwargs)
        return np.array(audio, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# XTTS-v2
# ═══════════════════════════════════════════════════════════════════

class XTTSv2Model(_BaseTTSModel):
    """coqui-ai/TTS — zero-shot voice cloning, multi-language"""
    model_key = "xtts_v2"
    sample_rate = 24000
    _loaded: bool = False
    _model: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("XTTS-v2 not installed. Run: pip install TTS")

        self._model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        self._loaded = True
        logger.info("XTTS-v2 loaded (~6GB VRAM) — CPML license, non-commercial only")

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        self.load()
        # XTTS requires a speaker reference or waveform; use built-in voice for simplicity
        kwargs = {"text": text, "language": "en"}
        output = self._model.tts(**kwargs)
        return np.array(output, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# F5-TTS
# ═══════════════════════════════════════════════════════════════════

class F5TTSModel(_BaseTTSModel):
    """SWivid/F5-TTS — fast, strong voice cloning"""
    model_key = "f5_tts"
    sample_rate = 24000
    _loaded: bool = False
    _model: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_vocoder, preprocess_ref_audio_text, \
                load_checkpoint
        except ImportError:
            raise RuntimeError(
                "F5-TTS not installed. Run: pip install git+https://github.com/SWivid/F5-TTS.git"
            )

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = _get_cache_dir("f5_tts")

        # Load model
        self._model = DiT(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        ckpt_path = self._download_checkpoint()
        self._model = load_checkpoint(self._model, ckpt_path, self._device)
        self._vocoder = load_vocoder()

        self._loaded = True
        logger.info("F5-TTS loaded (~4GB VRAM)")

        # Cache to Drive
        if cache_dir.exists():
            _save_to_drive_cache("f5_tts", cache_dir)

    def _download_checkpoint(self) -> str:
        """Download F5-TTS checkpoint from HF if not cached."""
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "SWivid/F5-TTS",
            "F5TTS_Base/model_1200000.pt",
            cache_dir=str(Path(tempfile.gettempdir()) / "f5_tts_hf"),
        )
        return path

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        self.load()
        import torch
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Simplified generation — full implementation would handle ref audio
        # For now, generate with neutral conditioning
        from f5_tts.infer.utils_infer import infer_process
        gen_text = text[:500]  # limit for speed
        audio, _ = infer_process(
            self._model, self._vocoder, gen_text,
            ref_text="", device=self._device,
        )

        sf.write(tmp_path, audio, self.sample_rate)
        audio_arr, _ = sf.read(tmp_path)
        os.unlink(tmp_path)
        return audio_arr.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════

_MODEL_CLASSES: dict[str, type[_BaseTTSModel]] = {
    "dia": DiaModel,
    "kokoro": KokoroModel,
    "bark": BarkModel,
    "xtts_v2": XTTSv2Model,
    "f5_tts": F5TTSModel,
}

_LOADED_MODELS: dict[str, _BaseTTSModel] = {}


def _get_model(model_key: str) -> _BaseTTSModel:
    """Lazy-load and cache a TTS model in memory."""
    if model_key not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown TTS model: {model_key}. "
            f"Available: {list(_MODEL_CLASSES.keys())}"
        )
    if model_key not in _LOADED_MODELS:
        model = _MODEL_CLASSES[model_key]()
        model.load()
        _LOADED_MODELS[model_key] = model
    return _LOADED_MODELS[model_key]


def _unload_all() -> None:
    """Unload all cached models to free VRAM."""
    for key, model in list(_LOADED_MODELS.items()):
        try:
            model.unload()
        except Exception:
            pass
        del _LOADED_MODELS[key]
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("All TTS models unloaded — VRAM freed")


# ═══════════════════════════════════════════════════════════════════
# Audio concatenation
# ═══════════════════════════════════════════════════════════════════

def _concat_segments(
    audio_segments: list[np.ndarray],
    output_path: str,
    sample_rate: int = 24000,
) -> str:
    """Concatenate audio segments with silence between them. Export MP3."""
    if not audio_segments:
        raise ValueError("No audio segments to concatenate")

    silence_samples = int(SILENCE_SEC * sample_rate)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    for seg in audio_segments:
        parts.append(seg.astype(np.float32))
        parts.append(silence.copy())

    # Remove trailing silence
    combined = np.concatenate(parts[:-1])

    # Write WAV first, then convert to MP3 via pydub
    from pydub import AudioSegment
    import scipy.io.wavfile as wavfile

    wav_path = output_path.replace(".mp3", ".wav")
    wavfile.write(wav_path, sample_rate, (combined * 32767).astype(np.int16))
    audio_seg = AudioSegment.from_wav(wav_path)
    audio_seg.export(output_path, format="mp3", bitrate=MP3_BITRATE)
    os.unlink(wav_path)

    duration = len(combined) / sample_rate
    logger.info("Exported MP3: %.1fs, %d segments → %s", duration, len(audio_segments), output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def generate_podcast(job_config: dict) -> dict:
    """
    Full podcast generation pipeline.

    Args:
        job_config: PodcastJobDict from Drive input/podcast_jobs/{id}.json

    Returns:
        dict with keys: mp3_path, duration_sec, model_info, word_count
    """
    job_id = job_config.get("job_id", "unknown")
    speaker_a = job_config.get("speaker_a", {})
    speaker_b = job_config.get("speaker_b", {})
    model_a = speaker_a.get("tts_model", "kokoro")
    model_b = speaker_b.get("tts_model", "kokoro")

    logger.info("🎙️ Podcast job %s: A=%s, B=%s", job_id, model_a, model_b)

    # ── 1. Script generation ──
    script = job_config.get("source_text", "")
    if not script or not any(l.startswith("Speaker A:") or l.startswith("Speaker B:") for l in script.splitlines()):
        logger.info("Generating script from source material...")
        script = _generate_script(job_config)

    # ── 2. Parse script into segments ──
    segments = _parse_script(script)
    if not segments:
        raise ValueError("Script parsing produced zero segments")

    # ── 3. TTS synthesis ──
    output_dir = Path(tempfile.mkdtemp(prefix="podcast_"))

    # Check if both speakers use Dia — native multi-speaker mode
    if model_a == "dia" and model_b == "dia":
        logger.info("Both speakers use Dia — native multi-speaker single pass")
        dia = _get_model("dia")
        # Convert entire script to Dia format
        dia_script = script.replace("Speaker A:", "[S1]").replace("Speaker B:", "[S2]")
        audio = dia.synthesize_multi_speaker(dia_script)
        audio_segments = [audio]
        model_info = {"speaker_a": "Dia-1.6B", "speaker_b": "Dia-1.6B", "mode": "native_multi_speaker"}
        sample_rate = dia.sample_rate
    else:
        # Per-speaker, per-segment synthesis
        audio_segments = []
        model_info = {}

        model_a_obj = _get_model(model_a)
        model_b_obj = _get_model(model_b)
        sample_rate = model_a_obj.sample_rate  # Use speaker A's rate as reference

        # Audio sample rate may differ between models — resample if needed
        for seg in segments:
            model = model_a_obj if seg["speaker"] == "A" else model_b_obj
            voice = speaker_a.get("voice") if seg["speaker"] == "A" else speaker_b.get("voice")
            audio = model.synthesize(seg["text"], voice=voice)

            # Resample to match reference rate if needed
            if model.sample_rate != sample_rate:
                import scipy.signal
                ratio = sample_rate / model.sample_rate
                audio = scipy.signal.resample(audio, int(len(audio) * ratio))

            audio_segments.append(audio)

        model_info = {
            "speaker_a": MODEL_REGISTRY.get(model_a, {}).get("name", model_a),
            "speaker_b": MODEL_REGISTRY.get(model_b, {}).get("name", model_b),
            "mode": "per_segment",
        }

    # ── 4. Concatenate & export ──
    mp3_path = str(output_dir / f"{job_id}.mp3")
    _concat_segments(audio_segments, mp3_path, sample_rate)

    # ── 5. Unload models to free VRAM ──
    _unload_all()

    duration_sec = float(len(audio_segments[0]) / sample_rate) if audio_segments else 0
    word_count = len(script.split())

    return {
        "mp3_path":    mp3_path,
        "script_text": script,
        "duration_sec": round(duration_sec, 1),
        "model_info":  model_info,
        "word_count":  word_count,
    }


# ═══════════════════════════════════════════════════════════════════
# Comparison mode
# ═══════════════════════════════════════════════════════════════════

def compare_models(script_snippet: str, models: list[str]) -> dict[str, bytes]:
    """
    Synthesize the same short script with each model for A/B comparison.

    Runs sequentially — each model is loaded, used, then unloaded before
    the next to stay within T4 VRAM limits.

    Returns: dict of {model_name: wav_bytes}
    """
    results: dict[str, bytes] = {}
    script_snippet = script_snippet[:300]

    for model_key in models:
        if model_key not in _MODEL_CLASSES:
            logger.warning("Skipping unknown model: %s", model_key)
            continue

        name = MODEL_REGISTRY.get(model_key, {}).get("name", model_key)
        logger.info("Comparing %s...", name)

        try:
            model = _get_model(model_key)
            audio = model.synthesize(script_snippet)

            import io
            import scipy.io.wavfile as wavfile
            buf = io.BytesIO()
            wavfile.write(buf, model.sample_rate, (audio * 32767).astype(np.int16))
            buf.seek(0)
            results[name] = buf.read()

            # Unload before loading next model
            model.unload()
            _LOADED_MODELS.pop(model_key, None)

        except Exception as e:
            logger.error("Comparison failed for %s: %s", name, e)

    logger.info("Comparison done: %d/%d models succeeded", len(results), len(models))
    return results


# ═══════════════════════════════════════════════════════════════════
# Quick test (for development on Colab)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    test_config = {
        "job_id": "test_001",
        "source_text": (
            "Speaker A: Welcome to this test podcast about artificial intelligence.\n"
            "Speaker B: Thanks for having me. AI has come a long way in recent years.\n"
            "Speaker A: What do you think is the most exciting development?\n"
            "Speaker B: Large language models have been transformative for how we interact with technology."
        ),
        "speaker_a": {"name": "Alex", "description": "tech enthusiast", "tts_model": "kokoro", "voice": "af_heart"},
        "speaker_b": {"name": "Sam", "description": "AI researcher", "tts_model": "kokoro", "voice": "am_michael"},
        "config": {"tone": "casual", "length": "short"},
    }

    print("Running podcast pipeline test with Kokoro...")
    result = generate_podcast(test_config)
    print(f"Done! MP3: {result['mp3_path']} ({result['duration_sec']}s)")
