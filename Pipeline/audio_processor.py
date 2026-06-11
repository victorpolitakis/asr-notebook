"""
audio_processor.py — Section 1: Audio Processing Pipeline (20 pts)

=============================================================================
OUTPUT JSON SCHEMA  (one dict per chunk emitted by process_audio_file)
=============================================================================
{
    "chunk_id":        int,        # Sequential index, 0-based. Example: 0
    "audio_data":      "np.ndarray (float32)",  # Raw PCM samples, mono, 16 kHz
    "sample_rate":     int,        # Always 16000
    "duration_sec":    float,      # Chunk length in seconds.  Example: 6.42
    "start_time_sec":  float,      # Absolute offset from file start. Example: 0.0
    "end_time_sec":    float,      # Absolute offset from file start. Example: 6.42
    "is_speech":       bool,       # True if VAD detected speech in this chunk
    "rms_db":          float,      # RMS loudness after normalization (dBFS). Example: -18.3
    "detected_language": null,     # Reserved — populated downstream by asr_pipeline.py
    "processing_time_sec": float   # Wall-clock time to produce this chunk. Example: 0.12
}
=============================================================================
"""

from __future__ import annotations

import io
import time
import logging
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants — all tuneable knobs live here for easy experimentation
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 16_000          # Whisper expects 16 kHz mono
TARGET_LOUDNESS_DBFS: float = -20.0       # Normalize every file to this RMS level
MIN_CHUNK_SEC: float = 25.0              # Target 25-30 s for whisper-large-v3-turbo receptive field
MAX_CHUNK_SEC: float = 30.0              # 30 s hard ceiling — model's max receptive field
SILENCE_THRESHOLD_SEC: float = 0.5       # VAD must see ≥0.5 s of silence to cut
VAD_WINDOW_SIZE_SAMPLES: int = 512       # Silero VAD operates on 512-sample frames @ 16 kHz


# ═══════════════════════════════════════════════════════════════════════════
# 1. Audio Loading & Normalization  (handles variable audio quality)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_normalize(source: str | Path) -> np.ndarray:
    """
    Load any audio file, normalise loudness, return float32 16 kHz mono.

    Tries pydub first (handles mp3, ogg, m4a via ffmpeg).
    Falls back to native WAV parsing if pydub fails (no ffmpeg needed).
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.info("Loading audio: %s", path.name)

    # ── Try pydub (covers all formats when ffmpeg is available) ──────
    try:
        audio = AudioSegment.from_file(str(path))
        loudness_delta = TARGET_LOUDNESS_DBFS - audio.dBFS
        audio = audio.apply_gain(loudness_delta)
        logger.info(
            "Normalized loudness: original %.1f dBFS → target %.1f dBFS (Δ %.1f dB)",
            audio.dBFS - loudness_delta, audio.dBFS, loudness_delta,
        )
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        raw_bytes = buf.read()
        samples = np.frombuffer(raw_bytes[44:], dtype=np.int16)
    except Exception:
        logger.warning("pydub load failed — trying native WAV parser (no ffmpeg needed).")
        # Native WAV fallback — no ffmpeg dependency
        samples = _load_wav_native(path)

    # Normalise to float32 [-1.0, 1.0]
    samples_f32: np.ndarray = samples.astype(np.float32) / 32768.0

    logger.info(
        "Loaded %.2f s of audio  (%d samples @ %d Hz)",
        len(samples_f32) / TARGET_SAMPLE_RATE,
        len(samples_f32),
        TARGET_SAMPLE_RATE,
    )
    return samples_f32


def _load_wav_native(path: Path) -> np.ndarray:
    """Load a 16-bit PCM WAV file using Python's built-in wave module."""
    import wave
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Only 16-bit WAV supported for native loading"
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Silero VAD wrapper  (real-time Voice Activity Detection)
# ═══════════════════════════════════════════════════════════════════════════

class SileroVAD:
    """
    Thin wrapper around the Silero VAD v5 model.

    Why Silero?
    - Runs on CPU in < 1 ms per frame → negligible overhead
    - Torch-based, so MPS is available if we ever need it
    - Much more robust than energy-based VAD for noisy podcasts
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Args:
            threshold: Speech probability above this → "speech detected".
                       0.5 is the Silero-recommended default.
        """
        # Load the official Silero VAD model from torch.hub
        self.model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.threshold = threshold
        # Silero VAD is stateful (hidden states); reset between files
        self.model.reset_states()
        logger.info("Silero VAD loaded (threshold=%.2f)", self.threshold)

    def reset(self) -> None:
        """Reset internal RNN hidden states between audio files."""
        self.model.reset_states()

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return the raw speech probability for a frame (useful for debugging)."""
        return self.model(torch.from_numpy(frame).float(), TARGET_SAMPLE_RATE).item()

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the frame's speech probability ≥ threshold."""
        return self.speech_probability(frame) >= self.threshold


# ═══════════════════════════════════════════════════════════════════════════
# 3. VAD-aware Chunker  (intelligent 5-10 s splitting)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rms_db(samples: np.ndarray) -> float:
    """
    Compute RMS loudness in dBFS for a chunk of float32 audio.
    Returns -inf for silence (all zeros).
    """
    rms = np.sqrt(np.mean(samples ** 2))
    if rms == 0:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def vad_chunker(
    audio: np.ndarray,
    vad: SileroVAD,
    *,
    min_chunk_sec: float = MIN_CHUNK_SEC,
    max_chunk_sec: float = MAX_CHUNK_SEC,
    silence_threshold_sec: float = SILENCE_THRESHOLD_SEC,
) -> Generator[dict, None, None]:
    """
    Split a full audio array into variable-length chunks using VAD.

    Strategy (from implementation plan):
    ─────────────────────────────────────
    "Once the chunk is >5 s, chop ONLY on the very next 0.5 s silence
     detected by VAD."  This guarantees we only feed complete sentences
     to Whisper, preventing hallucinations mid-word.

    If we hit the 10 s hard ceiling without finding silence, we force-cut
    to stay within the rubric window.

    Yields:
        dict matching the output JSON schema documented at the top of this file.
    """
    total_samples = len(audio)
    frame_size = VAD_WINDOW_SIZE_SAMPLES  # 512 samples = 32 ms @ 16 kHz
    min_samples = int(min_chunk_sec * TARGET_SAMPLE_RATE)
    max_samples = int(max_chunk_sec * TARGET_SAMPLE_RATE)
    silence_frames_needed = int(
        silence_threshold_sec * TARGET_SAMPLE_RATE / frame_size
    )

    chunk_id = 0
    pos = 0  # current read position in samples

    while pos < total_samples:
        t_start = time.perf_counter()

        chunk_start = pos
        consecutive_silence = 0
        cut_pos = None

        # Walk frame-by-frame from current position
        frame_pos = pos
        while frame_pos + frame_size <= total_samples:
            frame = audio[frame_pos : frame_pos + frame_size]
            elapsed_samples = frame_pos - chunk_start + frame_size

            speech = vad.is_speech(frame)

            if not speech:
                consecutive_silence += 1
            else:
                consecutive_silence = 0

            # --- Decision logic ---
            # If chunk ≥ min_chunk AND we just saw enough silence → cut here
            if (
                elapsed_samples >= min_samples
                and consecutive_silence >= silence_frames_needed
            ):
                cut_pos = frame_pos + frame_size
                break

            # Hard ceiling: force-cut at max_chunk_sec regardless
            if elapsed_samples >= max_samples:
                cut_pos = frame_pos + frame_size
                logger.debug(
                    "Chunk %d: force-cut at %.2f s (no silence found)",
                    chunk_id,
                    elapsed_samples / TARGET_SAMPLE_RATE,
                )
                break

            frame_pos += frame_size

        # If we ran out of audio before hitting any cut condition
        if cut_pos is None:
            cut_pos = total_samples

        chunk_audio = audio[chunk_start:cut_pos]
        duration = len(chunk_audio) / TARGET_SAMPLE_RATE

        # Skip extremely short trailing fragments (< 0.3 s) — likely silence
        if duration < 0.3:
            pos = cut_pos
            continue

        # Check if this chunk actually contains speech
        # (quick scan: if >30 % of frames are speech → mark as speech)
        n_frames = max(1, len(chunk_audio) // frame_size)
        speech_frame_count = sum(
            1 for i in range(0, len(chunk_audio) - frame_size + 1, frame_size)
            if vad.is_speech(chunk_audio[i : i + frame_size])
        )
        has_speech = (speech_frame_count / n_frames) >= 0.30

        t_end = time.perf_counter()

        chunk_dict: dict = {
            "chunk_id": chunk_id,
            "audio_data": chunk_audio,
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_sec": round(duration, 4),
            "start_time_sec": round(chunk_start / TARGET_SAMPLE_RATE, 4),
            "end_time_sec": round(cut_pos / TARGET_SAMPLE_RATE, 4),
            "is_speech": has_speech,
            "rms_db": round(_compute_rms_db(chunk_audio), 2),
            "detected_language": None,       # Populated by asr_pipeline.py
            "processing_time_sec": round(t_end - t_start, 4),
        }

        logger.info(
            "Chunk %d: %.2f–%.2f s  (%.2f s)  speech=%s  rms=%.1f dB  [%.3f s proc]",
            chunk_id,
            chunk_dict["start_time_sec"],
            chunk_dict["end_time_sec"],
            duration,
            has_speech,
            chunk_dict["rms_db"],
            chunk_dict["processing_time_sec"],
        )

        yield chunk_dict

        chunk_id += 1
        pos = cut_pos

    # Reset VAD states so the model is clean for the next file
    vad.reset()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Public API  —  single entry-point for downstream consumers
# ═══════════════════════════════════════════════════════════════════════════

def process_audio_file(
    file_path: str | Path,
    *,
    vad_threshold: float = 0.5,
    min_chunk_sec: float = MIN_CHUNK_SEC,
    max_chunk_sec: float = MAX_CHUNK_SEC,
) -> Generator[dict, None, None]:
    """
    End-to-end audio processing pipeline.

    1. Load & normalise any audio format  (handles variable quality)
    2. Initialise Silero VAD               (real-time voice activity detection)
    3. Yield chunks via VAD-aware splitter  (5-10 s, cut on silence)

    This function is a **generator** — it lazily yields one chunk dict at a
    time so that asr_pipeline.py can transcribe each chunk as it arrives,
    simulating stream-based / real-time processing without buffering the
    entire file.

    No language hints are passed here; Whisper's zero-shot detection handles
    multilingual / code-switching scenarios (Greek + English "Greeklish").

    Args:
        file_path:      Path to any audio file (mp3, wav, ogg, m4a …).
        vad_threshold:  Silero speech-probability cutoff (default 0.5).
        min_chunk_sec:  Minimum chunk duration before we look for silence.
        max_chunk_sec:  Hard ceiling for chunk duration.

    Yields:
        dict — see OUTPUT JSON SCHEMA at the top of this file.
    """
    # Step 1: Load + normalise
    audio = load_and_normalize(file_path)

    # Step 2: Initialise VAD (loads Silero model once, reuses across chunks)
    vad = SileroVAD(threshold=vad_threshold)

    # Step 3: Yield chunks one-by-one (stream-based processing)
    yield from vad_chunker(
        audio,
        vad,
        min_chunk_sec=min_chunk_sec,
        max_chunk_sec=max_chunk_sec,
    )
