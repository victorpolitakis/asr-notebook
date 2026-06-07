"""
asr_pipeline.py — Speech Recognition with Whisper large-v3-turbo

Uses faster-whisper (CTranslate2 backend) with int8 quantisation for
Apple Silicon performance. VAD-based 30 s chunking. Pyannote diarization.
"""

from __future__ import annotations

import math
import logging
import time
from pathlib import Path
from typing import Any, Generator

import numpy as np
np.seterr(invalid='ignore')
import torch
from faster_whisper import WhisperModel

from audio_processor import process_audio_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

WHISPER_MODEL_SIZE: str = "turbo"

if torch.cuda.is_available():
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    # Fix ctranslate2 "cudaErrorInvalidDevice" on Colab T4
    import os as _os
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
else:
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

MIN_CONFIDENCE_PROB: float = 0.40
MAX_NO_SPEECH_PROB: float = 0.80

DIARIZATION_ENABLED: bool = True
PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"

_whisper_model = None
_diarization_pipeline = None
_diarization_available: bool | None = None


def _get_whisper() -> WhisperModel:
    global _whisper_model, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    if _whisper_model is not None:
        return _whisper_model
    logger.info("Loading faster-whisper '%s' (device=%s, compute=%s)…",
                 WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    try:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
                                       device_index=[0], num_workers=1,
                                       compute_type=WHISPER_COMPUTE_TYPE)
    except RuntimeError:
        if WHISPER_DEVICE == "cuda":
            logger.warning("CUDA init failed, falling back to CPU…")
            WHISPER_DEVICE = "cpu"
            WHISPER_COMPUTE_TYPE = "int8"
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu",
                                           device_index=[0], num_workers=1,
                                           compute_type="int8")
        else:
            raise
    logger.info("Whisper model loaded.")
    return _whisper_model


def _get_diarization() -> Any:
    global _diarization_pipeline, _diarization_available
    if _diarization_available is False:
        return None
    if _diarization_pipeline is not None:
        return _diarization_pipeline
    try:
        import os, warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        from pyannote.audio import Pipeline as PypPipe
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        logger.info("Loading pyannote …")
        _diarization_pipeline = PypPipe.from_pretrained(PYANNOTE_MODEL, token=hf_token)
        if torch.backends.mps.is_available():
            _diarization_pipeline = _diarization_pipeline.to(torch.device("mps"))
            logger.info("Pyannote → MPS.")
        _diarization_available = True
        logger.info("Pyannote ready.")
    except Exception as e:
        logger.warning("Diarization unavailable: %s", e)
        _diarization_available = False
        return None
    return _diarization_pipeline


def _diarize(audio: np.ndarray, sr: int) -> list[dict]:
    p = _get_diarization()
    if p is None:
        return []
    try:
        wf = torch.from_numpy(audio).unsqueeze(0).float()
        d = p({"waveform": wf, "sample_rate": sr})
        if hasattr(d, "speaker_diarization"):
            d = getattr(d, "speaker_diarization")
        turns, smap, c = [], {}, 0
        for turn, _, spk in d.itertracks(yield_label=True):
            if spk not in smap:
                smap[spk] = f"Speaker {chr(65 + c)}"
                c += 1
            turns.append({"speaker": smap[spk], "start": round(turn.start, 3), "end": round(turn.end, 3)})
        return turns
    except Exception as e:
        logger.warning("Diarization failed: %s", e)
        return []


def _assign_speaker(seg_start: float, seg_end: float, turns: list[dict]) -> str | None:
    if not turns:
        return None
    best, best_ov = None, 0.0
    for t in turns:
        ov = max(0.0, min(seg_end, t["end"]) - max(seg_start, t["start"]))
        if ov > best_ov:
            best_ov, best = ov, t["speaker"]
    return best


def transcribe_chunk(chunk: dict) -> dict:
    t_start = time.perf_counter()
    model = _get_whisper()
    audio_data: np.ndarray = chunk["audio_data"]
    sample_rate: int = chunk["sample_rate"]
    chunk_offset: float = chunk["start_time_sec"]

    segments_iter, info = model.transcribe(
        audio_data,
        beam_size=3,
        word_timestamps=True,
        vad_filter=False,
        language=None,
        task="transcribe",
    )

    detected_language = info.language
    language_probability = round(info.language_probability, 4)
    all_lang_probs = [
        {"language": l, "probability": round(p, 4)}
        for l, p in (info.all_language_probs or [])
    ]

    speaker_turns = _diarize(audio_data, sample_rate) if DIARIZATION_ENABLED and chunk["is_speech"] else []

    processed_segments: list[dict] = []
    reliable_texts: list[str] = []
    all_speakers: set[str] = set()
    hallucination_count = 0

    for seg_id, segment in enumerate(segments_iter):
        confidence = math.exp(segment.avg_logprob) if segment.avg_logprob else 0.0
        is_reliable = confidence >= MIN_CONFIDENCE_PROB and segment.no_speech_prob <= MAX_NO_SPEECH_PROB

        if not is_reliable:
            hallucination_count += 1

        speaker = _assign_speaker(segment.start, segment.end, speaker_turns)
        if speaker:
            all_speakers.add(speaker)

        words_list = [
            {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
             "probability": round(w.probability, 4)}
            for w in (segment.words or [])
        ]

        seg_dict = {
            "segment_id": seg_id,
            "text": segment.text.strip(),
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "start_absolute": round(segment.start + chunk_offset, 3),
            "end_absolute": round(segment.end + chunk_offset, 3),
            "avg_logprob": round(segment.avg_logprob, 4) if segment.avg_logprob else 0.0,
            "no_speech_prob": round(segment.no_speech_prob, 4),
            "confidence": round(confidence, 4),
            "is_reliable": is_reliable,
            "speaker": speaker,
            "words": words_list,
        }
        processed_segments.append(seg_dict)

        if is_reliable:
            prefix = f"[{speaker}]: " if speaker else ""
            reliable_texts.append(f"{prefix}{segment.text.strip()}")

    del chunk["audio_data"]

    # Free MPS GPU cache to prevent memory accumulation across chunks
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    t_end = time.perf_counter()

    result_dict = {
        "chunk_id": chunk["chunk_id"],
        "start_time_sec": chunk["start_time_sec"],
        "end_time_sec": chunk["end_time_sec"],
        "duration_sec": chunk["duration_sec"],
        "detected_language": detected_language,
        "language_probability": language_probability,
        "all_language_probs": all_lang_probs,
        "is_speech": chunk["is_speech"],
        "segments": processed_segments,
        "full_text": " ".join(reliable_texts),
        "speakers_detected": sorted(all_speakers),
        "processing_time_sec": round(t_end - t_start, 4),
        "rms_db": chunk["rms_db"],
        "hallucination_filtered_count": hallucination_count,
    }
    logger.info("Chunk %d: lang=%s (%.2f) | %d segs (%d filt) | %.2f s",
                chunk["chunk_id"], detected_language, language_probability,
                len(processed_segments), hallucination_count,
                result_dict["processing_time_sec"])
    return result_dict


def transcribe_file(
    file_path: str | Path,
    *,
    vad_threshold: float = 0.5,
    min_chunk_sec: float = 25.0,
    max_chunk_sec: float = 30.0,
    skip_non_speech: bool = True,
) -> Generator[dict, None, None]:
    logger.info("=" * 60)
    logger.info("  ASR — faster-whisper turbo (int8)")
    logger.info("  File: %s", file_path)
    logger.info("=" * 60)

    for chunk in process_audio_file(
        file_path, vad_threshold=vad_threshold,
        min_chunk_sec=min_chunk_sec, max_chunk_sec=max_chunk_sec,
    ):
        if skip_non_speech and not chunk["is_speech"]:
            logger.info("Chunk %d: skip (no speech)", chunk["chunk_id"])
            yield {"chunk_id": chunk["chunk_id"], "start_time_sec": chunk["start_time_sec"],
                   "end_time_sec": chunk["end_time_sec"], "duration_sec": chunk["duration_sec"],
                   "detected_language": None, "language_probability": 0.0,
                   "all_language_probs": [], "is_speech": False,
                   "segments": [], "full_text": "", "speakers_detected": [],
                   "processing_time_sec": 0.0, "rms_db": chunk["rms_db"], "hallucination_filtered_count": 0}
            del chunk["audio_data"]
            continue
        yield transcribe_chunk(chunk)
