"""
config.py — Shared constants for the Drive Bridge and Colab job watcher.

Imported by both drive_bridge.py and colab_job_watcher.py to ensure
consistent folder paths, polling intervals, and status schemas.

No file in the project hardcodes Drive paths, polling rates, or
status JSON keys — they all import from here.
"""

from __future__ import annotations

from typing import TypedDict

# ═══════════════════════════════════════════════════════════════════
# Google Drive folder structure
# ═══════════════════════════════════════════════════════════════════

DRIVE_ROOT: str = "ece22073"
DRIVE_INPUT: str = f"{DRIVE_ROOT}/input"
DRIVE_INPUT_JOBS: str = f"{DRIVE_INPUT}/podcast_jobs"
DRIVE_INPUT_PROCESSED: str = f"{DRIVE_INPUT}/processed"
DRIVE_OUTPUT: str = f"{DRIVE_ROOT}/output"
DRIVE_OUTPUT_PODCASTS: str = f"{DRIVE_OUTPUT}/podcasts"
DRIVE_MODELS_CACHE: str = f"{DRIVE_ROOT}/models"

# ═══════════════════════════════════════════════════════════════════
# Polling intervals (seconds)
# ═══════════════════════════════════════════════════════════════════

POLL_INTERVAL_SEC: int = 10       # Colab watcher loop
LOCAL_POLL_INTERVAL_SEC: int = 15  # Streamlit refresh
STALL_TIMEOUT_SEC: int = 600      # 10 min — no status update → stalled

# ═══════════════════════════════════════════════════════════════════
# Job state machine
# ═══════════════════════════════════════════════════════════════════

# Valid stage values for status.json
JOB_STAGES: tuple[str, ...] = (
    "uploading",
    "asr",
    "normalization",
    "summary",
    "podcast_script",
    "podcast_tts",
    "done",
    "error",
    "stalled",
)

# Non-terminal states — Colab watcher will resume these on restart
TERMINAL_STAGES: tuple[str, ...] = ("done", "error", "stalled")

# ═══════════════════════════════════════════════════════════════════
# TypedDict schemas
# ═══════════════════════════════════════════════════════════════════


class StatusDict(TypedDict, total=False):
    """
    Written to output/{job_id}/status.json by the Colab watcher.
    Polled every LOCAL_POLL_INTERVAL_SEC seconds by the Streamlit app.

    Schema:
    {
      "job_id": str,
      "job_type": "asr" | "podcast",
      "stage": "uploading" | "asr" | "normalization" | "summary"
             | "podcast_script" | "podcast_tts" | "done" | "error" | "stalled",
      "progress_pct": float,       # 0.0 - 1.0
      "eta_seconds": float,        # estimated remaining
      "error": str | None,         # error message if stage == "error"
      "updated_at": str            # ISO 8601 timestamp
    }
    """

    job_id: str
    job_type: str
    stage: str
    progress_pct: float
    eta_seconds: float
    error: str | None
    updated_at: str


class PodcastJobDict(TypedDict, total=False):
    """
    Written to input/podcast_jobs/{job_id}.json by the Streamlit app.
    Picked up by the Colab watcher's podcast job handler.

    Schema:
    {
      "job_id": str,
      "source_text": str,
      "speaker_a": { "name": str, "description": str, "tts_model": str, "voice": str | null },
      "speaker_b": { "name": str, "description": str, "tts_model": str, "voice": str | null },
      "config": { "tone": "casual" | "academic" | "debate" | "interview",
                  "length": "short" | "medium" | "long" },
      "created_at": str            # ISO 8601 timestamp
    }
    """

    job_id: str
    source_text: str
    speaker_a: dict
    speaker_b: dict
    config: dict
    created_at: str
