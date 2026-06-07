"""
colab_job_watcher.py — Runs inside the Colab notebook as the main orchestration loop.

Polls Google Drive input/ folder for new WAV files (ASR jobs) and
input/podcast_jobs/ for JSON configs (podcast generation jobs).

For each job:
  1. Writes a status.json to output/{job_id}/ (live progress)
  2. Executes the pipeline (run_pipeline.py or podcast_pipeline.py)
  3. Uploads results to output/{job_id}/
  4. Moves the input file to input/processed/ archive

Stall recovery: if Colab dies and restarts, input files remain untouched
in input/ — the watcher picks them up immediately on cold restart with
zero manual cleanup needed.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import drive_bridge as db

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Status helpers
# ═══════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_status(
    job_id: str,
    job_type: str,
    stage: str,
    progress_pct: float = 0.0,
    eta_seconds: float = 0.0,
    error: str | None = None,
) -> config.StatusDict:
    return {
        "job_id": job_id,
        "job_type": job_type,
        "stage": stage,
        "progress_pct": round(progress_pct, 4),
        "eta_seconds": round(eta_seconds, 4),
        "error": error,
        "updated_at": _now_iso(),
    }


# ═══════════════════════════════════════════════════════════════════
# ASR job handler
# ═══════════════════════════════════════════════════════════════════

def _handle_asr_job(file_info: dict) -> None:
    """Download a WAV from Drive, run the pipeline, upload results."""
    file_id = file_info["id"]
    filename = file_info["name"]
    # Derive job_id from filename (Streamlit uploads as {job_id}.wav)
    job_id = filename.replace(".wav", "") if filename.endswith(".wav") else db.generate_job_id()
    logger.info("🎙️ ASR job %s — starting for %s", job_id, filename)

    try:
        db.write_status(
            job_id,
            _make_status(job_id, "asr", "asr", progress_pct=0.05, eta_seconds=600),
        )

        # Download WAV from Drive to temp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        db.download_file(file_id, tmp_path)
        logger.info("  Downloaded %s → %s", filename, tmp_path)

        db.write_status(
            job_id,
            _make_status(job_id, "asr", "asr", progress_pct=0.2, eta_seconds=500),
        )

        # Run the pipeline in a separate thread with its own event loop.
        # This isolates it from Colab's IPython event loop — no nesting.
        import threading
        import asyncio as _asyncio
        result_holder: dict[str, Any] = {}
        def _run_isolated() -> None:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            import run_pipeline
            result_holder["success"] = run_pipeline.run_pipeline(tmp_path)
        t = threading.Thread(target=_run_isolated)
        t.start()
        t.join()
        success = result_holder.get("success", False)

        db.write_status(
            job_id,
            _make_status(job_id, "asr", "normalization", progress_pct=0.8, eta_seconds=60),
        )

        # Upload results from Politakis/results/ to Drive
        results_dir = Path(__file__).parent / "results"
        job_output = f"{config.DRIVE_OUTPUT}/{job_id}"
        db.get_or_create_folder(job_output)

        for result_file in ("transcript.json", "transcript.txt",
                            "normalized_transcript.txt", "summary_outputs.json",
                            "quality_metrics.json", "processing_time_analysis.json"):
            rp = results_dir / result_file
            if rp.exists():
                db.upload_file(str(rp), job_output)

        # Upload normalized_diarized_transcript.txt if it exists (Phase 5 output)
        ndt = results_dir / "normalized_diarized_transcript.txt"
        ntf = results_dir / "normalized_transcript_flat.txt"
        for extra in (ndt, ntf):
            if extra.exists():
                db.upload_file(str(extra), job_output)

        state = "done" if success else "error"
        db.write_status(
            job_id,
            _make_status(job_id, "asr", state, progress_pct=1.0, eta_seconds=0,
                         error=None if success else "Pipeline returned False"),
        )

        # Archive the input file
        db.archive_input_file(file_id)

        # Clean up temp
        os.unlink(tmp_path)
        logger.info("✅ ASR job %s — %s", job_id, state)

    except Exception as e:
        logger.error("❌ ASR job %s failed: %s", job_id, e, exc_info=True)
        db.write_status(
            job_id,
            _make_status(job_id, "asr", "error", progress_pct=0.0, eta_seconds=0,
                         error=str(e)[:500]),
        )
        # Don't archive — leave input for retry on restart


# ═══════════════════════════════════════════════════════════════════
# Podcast job handler
# ═══════════════════════════════════════════════════════════════════

def _handle_podcast_job(file_info: dict) -> None:
    """Read a podcast job JSON from Drive, run TTS pipeline, upload MP3."""
    file_id = file_info["id"]
    filename = file_info["name"]

    # Derive job_id from filename (podcast_jobs are named {job_id}.json)
    job_id = filename.replace(".json", "")
    logger.info("🎙️ Podcast job %s — starting", job_id)

    try:
        db.write_status(
            job_id,
            _make_status(job_id, "podcast", "podcast_script", progress_pct=0.05, eta_seconds=300),
        )

        job_config = db.read_json(file_id)

        db.write_status(
            job_id,
            _make_status(job_id, "podcast", "podcast_tts", progress_pct=0.3, eta_seconds=240),
        )

        # Lazy import podcast_pipeline (may not be importable if deps missing)
        try:
            import podcast_pipeline as pp
        except ImportError as e:
            logger.warning("podcast_pipeline not available: %s — skipping TTS", e)
            db.write_status(
                job_id,
                _make_status(job_id, "podcast", "error", progress_pct=0.0, eta_seconds=0,
                             error=f"podcast_pipeline import failed: {e}"),
            )
            db.archive_input_file(file_id)
            return

        result = pp.generate_podcast(job_config)

        # Upload MP3 and metadata to output/podcasts/
        if result.get("mp3_path") and Path(result["mp3_path"]).exists():
            db.upload_file(result["mp3_path"], config.DRIVE_OUTPUT_PODCASTS, filename=f"{job_id}.mp3")

        # Write completion metadata
        db.write_json(
            {
                "job_id": job_id,
                "duration_sec": result.get("duration_sec", 0),
                "model_info": result.get("model_info", {}),
                "word_count": result.get("word_count", 0),
                "completed_at": _now_iso(),
                "mp3_filename": f"{job_id}.mp3",
            },
            config.DRIVE_OUTPUT_PODCASTS,
            f"{job_id}.json",
        )

        db.write_status(
            job_id,
            _make_status(job_id, "podcast", "done", progress_pct=1.0, eta_seconds=0),
        )

        db.archive_input_file(file_id)
        logger.info("✅ Podcast job %s — done", job_id)

    except Exception as e:
        logger.error("❌ Podcast job %s failed: %s", job_id, e, exc_info=True)
        db.write_status(
            job_id,
            _make_status(job_id, "podcast", "error", progress_pct=0.0, eta_seconds=0,
                         error=str(e)[:500]),
        )


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════

def main_loop() -> None:
    """Run indefinitely — poll Drive for new jobs and process them."""

    logger.info("🚀 Colab job watcher starting")
    logger.info("   Poll interval: %ds", config.POLL_INTERVAL_SEC)

    # Ensure Drive folder structure exists
    db.init_drive_structure()

    logger.info("   Drive folders initialized")
    logger.info("   Watching: %s + %s", config.DRIVE_INPUT, config.DRIVE_INPUT_JOBS)

    processed_ids: set[str] = set()  # Dedupe across restarts

    while True:
        try:
            # ── Check for ASR jobs (WAV files in input/) ──
            for f in db.find_new_input_files():
                if f["id"] in processed_ids:
                    continue
                if f["name"].lower().endswith(".wav"):
                    processed_ids.add(f["id"])
                    _handle_asr_job(f)

            # ── Check for podcast jobs (JSON in input/podcast_jobs/) ──
            for f in db.find_new_podcast_jobs():
                if f["id"] in processed_ids:
                    continue
                processed_ids.add(f["id"])
                _handle_podcast_job(f)

        except Exception as e:
            logger.error("Watcher loop exception: %s", e, exc_info=True)

        time.sleep(config.POLL_INTERVAL_SEC)


# ═══════════════════════════════════════════════════════════════════
# Entry point for notebook
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    main_loop()
