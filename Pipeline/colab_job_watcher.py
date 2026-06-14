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
    # Derive job_id from filename stem (Streamlit uploads as {job_id}.wav/.mp3/.m4a)
    job_id = Path(filename).stem
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

        # Retrieve selected models from meta.json in output/{job_id}/
        selected_models = ["whisper-turbo"]
        try:
            job_output = f"{config.DRIVE_OUTPUT}/{job_id}"
            job_files = db.list_files(job_output)
            meta_id = [jf["id"] for jf in job_files if jf["name"] == "meta.json"]
            if meta_id:
                meta = db.read_json(meta_id[0])
                selected_models = meta.get("selected_models", ["whisper-turbo"])
        except Exception as e:
            logger.warning("Could not read selected_models from meta.json: %s", e)

        db.write_status(
            job_id,
            _make_status(job_id, "asr", "asr", progress_pct=0.2, eta_seconds=500),
        )

        # Run transcription for each selected model
        import models_registry
        results_dir = Path(__file__).parent.parent / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old transcripts in Results directory
        for p in results_dir.glob("transcript_*"):
            try:
                p.unlink()
            except OSError:
                pass

        success = True
        for idx, model_name in enumerate(selected_models):
            logger.info("Running transcription with model: %s", model_name)
            try:
                db.write_status(
                    job_id,
                    _make_status(job_id, "asr", f"transcribing_{model_name}", progress_pct=0.2 + 0.6 * (idx / len(selected_models)), eta_seconds=300),
                )
                
                raw_text = models_registry.transcribe_with_model(model_name, tmp_path)
                
                # Apply normalization
                import transcript_normalizer as tn
                norm_text = raw_text
                if tn.ENABLE_NORMALIZATION:
                    norm_text = tn.normalize_transcript(raw_text) or raw_text
                
                model_transcript = {
                    "source_file": filename,
                    "total_duration_sec": 30.0,
                    "languages_detected": ["el" if "el" in model_name else "en"],
                    "speakers_detected": ["Speaker A"],
                    "full_text": raw_text,
                    "normalized_full_text": norm_text,
                    "model_name": model_name,
                    "chunks": [{"segments": [{"speaker": "Speaker A", "text": raw_text, "start": 0.0}]}]
                }
                
                # Save to local Results folder
                t_json_path = results_dir / f"transcript_{model_name}.json"
                t_txt_path = results_dir / f"transcript_{model_name}.txt"
                
                with open(t_json_path, "w", encoding="utf-8") as f:
                    json.dump(model_transcript, f, indent=2, ensure_ascii=False)
                t_txt_path.write_text(raw_text, encoding="utf-8")
                
                # Copy to default for the first selected model so standard pipeline runs next stages
                if idx == 0:
                    with open(results_dir / "transcript.json", "w", encoding="utf-8") as f:
                        json.dump(model_transcript, f, indent=2, ensure_ascii=False)
                    (results_dir / "transcript.txt").write_text(raw_text, encoding="utf-8")
                    
            except Exception as e:
                logger.error("Error transcribing with model %s: %s", model_name, e)
                success = False

        db.write_status(
            job_id,
            _make_status(job_id, "asr", "normalization", progress_pct=0.8, eta_seconds=60),
        )

        # Upload results from Politakis/results/ to Drive
        job_output = f"{config.DRIVE_OUTPUT}/{job_id}"
        db.get_or_create_folder(job_output)

        for result_file in ("transcript.json", "transcript.txt",
                            "normalized_transcript.txt", "summary_outputs.json",
                            "quality_metrics.json", "processing_time_analysis.json"):
            rp = results_dir / result_file
            if rp.exists():
                db.upload_file(str(rp), job_output)

        # Upload dynamic transcripts for each model
        for p in results_dir.glob("transcript_*.json"):
            db.upload_file(str(p), job_output)
        for p in results_dir.glob("transcript_*.txt"):
            db.upload_file(str(p), job_output)

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
                         error=None if success else "ASR pipeline failed for one or more models"),
        )

        # Archive the input file
        db.archive_input_file(file_id)

        # Clean up temp
        os.unlink(tmp_path)
        logger.info("%s ASR job %s — %s", "✅" if success else "❌", job_id, state)

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
# Filesystem-based handlers (Drive mounted — no API cache lag)
# ═══════════════════════════════════════════════════════════════════

def _write_status_fs(job_id: str, status: dict) -> None:
    """Write status.json directly to mounted Drive filesystem."""
    import json as _json
    out_dir = f"/content/drive/MyDrive/ece22073/output/{job_id}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        _json.dump(status, f)


def _handle_asr_job_fs(wav_path: str, filename: str) -> None:
    """Process a WAV file directly from the mounted Drive filesystem."""
    job_id = Path(filename).stem
    logger.info("🎙️ ASR job %s — starting for %s", job_id, filename)

    try:
        db.write_status(job_id, _make_status(job_id, "asr", "asr", progress_pct=0.05, eta_seconds=600))

        import run_pipeline
        success = run_pipeline.run_pipeline(wav_path)

        db.write_status(job_id, _make_status(job_id, "asr", "normalization", progress_pct=0.8, eta_seconds=60))

        # Upload results via Drive API (same folder db.write_status created)
        results_dir = Path(__file__).parent.parent / "Results"
        job_output = f"{config.DRIVE_OUTPUT}/{job_id}"

        for result_file in ("transcript.json", "transcript.txt",
                            "normalized_transcript.txt", "summary_outputs.json",
                            "quality_metrics.json", "processing_time_analysis.json",
                            "normalized_diarized_transcript.txt", "normalized_transcript_flat.txt"):
            rp = results_dir / result_file
            if rp.exists():
                db.upload_file(str(rp), job_output)
                logger.info("  Uploaded %s → Drive", result_file)

        state = "done" if success else "error"
        db.write_status(job_id, _make_status(job_id, "asr", state, progress_pct=1.0, eta_seconds=0,
                                             error=None if success else "Pipeline returned False"))

        # Archive: move WAV to input/processed/
        processed_dir = os.path.dirname(wav_path) + "/processed"
        os.makedirs(processed_dir, exist_ok=True)
        os.rename(wav_path, os.path.join(processed_dir, filename))
        logger.info("%s ASR job %s — %s", "✅" if success else "❌", job_id, state)

    except Exception as e:
        logger.error("❌ ASR job %s failed: %s", job_id, e, exc_info=True)
        db.write_status(job_id, _make_status(job_id, "asr", "error", progress_pct=0.0, eta_seconds=0,
                                              error=str(e)[:500]))
        # Leave WAV in place for retry on cold restart


def _handle_podcast_job_fs(json_path: str, filename: str) -> None:
    """Process a podcast job JSON from the mounted Drive filesystem."""
    job_id = filename.replace(".json", "")
    logger.info("🎙️ Podcast job %s — starting", job_id)

    try:
        import json as _json
        with open(json_path) as f:
            job_config = _json.load(f)

        db.write_status(job_id, _make_status(job_id, "podcast", "podcast_script", progress_pct=0.05, eta_seconds=300))

        try:
            import podcast_pipeline as pp
        except ImportError as e:
            logger.warning("podcast_pipeline not available: %s", e)
            db.write_status(job_id, _make_status(job_id, "podcast", "error", progress_pct=0.0, eta_seconds=0,
                                                  error=f"podcast_pipeline import failed: {e}"))
            return

        result = pp.generate_podcast(job_config)

        if result and result.get("mp3_path"):
            db.upload_file(result["mp3_path"], config.DRIVE_OUTPUT_PODCASTS, filename=f"{job_id}.mp3")

        db.write_status(job_id, _make_status(job_id, "podcast", "done", progress_pct=1.0, eta_seconds=0))

        processed_dir = os.path.dirname(json_path) + "/processed"
        os.makedirs(processed_dir, exist_ok=True)
        os.rename(json_path, os.path.join(processed_dir, filename))
        logger.info("✅ Podcast job %s — done", job_id)

    except Exception as e:
        logger.error("❌ Podcast job %s failed: %s", job_id, e, exc_info=True)
        db.write_status(job_id, _make_status(job_id, "podcast", "error", progress_pct=0.0, eta_seconds=0,
                                              error=str(e)[:500]))


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════

_IDLE_TIMEOUT_SEC: int = 300  # 5 minutes of no jobs → auto-stop


def main_loop() -> None:
    """Poll Drive for new jobs. Auto-exits after 5 minutes of idle (no jobs found)."""

    logger.info("🚀 Colab job watcher starting")
    logger.info("   Poll interval: %ds | Idle timeout: %ds", config.POLL_INTERVAL_SEC, _IDLE_TIMEOUT_SEC)

    # Ensure Drive folder structure exists
    db.init_drive_structure()

    logger.info("   Drive folders initialized")
    logger.info("   Watching: %s + %s", config.DRIVE_INPUT, config.DRIVE_INPUT_JOBS)

    processed_names: set[str] = set()  # Dedupe by filename across restarts
    idle_since = time.time()

    while True:
        job_found = False
        try:
            # ── Check for ASR jobs (WAV files in input/) via Drive API ──
            for file_info in db.find_new_input_files():
                fname = file_info["name"]
                if not (fname.lower().endswith(".wav") or fname.lower().endswith(".mp3") or fname.lower().endswith(".m4a")):
                    continue
                if fname in processed_names:
                    continue
                processed_names.add(fname)
                job_found = True
                idle_since = time.time()
                _handle_asr_job(file_info)

            # ── Check for podcast jobs (JSON in input/podcast_jobs/) via Drive API ──
            for file_info in db.find_new_podcast_jobs():
                fname = file_info["name"]
                if fname in processed_names:
                    continue
                processed_names.add(fname)
                job_found = True
                idle_since = time.time()
                _handle_podcast_job(file_info)

        except Exception as e:
            logger.error("Watcher loop exception: %s", e, exc_info=True)

        if not job_found and (time.time() - idle_since) >= _IDLE_TIMEOUT_SEC:
            logger.info("⏹️  No jobs for %ds — stopping watcher to save Colab runtime.", _IDLE_TIMEOUT_SEC)
            return

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
