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
# Extra-model runner (isolated thread per model)
# ═══════════════════════════════════════════════════════════════════

def _run_extra_model(model_name: str, audio_path: str) -> str:
    """Run one model from models_registry in a fresh thread+event-loop.
    Returns the raw transcript text, or raises on failure.
    """
    import threading
    import asyncio as _asyncio
    result_holder: dict[str, Any] = {}

    def _run() -> None:
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            import models_registry
            result_holder["text"] = models_registry.transcribe_with_model(model_name, audio_path)
        except Exception as exc:
            result_holder["exc"] = exc
        finally:
            # Drain pending async tasks (e.g. httpx connection pool teardown from
            # transformers/huggingface_hub) before closing the loop, to suppress
            # "Task exception was never retrieved / Event loop is closed" warnings.
            try:
                pending = _asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(_asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    if "exc" in result_holder:
        raise result_holder["exc"]
    if "text" not in result_holder:
        raise RuntimeError(f"Thread for {model_name} produced no output")
    return result_holder["text"]


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
            try:
                import run_pipeline
                result_holder["success"] = run_pipeline.run_pipeline(tmp_path)
            finally:
                try:
                    pending = _asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(_asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                loop.close()
        t = threading.Thread(target=_run_isolated)
        t.start()
        t.join()
        success = result_holder.get("success", False)

        # Free the turbo model from VRAM before extra models load.
        # Setting _whisper_model = None alone is insufficient — CTranslate2 has its
        # own allocator; unload_model() is the only way to actually release the VRAM.
        try:
            import asr_pipeline as _asr
            _asr.unload_whisper()
        except Exception as _e:
            logger.warning("Could not unload main whisper model: %s", _e)

        db.write_status(
            job_id,
            _make_status(job_id, "asr", "normalization", progress_pct=0.8, eta_seconds=60),
        )

        # Upload results from Results/ to Drive
        results_dir = Path(__file__).parent.parent / "Results"
        job_output = f"{config.DRIVE_OUTPUT}/{job_id}"
        db.get_or_create_folder(job_output)

        for result_file in ("transcript.json", "transcript.txt",
                            "normalized_transcript.txt", "summary_outputs.json",
                            "quality_metrics.json", "processing_time_analysis.json"):
            rp = results_dir / result_file
            if rp.exists():
                db.upload_file(str(rp), job_output)

        ndt = results_dir / "normalized_diarized_transcript.txt"
        ntf = results_dir / "normalized_transcript_flat.txt"
        for extra in (ndt, ntf):
            if extra.exists():
                db.upload_file(str(extra), job_output)

        # ── Extra comparison models (from meta.json selected_models) ──────────
        # whisper-turbo is already run by run_pipeline above — skip it.
        extra_models: list[str] = []
        try:
            job_files = db.list_files(job_output)
            meta_id = next((jf["id"] for jf in job_files if jf["name"] == "meta.json"), None)
            if meta_id:
                meta = db.read_json(meta_id)
                extra_models = [m for m in meta.get("selected_models", []) if m != "whisper-turbo"]
        except Exception as _me:
            logger.warning("Could not read selected_models from meta.json: %s", _me)

        # Read duration from main transcript so extra model files have correct duration
        main_duration_sec: float = 0.0
        try:
            main_t_path = results_dir / "transcript.json"
            if main_t_path.exists():
                with open(main_t_path, encoding="utf-8") as _f:
                    main_duration_sec = json.load(_f).get("total_duration_sec", 0.0)
        except Exception:
            pass

        for model_name in extra_models:
            logger.info("Running extra model: %s", model_name)
            db.write_status(
                job_id,
                _make_status(job_id, "asr", f"asr_{model_name}", progress_pct=0.85, eta_seconds=180),
            )
            try:
                raw_text = _run_extra_model(model_name, tmp_path)
                # Belt-and-suspenders: flush any lingering CUDA allocations between models
                try:
                    import gc, torch as _torch
                    gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                model_transcript = {
                    "source_file": filename,
                    "full_text": raw_text,
                    "model_name": model_name,
                    "total_duration_sec": main_duration_sec,
                    "chunks": [{"segments": [{"speaker": "Speaker A", "text": raw_text, "start": 0.0}]}],
                }
                t_json = results_dir / f"transcript_{model_name}.json"
                with open(t_json, "w", encoding="utf-8") as _f:
                    json.dump(model_transcript, _f, indent=2, ensure_ascii=False)
                db.upload_file(str(t_json), job_output)
                logger.info("✓ Extra model %s — done", model_name)
            except ModuleNotFoundError as _me:
                logger.warning("⏭  Extra model %s skipped — missing dependency: %s "
                               "(set INSTALL_NEMO=True in Cell 1 and re-run all cells)",
                               model_name, _me.name)
            except Exception as _me:
                logger.error("✗ Extra model %s failed: %s", model_name, _me, exc_info=True)

        state = "done" if success else "error"
        db.write_status(
            job_id,
            _make_status(job_id, "asr", state, progress_pct=1.0, eta_seconds=0,
                         error=None if success else "Pipeline returned False"),
        )

        db.archive_input_file(file_id)
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

        # Write completion metadata (includes script for Streamlit display)
        db.write_json(
            {
                "job_id":       job_id,
                "duration_sec": result.get("duration_sec", 0),
                "model_info":   result.get("model_info", {}),
                "word_count":   result.get("word_count", 0),
                "script_text":  result.get("script_text", ""),
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
        try:
            db.archive_input_file(file_id)
        except Exception as ae:
            logger.error("Failed to archive failed Podcast input file %s: %s", file_id, ae)


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

        db.write_json(
            {
                "job_id":       job_id,
                "duration_sec": result.get("duration_sec", 0) if result else 0,
                "model_info":   result.get("model_info", {}) if result else {},
                "word_count":   result.get("word_count", 0) if result else 0,
                "script_text":  result.get("script_text", "") if result else "",
                "completed_at": _now_iso(),
                "mp3_filename": f"{job_id}.mp3",
            },
            config.DRIVE_OUTPUT_PODCASTS,
            f"{job_id}.json",
        )
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
