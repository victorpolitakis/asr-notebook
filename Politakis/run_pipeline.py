#!/usr/bin/env python3
"""
run_pipeline.py — End-to-End Orchestrator for Multilingual Podcast Summarizer

Runs the full processing pipeline:
  audio_processor -> asr_pipeline -> llm_integration -> topic_extraction -> summary_generator

Measures execution time per stage, handles errors gracefully, and saves final
summary outputs to the results/ directory.
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import psutil

# Set up paths and imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    import audio_processor as ap
    import asr_pipeline as asr
    import llm_integration as llm
    import topic_extraction as te
    import summary_generator as sg
    import transcript_normalizer as tn
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("run_pipeline")

def _run_stage(name: str, stages: dict, stage_errors: dict, fn, *args, **kwargs):
    """Run one pipeline stage, timing it and capturing any exception."""
    t = time.time()
    try:
        result = fn(*args, **kwargs)
        stages[name] = time.time() - t
        return result, None
    except Exception as e:
        stages[name] = time.time() - t
        stage_errors[name] = str(e)
        logger.error("❌ %s failed: %s", name, e, exc_info=True)
        return None, e


def run_pipeline(audio_path: str) -> bool:
    """
    Execute the entire five-stage processing pipeline on the given audio file.

    Args:
        audio_path (str): Path to the spoken audio file to process.

    Returns:
        bool: True if the entire pipeline completes successfully without unhandled errors, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("STARTING END-TO-END PODCAST SUMMARIZATION PIPELINE")
    logger.info(f"Audio file: {audio_path}")
    logger.info("=" * 60)

    stages: dict = {}
    stage_errors: dict = {}

    # -------------------------------------------------------------
    # Stage 1 & 2: Audio Chunking & ASR Transcription (streamed)
    # -------------------------------------------------------------
    logger.info("🎙️ Stage 1 & 2: Loading, chunking, and transcribing audio…")
    t_asr = time.time()
    asr_chunks: list[dict] = []
    try:
        for chunk in asr.transcribe_file(audio_path):
            asr_chunks.append(chunk)
            if len(asr_chunks) % 5 == 0:
                rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
                logger.info("  Chunk %d | RSS %.2f GB", chunk["chunk_id"], rss_gb)
        stages["ASR Transcription"] = time.time() - t_asr
    except Exception as e:
        stages["ASR Transcription"] = time.time() - t_asr
        stage_errors["ASR Transcription"] = str(e)
        logger.error("❌ ASR Transcription failed: %s", e, exc_info=True)
        logger.error("Cannot continue without ASR chunks.")
        return False
    logger.info("✅ Stage 1 & 2 complete in %.2fs. Produced %d chunks.",
                stages["ASR Transcription"], len(asr_chunks))

    # -------------------------------------------------------------
    # Stage 3: LLM Integration (Pass-1 Live Ticker)
    # -------------------------------------------------------------
    logger.info("🧠 Stage 3: Running Pass-1 Live Ticker (LLM Named Entity & Segment Summarization)…")
    transcript, err = _run_stage("LLM Integration", stages, stage_errors,
                                  llm.process_asr_stream_sync, asr_chunks, source_file=audio_path)
    # Release full chunk list — chunks stored as slim metadata in transcript
    del asr_chunks
    if err:
        transcript = {
            "source_file": audio_path, "total_duration_sec": 20.0, "total_chunks": 1,
            "languages_detected": ["en"], "speakers_detected": ["Speaker A"],
            "chunks": [], "ticker_results": [], "full_text": "Failed transcript.",
            "all_persons": [], "all_organizations": [], "all_keywords": [], "all_main_ideas": []
        }
    else:
        logger.info("✅ Stage 3 complete in %.2fs.", stages["LLM Integration"])

    # ── Transcript Normalization ─────────────────────────────────
    raw_full_text = transcript.get("full_text", "")
    transcript["raw_full_text"] = raw_full_text  # immutable original

    if tn.ENABLE_NORMALIZATION and raw_full_text.strip():
        logger.info("Normalizing transcript: %d chars (model=%s)", len(raw_full_text), tn.NORMALIZATION_MODEL)
        normalized = tn.normalize_transcript(raw_full_text)
        if normalized is not None:
            transcript["full_text"] = normalized
            transcript["normalized_full_text"] = normalized

            # ── Entity Re-Extraction on normalized text ──
            re_extracted = tn.re_extract_entities(transcript)
            if re_extracted:
                # Replace raw ticker entities with normalized re-extracted entities
                transcript["all_persons"] = re_extracted.get("persons", [])
                transcript["all_organizations"] = re_extracted.get("organizations", [])
                transcript["all_keywords"] = re_extracted.get("keywords", [])
                for idea in re_extracted.get("main_ideas", []):
                    if idea not in transcript.get("all_main_ideas", []):
                        transcript.setdefault("all_main_ideas", []).append(idea)
                logger.info("Entity re-extraction: replaced ticker entities with %d persons, %d orgs, %d keywords.",
                           len(transcript["all_persons"]), len(transcript["all_organizations"]),
                           len(transcript["all_keywords"]))
        else:
            transcript["normalized_full_text"] = raw_full_text
    else:
        transcript["normalized_full_text"] = raw_full_text

    # -------------------------------------------------------------
    # Stage 3b: Entity Registry Building
    # -------------------------------------------------------------
    logger.info("🗂️ Stage 3b: Building Entity Registry...")
    entity_registry, err = _run_stage("Entity Registry", stages, stage_errors,
                                       te.build_entity_registry, transcript)
    if err:
        entity_registry = {
            "persons": [], "organizations": [], "keywords": [], "main_ideas": [],
            "segment_summaries": [], "total_windows": 0,
            "time_range_sec": {"start": 0.0, "end": 0.0}
        }
    else:
        logger.info("✅ Stage 3b complete in %.2fs.", stages["Entity Registry"])

    # -------------------------------------------------------------
    # Stage 4: Summary Generation (Pass-2 Summary)
    # -------------------------------------------------------------
    logger.info("📝 Stage 4: Running Pass-2 Summary & Q&A Generation...")
    summary_outputs, err = _run_stage("Summary Generation", stages, stage_errors,
                                       sg.generate_summary, transcript, entity_registry)
    if err:
        summary_outputs = {
            "source_file": audio_path, "generated_at": time.asctime(),
            "chapters": [], "entities": entity_registry,
            "summaries": {
                "tldr": "Failed to generate TL;DR.",
                "executive": "Failed to generate Executive Summary.",
                "deep_dive": {"overview": "Failed.", "bullet_points": [], "key_takeaways": [], "action_items": []}
            },
            "qa_logs": []
        }
    else:
        logger.info("✅ Stage 4 complete in %.2fs.", stages["Summary Generation"])

    # -------------------------------------------------------------
    # Save Outputs
    # -------------------------------------------------------------
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "summary_outputs.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_outputs, f, indent=2, ensure_ascii=False)
        
    transcript_path = results_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # transcript.txt = raw ASR (for WER evaluation, immutable)
    raw_path = results_dir / "transcript.txt"
    raw_out = re.sub(r'\[Speaker [A-Z]\]:\s*', '', transcript["raw_full_text"])
    raw_path.write_text(raw_out, encoding="utf-8")

    # normalized_transcript.txt = cleaned version (for debugging)
    norm_path = results_dir / "normalized_transcript.txt"
    norm_text = transcript.get("normalized_full_text", transcript["raw_full_text"])
    norm_out = re.sub(r'\[Speaker [A-Z]\]:\s*', '', norm_text)
    norm_path.write_text(norm_out, encoding="utf-8")

    logger.info("Raw transcript saved: %d bytes", raw_path.stat().st_size)
    logger.info("Normalized transcript saved: %d bytes", norm_path.stat().st_size if norm_path.exists() else 0)

    # ── Produce diarized transcript (speaker-prefixed, one turn/line) ──
    try:
        import diarize_transcript as dt
        dt.diarize_transcript()
        # Produce flat version for WER evaluation
        import strip_newlines as sn
        sn.strip_for_wer()
    except Exception as e:
        logger.warning("Diarized transcript generation failed (non-fatal): %s", e)

    logger.info("=" * 60)
    logger.info("PIPELINE RUN SUMMARY")
    logger.info("=" * 60)
    for stage, duration in stages.items():
        passed = stage not in stage_errors
        logger.info(" - %-25s: %6.2fs [%s]", stage, duration, "PASSED" if passed else "FAILED")
        if not passed:
            logger.info("   Reason: %s", stage_errors[stage])
    logger.info("\nFinal output saved to: %s", output_path.resolve())
    logger.info("=" * 60)

    if stage_errors:
        return False
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline.py <audio_file_path>")
        sys.exit(1)
    
    success = run_pipeline(sys.argv[1])
    sys.exit(0 if success else 1)
