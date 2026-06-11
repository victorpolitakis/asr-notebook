"""
real_time_processor.py — Real-Time Streaming Pipeline Orchestrator

Integrates all five pipeline stages into a single streaming coroutine that
processes audio chunks as they arrive from the VAD chunker, fires LLM ticker
calls concurrently, and emits live events (transcripts, entities, summaries)
via an async generator.

Architecture:
  audio_processor.process_audio_file()  [generator]
      └─► asr_pipeline.transcribe_chunk()  [per-chunk, blocking]
              └─► llm_integration.AccumulatedTranscript.add_chunk()  [async ticker]
                      └─► topic_extraction.EntityRegistry  [incremental update]
                              └─► summary_generator.generate_summary_async()  [on EOF]

Events emitted via real_time_events():
  {"type": "chunk",   "data": <asr chunk dict>}
  {"type": "ticker",  "data": <ticker result dict>}
  {"type": "summary", "data": <SummaryOutputs dict>}
  {"type": "done",    "data": {"total_chunks": int, "duration_sec": float}}
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ───────────────────────────────────────────────────────────────────────────
# Lazy imports (heavy ML deps may not be installed in all environments)
# ───────────────────────────────────────────────────────────────────────────

def _import_pipeline():
    """Return all pipeline modules as a namespace dict."""
    import audio_processor as ap      # noqa: F401
    import asr_pipeline   as asr      # noqa: F401
    import llm_integration as llm     # noqa: F401
    import topic_extraction as te     # noqa: F401
    import summary_generator as sg    # noqa: F401
    return dict(ap=ap, asr=asr, llm=llm, te=te, sg=sg)


# ───────────────────────────────────────────────────────────────────────────
# Core async streaming generator
# ───────────────────────────────────────────────────────────────────────────

async def real_time_events(
    file_path: str | Path,
    *,
    vad_threshold: float = 0.5,
    min_chunk_sec: float = 5.0,
    max_chunk_sec: float = 10.0,
    skip_non_speech: bool = True,
) -> AsyncIterator[dict]:
    """
    Stream pipeline events for a given audio file.

    Yields event dicts with "type" and "data" keys. Callers can display
    transcripts as they arrive, show live entity chips, and display the
    final summary when processing completes.

    Args:
        file_path:       Path to any audio file (mp3, wav, ogg, m4a …).
        vad_threshold:   Silero VAD speech-probability threshold.
        min_chunk_sec:   Minimum chunk duration before silence cut.
        max_chunk_sec:   Hard ceiling for chunk duration.
        skip_non_speech: Skip chunks with no detected speech.

    Yields:
        {"type": "chunk",   "data": <asr_pipeline chunk dict>}
        {"type": "ticker",  "data": <llm_integration SCHEMA 1 dict>}
        {"type": "summary", "data": <summary_generator SummaryOutputs dict>}
        {"type": "done",    "data": {"total_chunks": int, "duration_sec": float}}
    """
    mods = _import_pipeline()
    asr = mods["asr"]
    llm = mods["llm"]
    te  = mods["te"]
    sg  = mods["sg"]

    acc = llm.AccumulatedTranscript(source_file=str(file_path))
    registry = te.EntityRegistry()
    pending_tasks: list[asyncio.Task] = []
    total_chunks = 0
    total_duration = 0.0

    logger.info("RealTimeProcessor: starting  →  %s", file_path)
    t0 = time.perf_counter()

    # Run the blocking ASR pipeline in a thread so we don't block the event loop
    asr_chunks = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: list(asr.transcribe_file(
            file_path,
            vad_threshold=vad_threshold,
            min_chunk_sec=min_chunk_sec,
            max_chunk_sec=max_chunk_sec,
            skip_non_speech=skip_non_speech,
        )),
    )

    for chunk in asr_chunks:
        total_chunks += 1
        total_duration += chunk.get("duration_sec", 0.0)

        # Yield the live transcript chunk immediately
        yield {"type": "chunk", "data": chunk}

        # Feed into accumulator — may schedule a background LLM ticker task
        task = acc.add_chunk(chunk)
        if task is not None:
            pending_tasks.append(task)

            # Wait for ticker to complete and yield its result live
            ticker_result = await task
            if isinstance(ticker_result, dict):
                registry.update_from_ticker(ticker_result)
                yield {"type": "ticker", "data": ticker_result}

    # Flush any remaining text window (< TICKER_WINDOW_SEC)
    flush_task = await acc.flush_remaining()
    if flush_task is not None:
        ticker_result = await flush_task
        if isinstance(ticker_result, dict):
            registry.update_from_ticker(ticker_result)
            yield {"type": "ticker", "data": ticker_result}

    # Wait for any still-pending background tasks
    if pending_tasks:
        results = await asyncio.gather(*pending_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, dict):
                registry.update_from_ticker(r)
                yield {"type": "ticker", "data": r}

    # Pass-2 summary generation (triggered on EOF)
    transcript = acc.to_dict()
    entities   = registry.to_dict()
    summary    = await sg.generate_summary_async(transcript, entities)
    yield {"type": "summary", "data": summary}

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "RealTimeProcessor: done in %.2fs | %d chunks | %.1fs audio",
        elapsed, total_chunks, total_duration,
    )
    yield {"type": "done", "data": {"total_chunks": total_chunks, "duration_sec": total_duration}}


# ───────────────────────────────────────────────────────────────────────────
# Synchronous wrapper — convenience for scripts and notebooks
# ───────────────────────────────────────────────────────────────────────────

def process_file(
    file_path: str | Path,
    *,
    on_chunk=None,
    on_ticker=None,
    on_summary=None,
    **kwargs,
) -> dict:
    """
    Synchronous wrapper around real_time_events().

    Optional callbacks receive each event as it arrives:
        on_chunk(chunk_dict)   — called per transcribed chunk
        on_ticker(ticker_dict) — called per LLM ticker result
        on_summary(summary)    — called when Pass-2 summary is ready

    Returns the final SummaryOutputs dict.
    """
    async def _run():
        summary_result = {}
        async for event in real_time_events(file_path, **kwargs):
            t = event["type"]
            if t == "chunk"   and on_chunk   is not None: on_chunk(event["data"])
            elif t == "ticker"  and on_ticker  is not None: on_ticker(event["data"])
            elif t == "summary" and on_summary is not None: on_summary(event["data"])
            if t == "summary":
                summary_result = event["data"]
        return summary_result

    return asyncio.run(_run())


# ───────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python real_time_processor.py <audio_file>")
        sys.exit(1)

    print(f"\nReal-Time Pipeline  →  {sys.argv[1]}\n{'='*60}")

    def _on_chunk(c):
        if c.get("is_speech"):
            print(f"  [Chunk {c['chunk_id']:02d}] {c.get('detected_language','?')} | "
                  f"{c['start_time_sec']:.1f}–{c['end_time_sec']:.1f}s | "
                  f"{c.get('full_text','')[:80]}")

    def _on_ticker(t):
        print(f"\n  [Ticker] window {t['window_start']:.0f}–{t['window_end']:.0f}s")
        print(f"    persons={t['persons'][:3]}  orgs={t['organizations'][:3]}")
        print(f"    summary: {t['segment_summary'][:100]}\n")

    result = process_file(sys.argv[1], on_chunk=_on_chunk, on_ticker=_on_ticker)
    print("\n" + "="*60)
    print("SUMMARY (TL;DR):", result.get("summaries", {}).get("tldr", "N/A"))
    print("="*60)
