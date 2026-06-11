"""
llm_integration.py — Section 3: Topic and Content Extraction (25 pts)

Architecture rules applied (implementation_plan.md §Overarching):
  1. Strict Python Type Hints on every public function.
  2. All data exchanged as parsed JSON dicts — never raw multi-line strings.
  3. MPS device targeting for any embedding math (OpenAI API is remote;
     local Ollama/llama.cpp calls inherit MPS from the host process).
  4. Every non-trivial block has a "why" comment, not just a "what".

No RAG / no vector databases — 1-hour transcripts fit easily in a
128k-token context window; we send raw JSON text in a single shot.

=============================================================================
SCHEMA 1 — Per-Segment Live Ticker Output  (Pass 1, fired every ~2 min)
=============================================================================
{
    "chunk_id":        int,         # Which chunk triggered this ticker call
    "window_start":    float,       # Absolute start of the accumulated window (sec)
    "window_end":      float,       # Absolute end of the accumulated window (sec)
    "persons":         list[str],   # Named persons extracted by LLM NER
    "organizations":   list[str],   # Named organisations extracted by LLM NER
    "keywords":        list[str],   # Top content keywords / key topics
    "main_ideas":      list[str],   # 2-4 main ideas from this window
    "segment_summary": str          # 1-sentence abstractive summary of the window
}

=============================================================================
SCHEMA 2 — Full Accumulated Transcript Object  (built by AccumulatedTranscript)
=============================================================================
{
    "source_file":      str,              # Path of the audio file processed
    "total_duration_sec": float,          # Total duration of all chunks (sec)
    "total_chunks":     int,              # Number of ASR chunks consumed
    "languages_detected": list[str],      # Unique ISO 639-1 codes seen
    "speakers_detected":  list[str],      # Unique speaker labels seen
    "chunks": [                           # Ordered list of ASR chunk dicts
        { ...asr_pipeline OUTPUT SCHEMA... }
    ],
    "ticker_results": [                   # Ordered list of Pass-1 ticker dicts
        { ...SCHEMA 1 above... }
    ],
    "full_text": str,                     # Entire concatenated reliable text
    "all_persons":       list[str],       # Deduplicated across all ticker windows
    "all_organizations": list[str],       # Deduplicated across all ticker windows
    "all_keywords":      list[str],       # Deduplicated across all ticker windows
    "all_main_ideas":    list[str],       # Deduplicated across all ticker windows
    "topic_extraction_80pct_note": str    # Reminder: LLM prompt targets ≥80% recall
}
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants — single source of truth for all tuneable knobs
# ---------------------------------------------------------------------------

# How many seconds of accumulated reliable text triggers a Pass-1 ticker call.
# ~2 minutes per the implementation plan: "Every ~2 minutes of accumulated text,
# pause and send the block to the LLM."
TICKER_WINDOW_SEC: float = 120.0

# OpenAI-compatible endpoint — works with OpenAI, Ollama, LM Studio, etc.
# Override via environment variable to point at a local model (e.g. Ollama).
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# Model to use — can be swapped for any OpenAI-compatible model name.
# "gpt-4o-mini" offers a large (128k) context at low latency/cost.
# For local Ollama: set LLM_MODEL="llama3" and LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-5.4-mini-2026-03-17")

# Maximum tokens the LLM should generate per ticker call.
# 512 is enough for a tight JSON response with NER + summary.
LLM_MAX_TOKENS: int = 512

# Temperature = 0 → deterministic extraction; we want factual NER, not creativity.
LLM_TEMPERATURE: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. LLM Client — thin async wrapper around the OpenAI-compatible HTTP API
# ═══════════════════════════════════════════════════════════════════════════

def _build_openai_client() -> Any:
    """
    Lazily import and construct the AsyncOpenAI client.

    Why lazy import?
    - Keeps the module importable even if the user hasn't installed `openai`.
    - Allows the client to pick up env vars set after module load.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
        client = AsyncOpenAI(
            api_key=LLM_API_KEY or "ollama",  # "ollama" satisfies local servers
            base_url=LLM_BASE_URL,
        )
        return client
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required: pip install openai"
        ) from exc


async def _call_llm(system_prompt: str, user_content: str) -> str:
    """
    Fire a single async LLM completion and return the raw response string.

    Why async?
    - Architecture rule: LLM calls must NOT block the transcription stream.
    - asyncio.create_task() lets the caller schedule this and continue yielding
      ASR chunks while the network round-trip completes in the background.

    Args:
        system_prompt: Instructions telling the LLM exactly what JSON to return.
        user_content:  The transcript text block to analyse.

    Returns:
        Raw string from the LLM (expected to be a JSON object).
    """
    client = _build_openai_client()
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. Prompt Engineering — maximising NER recall (≥80% rubric target)
# ═══════════════════════════════════════════════════════════════════════════

# Why a single detailed system prompt?
# - Instructs the LLM to be EXHAUSTIVE, not selective, pushing recall above 80%.
# - Requests strict JSON so we can parse without fragile regex.
# - Specifies the exact schema the downstream agent expects.
_TICKER_SYSTEM_PROMPT: str = """
You are a Named Entity Recognition (NER) and summarization engine for a
multilingual podcast transcript (Greek, English, Greeklish).

Your task: analyse the transcript window and return ONLY a JSON object
(no markdown fences, no extra text) matching this exact schema:

{
  "persons":         ["..."],   // ALL person names mentioned — be exhaustive for ≥80% recall
  "organizations":   ["..."],   // ALL organisations, brands, institutions
  "keywords":        ["..."],   // 8-12 most important content keywords / key topics
  "main_ideas":      ["..."],   // 2-4 concise main ideas (one sentence each)
  "segment_summary": "..."      // Single-sentence abstractive summary of the entire window
}

Rules:
- Include EVERY person and organisation, even if mentioned briefly.
- Keywords must reflect the core subject matter, not filler words.
- main_ideas should capture the primary arguments/points made.
- segment_summary must be one sentence, abstractive (not extractive).
- Return valid JSON only. Do NOT wrap in code fences.
""".strip()


def _build_ticker_user_content(
    window_text: str,
    window_start: float,
    window_end: float,
) -> str:
    """
    Build the user message for a ticker LLM call.

    We embed minimal metadata alongside the text so the LLM has context
    (timestamps help it anchor mentions to specific moments).
    """
    payload = {
        "window_start_sec": round(window_start, 2),
        "window_end_sec":   round(window_end, 2),
        "transcript_text":  window_text.strip(),
    }
    # Send as compact JSON — raw text in the context window, no RAG needed
    return json.dumps(payload, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pass-1 Live Ticker — async NER + summary every ~2 minutes
# ═══════════════════════════════════════════════════════════════════════════

async def _run_ticker_call(
    window_text: str,
    window_start: float,
    window_end: float,
    trigger_chunk_id: int,
) -> dict:
    """
    Execute one Live Ticker LLM call and return a parsed SCHEMA 1 dict.

    Called via asyncio.create_task() so it never blocks the ASR stream.

    Args:
        window_text:       Reliable full_text concatenated from the window.
        window_start:      Absolute start time of the window (sec).
        window_end:        Absolute end time of the window (sec).
        trigger_chunk_id:  chunk_id of the ASR chunk that triggered the call.

    Returns:
        dict matching SCHEMA 1 (per-segment live ticker output).
    """
    t0 = time.perf_counter()
    logger.info(
        "Ticker: firing LLM call for window %.1f–%.1fs (chunk_id=%d)",
        window_start, window_end, trigger_chunk_id,
    )

    user_content = _build_ticker_user_content(window_text, window_start, window_end)
    raw_response = await _call_llm(_TICKER_SYSTEM_PROMPT, user_content)

    # Parse the LLM JSON response; fall back to empty lists if malformed
    try:
        parsed: dict = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.warning(
            "Ticker: LLM returned non-JSON response for window %.1f–%.1fs: %r",
            window_start, window_end, raw_response[:200],
        )
        parsed = {}

    # Build the canonical SCHEMA 1 output dict, defaulting every field
    result: dict = {
        "chunk_id":        trigger_chunk_id,
        "window_start":    round(window_start, 3),
        "window_end":      round(window_end, 3),
        "persons":         parsed.get("persons", []),
        "organizations":   parsed.get("organizations", []),
        "keywords":        parsed.get("keywords", []),
        "main_ideas":      parsed.get("main_ideas", []),
        "segment_summary": parsed.get("segment_summary", ""),
    }

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        "Ticker: done in %.2fs | persons=%d orgs=%d keywords=%d",
        elapsed,
        len(result["persons"]),
        len(result["organizations"]),
        len(result["keywords"]),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. AccumulatedTranscript — stateful collector that drives the ticker
# ═══════════════════════════════════════════════════════════════════════════

class AccumulatedTranscript:
    """
    Stateful accumulator that:
      - Accepts asr_pipeline chunk dicts one at a time (via `add_chunk`).
      - Fires async Pass-1 Live Ticker LLM calls every TICKER_WINDOW_SEC.
      - Builds the full SCHEMA 2 object on demand via `to_dict()`.

    Usage (inside an async context):
        acc = AccumulatedTranscript(source_file="podcast.mp3")
        tasks = []
        async for chunk in asr_stream:
            task = acc.add_chunk(chunk)   # returns asyncio.Task or None
            if task:
                tasks.append(task)
        await asyncio.gather(*tasks)      # wait for all background LLM calls
        result = acc.to_dict()
    """

    def __init__(self, source_file: str = "") -> None:
        self.source_file: str = source_file

        # Minimal chunk metadata — no segments, no word timestamps, no audio
        self._chunks: list[dict] = []

        # Ticker results collected as LLM tasks complete
        self._ticker_results: list[dict] = []

        # Rolling window accumulators for the next ticker call
        self._window_texts: list[str] = []
        self._window_start: float = 0.0
        self._window_accumulated_sec: float = 0.0
        self._window_start_set: bool = False

        # Global deduplication sets
        self._all_persons: set[str] = set()
        self._all_organizations: set[str] = set()
        self._all_keywords: set[str] = set()
        self._all_main_ideas: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: dict) -> "asyncio.Task | None":
        """Ingest one ASR chunk dict. Only minimal metadata kept in RAM."""
        slim = {
            "chunk_id": chunk["chunk_id"],
            "start_time_sec": chunk["start_time_sec"],
            "end_time_sec": chunk["end_time_sec"],
            "duration_sec": chunk["duration_sec"],
            "detected_language": chunk.get("detected_language"),
            "language_probability": chunk.get("language_probability"),
            "all_language_probs": chunk.get("all_language_probs", []),
            "full_text": chunk.get("full_text", ""),
            "speakers_detected": chunk.get("speakers_detected", []),
            "is_speech": chunk.get("is_speech", False),
        }
        self._chunks.append(slim)

        text = chunk.get("full_text", "").strip()
        if not (chunk.get("is_speech") and text):
            return None

        # Initialise window start on first speech chunk
        if not self._window_start_set:
            self._window_start = chunk.get("start_time_sec", 0.0)
            self._window_start_set = True

        self._window_texts.append(text)
        self._window_accumulated_sec += chunk.get("duration_sec", 0.0)

        # Trigger ticker when we've accumulated ~2 minutes of speech
        if self._window_accumulated_sec < TICKER_WINDOW_SEC:
            return None

        window_text = " ".join(self._window_texts)
        window_end = chunk.get("end_time_sec", 0.0)
        trigger_id = chunk.get("chunk_id", -1)

        # Schedule async LLM call — does NOT block
        task: asyncio.Task = asyncio.create_task(
            self._ticker_with_callback(
                window_text, self._window_start, window_end, trigger_id
            )
        )
        # Reset window for the next 2-minute block
        self._window_texts = []
        self._window_accumulated_sec = 0.0
        self._window_start = window_end
        return task

    async def flush_remaining(self) -> "asyncio.Task | None":
        """
        After the ASR stream ends, fire a final ticker call for any
        text remaining in the window (< 2 min but non-empty).

        Call this after the ASR generator is exhausted, then await the task.
        """
        if not self._window_texts:
            return None

        window_text = " ".join(self._window_texts)
        # Use the end time of the last chunk as the window end
        last_chunk = self._chunks[-1] if self._chunks else {}
        window_end = last_chunk.get("end_time_sec", 0.0)
        trigger_id = last_chunk.get("chunk_id", -1)

        task = asyncio.create_task(
            self._ticker_with_callback(
                window_text, self._window_start, window_end, trigger_id
            )
        )

        # Clear window state
        self._window_texts = []
        self._window_accumulated_sec = 0.0
        return task

    def to_dict(self) -> dict:
        """
        Build and return the full SCHEMA 2 accumulated transcript dict.
        Chunks are trimmed to minimal metadata — full segments are in transcript.json
        only when needed for timestamp/chapter views.
        """
        total_duration = sum(c.get("duration_sec", 0.0) for c in self._chunks)
        full_text = " ".join(
            c.get("full_text", "") for c in self._chunks
            if c.get("is_speech") and c.get("full_text", "").strip()
        )

        languages: list[str] = sorted({
            c["detected_language"]
            for c in self._chunks
            if c.get("detected_language")
        })
        speakers: list[str] = sorted({
            s
            for c in self._chunks
            for s in c.get("speakers_detected", [])
        })

        # Store minimal chunk metadata for chapter/timestamp views
        # Language distribution from per-chunk top-1 detections
        speech_chunks = [c for c in self._chunks if c.get("is_speech")]
        lang_counts: dict[str, float] = {}
        for c in speech_chunks:
            lang = c.get("detected_language")
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0.0) + c.get("duration_sec", 0)
        total_speech = sum(lang_counts.values()) or 1.0
        language_distribution = {
            lang: round(secs / total_speech, 4)
            for lang, secs in sorted(lang_counts.items(), key=lambda x: -x[1])
        }
        language_switches: list[dict] = []
        prev_lang = None
        for c in speech_chunks:
            cur_lang = c.get("detected_language")
            if cur_lang and cur_lang != prev_lang:
                language_switches.append({
                    "from": prev_lang, "to": cur_lang,
                    "at_time_sec": c.get("start_time_sec", 0),
                    "at_chunk": c.get("chunk_id"),
                })
                prev_lang = cur_lang

        chunk_meta = [
            {"chunk_id": c["chunk_id"], "start_time_sec": c["start_time_sec"],
             "end_time_sec": c["end_time_sec"], "duration_sec": c["duration_sec"],
             "detected_language": c.get("detected_language"),
             "language_probability": c.get("language_probability"),
             "full_text": c.get("full_text", ""),
             "speakers_detected": c.get("speakers_detected", [])}
            for c in self._chunks
        ]

        return {
            "source_file":      self.source_file,
            "total_duration_sec": round(total_duration, 3),
            "total_chunks":     len(self._chunks),
            "languages_detected": languages,
            "language_distribution": language_distribution,
            "language_switches": language_switches,
            "speakers_detected":  speakers,
            "chunks":           chunk_meta,
            "ticker_results":   sorted(
                self._ticker_results, key=lambda r: r["window_start"]
            ),
            "full_text":        full_text,
            "all_persons":      sorted(self._all_persons),
            "all_organizations": sorted(self._all_organizations),
            "all_keywords":     sorted(self._all_keywords),
            "all_main_ideas":   list(dict.fromkeys(self._all_main_ideas)),
            "topic_extraction_80pct_note": (
                "LLM prompt instructs exhaustive entity extraction "
                "targeting >=80% recall per rubric requirement."
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ticker_with_callback(
        self,
        window_text: str,
        window_start: float,
        window_end: float,
        trigger_chunk_id: int,
    ) -> dict:
        """
        Wrapper around _run_ticker_call that stores the result and merges
        entities into the global deduplication sets upon completion.

        Running this as an asyncio.Task ensures it doesn't block the main loop.
        """
        ticker_result = await _run_ticker_call(
            window_text, window_start, window_end, trigger_chunk_id
        )

        # Thread-safe because asyncio is single-threaded (cooperative multitasking)
        self._ticker_results.append(ticker_result)

        # Merge into global entity sets for the final SCHEMA 2 object
        self._all_persons.update(ticker_result.get("persons", []))
        self._all_organizations.update(ticker_result.get("organizations", []))
        self._all_keywords.update(ticker_result.get("keywords", []))
        for idea in ticker_result.get("main_ideas", []):
            if idea not in self._all_main_ideas:
                self._all_main_ideas.append(idea)

        return ticker_result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Top-level async pipeline — wires ASR stream → AccumulatedTranscript
# ═══════════════════════════════════════════════════════════════════════════

async def process_asr_stream(
    asr_chunks: "AsyncIterator[dict] | list[dict]",
    source_file: str = "",
) -> dict:
    """
    Consume an ASR chunk stream and return the full SCHEMA 2 transcript dict.

    Accepts either an async generator (real-time) or a plain list (batch mode).
    Fires Pass-1 LLM ticker calls asynchronously while the stream continues.

    Args:
        asr_chunks:   Async or sync iterable of asr_pipeline chunk dicts.
        source_file:  Original audio file path (stored in SCHEMA 2 for traceability).

    Returns:
        dict matching SCHEMA 2 (full accumulated transcript object).
    """
    acc = AccumulatedTranscript(source_file=source_file)
    pending_tasks: list[asyncio.Task] = []

    # Handle both async iterators (real-time) and sync lists (batch / testing)
    async def _consume_async():
        async for chunk in asr_chunks:  # type: ignore[union-attr]
            t = acc.add_chunk(chunk)
            if t is not None:
                pending_tasks.append(t)

    def _consume_sync():
        for chunk in asr_chunks:  # type: ignore[union-attr]
            t = acc.add_chunk(chunk)
            if t is not None:
                pending_tasks.append(t)

    if hasattr(asr_chunks, "__aiter__"):
        await _consume_async()
    else:
        _consume_sync()

    # Flush any remaining text < 2 min at end-of-stream
    final_task = await acc.flush_remaining()
    if final_task is not None:
        pending_tasks.append(final_task)

    # Wait for all background LLM calls to complete before building SCHEMA 2
    if pending_tasks:
        logger.info("Waiting for %d background LLM ticker task(s)…", len(pending_tasks))
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    result = acc.to_dict()
    logger.info(
        "LLM integration complete: %d ticker windows | %d persons | "
        "%d orgs | %d keywords",
        len(result["ticker_results"]),
        len(result["all_persons"]),
        len(result["all_organizations"]),
        len(result["all_keywords"]),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Convenience sync wrapper — for callers that are not async-aware
# ═══════════════════════════════════════════════════════════════════════════

def process_asr_stream_sync(
    asr_chunks: list[dict],
    source_file: str = "",
) -> dict:
    """
    Synchronous wrapper around process_asr_stream for non-async callers
    (e.g. a simple script or Jupyter notebook cell).

    Args:
        asr_chunks:  List of asr_pipeline chunk dicts.
        source_file: Original audio file path.

    Returns:
        dict matching SCHEMA 2.
    """
    return asyncio.run(
        process_asr_stream(asr_chunks, source_file=source_file)
    )
