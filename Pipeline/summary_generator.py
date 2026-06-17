"""
summary_generator.py — Section 4: Summary Generation (15 pts)

Architecture rules applied (implementation_plan.md §Overarching):
  1. Strict Python Type Hints on every public function.
  2. All data exchanged as parsed JSON dicts — never raw multi-line strings.
  3. MPS device targeting — no local embedding math here; all LLM calls are
     remote (OpenAI API).  Any future local re-ranking step should target MPS.
  4. Every non-trivial block has a "why" comment, not just a "what".

Responsibility:
  Pass 2 — triggered by EOF signal (all ASR chunks + Pass-1 tickers done).
  Consumes:
    • SCHEMA 2 dict  (llm_integration.AccumulatedTranscript.to_dict())
    • EntityRegistryDict  (topic_extraction.build_entity_registry())
  Produces three summary levels + YouTube chapters via a single LLM call per
  level, then persists the full payload to results/summary_outputs.json.

  Also exposes query_transcript() — the Q&A backend for streamlit_app.py.

=============================================================================
OUTPUT SCHEMA — SummaryOutputs  (written to results/summary_outputs.json)
=============================================================================
{
    "source_file":   str,               # Echoed from SCHEMA 2
    "generated_at":  str,               # ISO-8601 UTC timestamp
    "chapters": [                       # YouTube-style timestamped chapters
        {
            "index":       int,         # 1-based chapter number
            "title":       str,         # Short chapter title
            "start_sec":   float,       # Chapter start time (seconds)
            "end_sec":     float,       # Chapter end time (seconds)
            "summary":     str          # 1-sentence chapter summary
        },
        ...
    ],
    "entities": { ...EntityRegistryDict... },   # Passed through unchanged
    "summaries": {
        "tldr": str,                    # Level 1 — 1-sentence overarching thesis
        "executive": str,               # Level 3 — 3-paragraph executive summary
        "deep_dive": {
            "overview":     str,        # Opening paragraph
            "bullet_points": list[str], # Key points as bullet strings
            "key_takeaways": list[str], # Distilled actionable takeaways
            "action_items":  list[str]  # Concrete next steps / recommendations
        }
    },
    "qa_logs": [                        # All Q&A interactions from Streamlit chat
        {
            "timestamp":  str,          # ISO-8601 UTC
            "question":   str,
            "answer":     str,
            "model":      str           # LLM model used
        },
        ...
    ]
}
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — single source of truth (mirrors llm_integration.py knobs)
# ---------------------------------------------------------------------------

# Reuse the same LLM config env vars as llm_integration.py so the operator
# only needs to configure one place.
LLM_BASE_URL: str   = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY: str    = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL: str      = os.environ.get("LLM_MODEL", "gpt-5.4-mini-2026-03-17")

# Pass-2 calls can be longer — allow more tokens for deep-dive output
LLM_MAX_TOKENS_SUMMARY: int = 2048
LLM_MAX_TOKENS_QA: int      = 1024
LLM_TEMPERATURE: float      = 0.3   # Slight creativity for readable prose

# Output file path — created relative to CWD; Streamlit CWD is project root
RESULTS_DIR: Path       = Path(__file__).parent.parent / "Results"
SUMMARY_OUTPUT_FILE: Path = RESULTS_DIR / "summary_outputs.json"


# ═══════════════════════════════════════════════════════════════════════════
# TypedDicts — strict type contracts for module boundaries
# ═══════════════════════════════════════════════════════════════════════════

class ChapterDict(TypedDict):
    index:     int
    title:     str
    start_sec: float
    end_sec:   float
    summary:   str


class DeepDiveDict(TypedDict):
    overview:      str
    bullet_points: list[str]
    key_takeaways: list[str]
    action_items:  list[str]


class SummariesDict(TypedDict):
    tldr:      str
    executive: str
    deep_dive: DeepDiveDict


class QALogEntry(TypedDict):
    timestamp: str
    question:  str
    answer:    str
    model:     str


class SummaryOutputs(TypedDict):
    source_file:  str
    generated_at: str
    chapters:     list[ChapterDict]
    entities:     dict          # EntityRegistryDict (opaque pass-through)
    summaries:    SummariesDict
    qa_logs:      list[QALogEntry]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LLM Client — thin async wrapper (mirrors llm_integration pattern)
# ═══════════════════════════════════════════════════════════════════════════

def _build_openai_client() -> Any:
    """
    Lazily import and return AsyncOpenAI.
    Mirrors llm_integration._build_openai_client — kept separate so this module
    can be imported independently without llm_integration.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
        return AsyncOpenAI(api_key=LLM_API_KEY or "ollama", base_url=LLM_BASE_URL)
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc


async def _call_llm(
    system_prompt: str,
    user_content: str,
    max_tokens: int = LLM_MAX_TOKENS_SUMMARY,
) -> str:
    """
    Single async LLM call returning raw response text.
    Why async? Non-blocking — Streamlit can remain interactive while Pass-2 runs.
    """
    client = _build_openai_client()
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=max_tokens,
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
# 2. Prompt Engineering — Pass-2 YouTube Chapters
# ═══════════════════════════════════════════════════════════════════════════

_CHAPTERS_SYSTEM_PROMPT: str = """
You are a podcast chapter generator for YouTube.
Given a JSON object with segment_summaries and time windows, produce ONLY a
JSON array of chapter objects. No markdown fences. No extra text.

Each chapter object must match:
{
  "index":     <int, 1-based>,
  "title":     "<5-7 word descriptive chapter title>",
  "start_sec": <float>,
  "end_sec":   <float>,
  "summary":   "<one-sentence chapter summary>"
}

Rules:
- Merge adjacent ticker windows that cover the same logical topic.
- Chapter titles must be descriptive, not generic (e.g. NOT "Introduction").
- Return valid JSON array only.
""".strip()


async def _generate_chapters(transcript: dict) -> list[ChapterDict]:
    """
    Pass-2 YouTube Chapter generation.

    Why use ticker_results rather than raw full_text?
    - Ticker results already have timestamps and segment summaries — ideal
      for chapter boundary detection without re-parsing the entire transcript.
    - Sending the full text (potentially 50k+ tokens) would be wasteful here.

    Args:
        transcript: SCHEMA 2 dict from llm_integration.py.

    Returns:
        List of ChapterDict objects, sorted by start_sec.
    """
    ticker_results: list[dict] = transcript.get("ticker_results", [])
    if not ticker_results:
        logger.warning("No ticker results — cannot generate chapters.")
        return []

    # Build a compact chapter-hint payload from ticker window metadata
    # Why compact JSON instead of full text? Minimise token usage for this call.
    chapter_hints = [
        {
            "window":   i + 1,
            "start_sec": r.get("window_start", 0.0),
            "end_sec":   r.get("window_end", 0.0),
            "summary":   r.get("segment_summary", ""),
            "keywords":  r.get("keywords", [])[:5],   # top-5 to keep payload small
        }
        for i, r in enumerate(ticker_results)
    ]
    user_content = json.dumps(
        {"segment_hints": chapter_hints, "total_windows": len(ticker_results)},
        ensure_ascii=False,
    )

    raw = await _call_llm(_CHAPTERS_SYSTEM_PROMPT, user_content)
    try:
        parsed: list[dict] = json.loads(_strip_trailing_commas(raw))
    except json.JSONDecodeError:
        logger.warning("Chapter LLM returned non-JSON: %r", raw[:200])
        # Graceful fallback: one chapter per ticker window
        parsed = [
            {
                "index":     i + 1,
                "title":     f"Chapter {i + 1}",
                "start_sec": r.get("window_start", 0.0),
                "end_sec":   r.get("window_end", 0.0),
                "summary":   r.get("segment_summary", ""),
            }
            for i, r in enumerate(ticker_results)
        ]

    chapters: list[ChapterDict] = []
    for item in parsed:
        chapters.append(
            ChapterDict(
                index     = int(item.get("index", 0)),
                title     = str(item.get("title", "")),
                start_sec = float(item.get("start_sec", 0.0)),
                end_sec   = float(item.get("end_sec", 0.0)),
                summary   = str(item.get("summary", "")),
            )
        )
    # Ensure time-sorted output regardless of LLM ordering
    chapters.sort(key=lambda c: c["start_sec"])
    return chapters


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prompt Engineering — Three Summary Levels
# ═══════════════════════════════════════════════════════════════════════════

_TLDR_SYSTEM_PROMPT: str = """
You are a podcast summariser. Return ONLY a single sentence (max 30 words)
capturing the overarching thesis of the podcast. No fences, no labels.
""".strip()

_EXECUTIVE_SYSTEM_PROMPT: str = """
You are a professional podcast summariser writing for a senior executive.
Return ONLY three well-written paragraphs (no labels, no markdown fences,
no bullet points) that cover:
  1. Context and main topic.
  2. Key arguments and evidence.
  3. Conclusions and implications.
Plain prose only. No preamble.
""".strip()

_DEEP_DIVE_SYSTEM_PROMPT: str = """
You are a comprehensive podcast analyst. Return ONLY a JSON object (no markdown
fences) matching this exact schema:

{
  "overview":      "<opening paragraph, 3-5 sentences>",
  "bullet_points": ["<key point 1>", "<key point 2>", ...],
  "key_takeaways": ["<distilled insight 1>", ...],
  "action_items":  ["<concrete recommendation/next step 1>", ...]
}

Rules:
- bullet_points: 6-10 items, each one complete sentence.
- key_takeaways: 3-5 distilled insights (what the listener should remember).
- action_items:  3-5 actionable next steps or recommendations.
- Return valid JSON only.
""".strip()


def _build_summary_user_content(transcript: dict, entities: dict) -> str:
    """
    Compact user content for all three summary-level LLM calls.

    Why include both full_text and entity registry?
    - full_text gives the LLM the complete argument/narrative.
    - Entities add high-salience signals so the LLM leads with the right names.
    - Combined, this maximises ROUGE-1 recall without RAG.
    """
    payload = {
        "full_text":      transcript.get("full_text", ""),
        "main_ideas":     transcript.get("all_main_ideas", []),
        "top_persons":    [e["name"] for e in entities.get("persons", [])[:5]],
        "top_orgs":       [e["name"] for e in entities.get("organizations", [])[:5]],
        "top_keywords":   [e["name"] for e in entities.get("keywords", [])[:10]],
        "segment_summaries": entities.get("segment_summaries", []),
    }
    return json.dumps(payload, ensure_ascii=False)


async def _generate_tldr(user_content: str) -> str:
    """Level 1 — one-sentence thesis."""
    raw = await _call_llm(_TLDR_SYSTEM_PROMPT, user_content, max_tokens=100)
    return raw.strip()


async def _generate_executive(user_content: str) -> str:
    """Level 3 — three-paragraph executive summary."""
    raw = await _call_llm(_EXECUTIVE_SYSTEM_PROMPT, user_content, max_tokens=600)
    return raw.strip()


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before ] or } so LLM-generated JSON parses cleanly."""
    return re.sub(r',\s*([}\]])', r'\1', s)


async def _generate_deep_dive(user_content: str) -> DeepDiveDict:
    """Level 5 — bullet points, key takeaways, action items."""
    raw = await _call_llm(_DEEP_DIVE_SYSTEM_PROMPT, user_content, max_tokens=1200)
    try:
        parsed: dict = json.loads(_strip_trailing_commas(raw))
    except json.JSONDecodeError:
        logger.warning("Deep-dive LLM returned non-JSON: %r", raw[:200])
        # Graceful fallback — return empty structured object
        parsed = {
            "overview":      raw.strip(),
            "bullet_points": [],
            "key_takeaways": [],
            "action_items":  [],
        }
    return DeepDiveDict(
        overview      = str(parsed.get("overview", "")),
        bullet_points = [str(x) for x in parsed.get("bullet_points", [])],
        key_takeaways = [str(x) for x in parsed.get("key_takeaways", [])],
        action_items  = [str(x) for x in parsed.get("action_items", [])],
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Q&A — persistent chat backend used by streamlit_app.py
# ═══════════════════════════════════════════════════════════════════════════

_QA_SYSTEM_PROMPT: str = """
You are a precise Q&A assistant answering questions about a podcast transcript.
Answer concisely using ONLY information present in the transcript provided.
If the answer is not in the transcript, say so explicitly.
Return plain prose — no markdown fences.
""".strip()


async def query_transcript_async(
    question: str,
    transcript: dict,
) -> str:
    """
    Answer a free-form question about the transcript.  Called by streamlit_app.py
    on every chat submission.

    Why send full_text + entities?
    - No RAG/vector store — the whole transcript fits in 128k context.
    - Entities add high-salience anchors so the LLM finds relevant passages fast.

    Args:
        question:   User's natural-language question.
        transcript: SCHEMA 2 dict (llm_integration output).

    Returns:
        Plain-text answer string.
    """
    payload = {
        "question":  question,
        "full_text": transcript.get("full_text", ""),
        "persons":   transcript.get("all_persons", []),
        "orgs":      transcript.get("all_organizations", []),
        "keywords":  transcript.get("all_keywords", []),
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    answer = await _call_llm(_QA_SYSTEM_PROMPT, user_content, max_tokens=LLM_MAX_TOKENS_QA)
    return answer.strip()


def _call_llm_sync(
    system_prompt: str,
    user_content: str,
    max_tokens: int = LLM_MAX_TOKENS_SUMMARY,
) -> str:
    """Synchronous LLM call using OpenAI client — safe for Streamlit callbacks."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY or "ollama", base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


def query_transcript(question: str, transcript: dict) -> str:
    """
    Answer a free-form question about the transcript.
    Uses synchronous LLM client to avoid asyncio event-loop conflicts in Streamlit.

    Args:
        question:   User's natural-language question.
        transcript: SCHEMA 2 dict (llm_integration output).

    Returns:
        Plain-text answer string.
    """
    payload = {
        "question":  question,
        "full_text": transcript.get("full_text", ""),
        "persons":   transcript.get("all_persons", []),
        "orgs":      transcript.get("all_organizations", []),
        "keywords":  transcript.get("all_keywords", []),
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    answer = _call_llm_sync(_QA_SYSTEM_PROMPT, user_content, max_tokens=LLM_MAX_TOKENS_QA)
    return answer.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Pass-2 Orchestrator — main public entry point
# ═══════════════════════════════════════════════════════════════════════════

async def generate_summary_async(
    transcript: dict,
    entities: dict,
) -> SummaryOutputs:
    """
    Pass-2 EOF-triggered summarisation pipeline.

    Fires four independent LLM calls concurrently (asyncio.gather) to minimise
    wall-clock latency:
      1. YouTube chapters
      2. TL;DR (Level 1)
      3. Executive Summary (Level 3)
      4. Deep Dive (Level 5)

    Args:
        transcript: SCHEMA 2 dict from llm_integration.process_asr_stream().
        entities:   EntityRegistryDict from topic_extraction.build_entity_registry().

    Returns:
        SummaryOutputs dict (also written to results/summary_outputs.json).
    """
    logger.info("Pass-2: starting summary generation for '%s'", transcript.get("source_file", ""))

    # Build shared user content once — reused across all three summary levels
    # Why reuse? Avoids re-serialising the same large payload multiple times.
    summary_user_content = _build_summary_user_content(transcript, entities)

    # Fire all four LLM calls concurrently — reduces total latency by ~75%
    # compared to sequential calls.
    chapters, tldr, executive, deep_dive = await asyncio.gather(
        _generate_chapters(transcript),
        _generate_tldr(summary_user_content),
        _generate_executive(summary_user_content),
        _generate_deep_dive(summary_user_content),
    )

    result = SummaryOutputs(
        source_file  = transcript.get("source_file", ""),
        generated_at = datetime.now(timezone.utc).isoformat(),
        chapters     = chapters,
        entities     = dict(entities),       # pass-through, no transformation
        summaries    = SummariesDict(
            tldr      = tldr,
            executive = executive,
            deep_dive = deep_dive,
        ),
        qa_logs = [],   # Populated incrementally by streamlit_app.py
    )

    # Persist to disk immediately so Streamlit can load it even if it starts late
    _save_to_disk(result)

    logger.info(
        "Pass-2 complete: %d chapters | tldr=%d chars | exec=%d chars",
        len(chapters), len(tldr), len(executive),
    )
    return result


def generate_summary(
    transcript: dict,
    entities: dict,
) -> SummaryOutputs:
    """
    Synchronous wrapper around generate_summary_async.
    Use this from scripts, Jupyter notebooks, or any non-async context.

    Args:
        transcript: SCHEMA 2 dict.
        entities:   EntityRegistryDict.

    Returns:
        SummaryOutputs dict.
    """
    return asyncio.run(generate_summary_async(transcript, entities))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Persistence helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_to_disk(outputs: SummaryOutputs) -> None:
    """
    Persist SummaryOutputs to results/summary_outputs.json.

    Why results/ subdirectory?
    - Keeps generated artifacts separate from source code.
    - Mirrors rubric requirement: "Save the final result payload cleanly to
      results/summary_outputs.json".
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2, ensure_ascii=False)
    logger.info("Summary outputs saved → %s", SUMMARY_OUTPUT_FILE)


def append_qa_log(entry: QALogEntry) -> None:
    """
    Append one Q&A log entry to results/summary_outputs.json.

    Called by streamlit_app.py after every chat interaction so the QA log
    grows incrementally without rewriting the full file from scratch.
    """
    if not SUMMARY_OUTPUT_FILE.exists():
        logger.warning("append_qa_log: %s not found — skipping.", SUMMARY_OUTPUT_FILE)
        return
    data: dict = json.loads(SUMMARY_OUTPUT_FILE.read_text(encoding="utf-8"))
    data.setdefault("qa_logs", []).append(entry)
    SUMMARY_OUTPUT_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_summary_outputs() -> SummaryOutputs | None:
    """
    Load a previously persisted SummaryOutputs dict from disk.
    Returns None if the file doesn't exist (pipeline hasn't run yet).
    """
    if not SUMMARY_OUTPUT_FILE.exists():
        return None
    return json.loads(SUMMARY_OUTPUT_FILE.read_text(encoding="utf-8"))
