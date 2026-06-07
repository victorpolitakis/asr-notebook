"""
transcript_normalizer.py — LLM-based ASR transcript normalization.

Uses a lightweight OpenAI model to correct proper nouns, entity names, and
technical terms in multilingual (Greek + English) transcripts. Designed
for production-grade safety: anti-hallucination validation, retry logic,
paragraph-aware chunking for long-form audio, and full feature-flag control.
"""

import difflib
import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Configuration (all env-overridable) ────────────────────────────────
# Auto-load OPENAI_API_KEY from: 1) env, 2) .env file, 3) Colab secrets
if not os.environ.get("OPENAI_API_KEY"):
    # Try .env files
    for env_path in (Path(__file__).parent / ".env", Path.home() / ".env", Path(".env")):
        try:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        os.environ.setdefault(key, val)
        except Exception:
            pass
    # Try Colab secrets
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            from google.colab import userdata
            os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
        except Exception:
            pass

NORMALIZATION_MODEL = os.getenv("NORMALIZATION_MODEL", "gpt-5.4-mini-2026-03-17")
ENABLE_NORMALIZATION = (
    os.getenv("ENABLE_TRANSCRIPT_NORMALIZATION", "true").lower() == "true"
)
MAX_NORMALIZATION_CHARS = int(os.getenv("MAX_NORMALIZATION_CHARS", "8000"))
MAX_RETRIES = 2
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 4096  # intentionally high — output ≈ input, never truncate

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Prompt (benchmark-proven Variant C) ───────────────────────────
NORMALIZATION_PROMPT = """\
You are repairing a multilingual ASR transcript.

Do ONLY these:
1. Restore corrupted person names (e.g. "yan lecun" → "Yann LeCun",
   "Ιαν Λε Κων" → "Yann LeCun", "jeffrey hinton" → "Geoffrey Hinton").
2. Restore corrupted organization names (e.g. "openai" → "OpenAI",
   "google deepmind" → "Google DeepMind").
3. Restore corrupted technical terms (e.g. "api silicon" → "Apple Silicon",
   "gpt 4" → "GPT-4", "metal performance aders" → "Metal Performance Shaders").
4. Fix capitalization of known entities when confidence is high.
5. Fix obvious tokenization issues (e.g. "CTranslate2" → "CTranslate2").

Do NOT:
- Summarize, paraphrase, reorder, translate, or rewrite sentences.
- Improve grammar, style, or wording.
- Add or remove any information.
- Modify anything other than proper nouns, entities, and technical terms.

If confidence in a correction is low, leave the text unchanged.

Return ONLY the repaired transcript — no markdown, no explanations."""


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def normalize_transcript(raw_text: str) -> str | None:
    """
    Normalize an ASR transcript via LLM.

    If the transcript exceeds MAX_NORMALIZATION_CHARS, it is split into
    paragraph-aware chunks, normalized independently, and re-assembled.

    Returns:
        Normalized text on success, None on any failure (caller falls back to raw).
    """
    if not raw_text.strip():
        return None

    if not LLM_API_KEY:
        logger.warning("OPENAI_API_KEY not set — skipping normalization.")
        return None

    # ── Chunking for long-form transcripts ──────────────────────────
    if len(raw_text) <= MAX_NORMALIZATION_CHARS:
        chunks = [raw_text]
    else:
        chunks = _split_into_chunks(raw_text, MAX_NORMALIZATION_CHARS)
        logger.info(
            "Transcript exceeds normalization threshold (%d chars). "
            "Processing in %d normalization chunks.",
            len(raw_text), len(chunks),
        )

    # ── Normalize each chunk ────────────────────────────────────────
    normalized_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        result = _call_normalization_api(chunk)
        if result is None:
            logger.warning("Normalization chunk %d/%d failed — discarding entire pass.", i + 1, len(chunks))
            return None
        normalized_parts.append(result)

    normalized = "\n\n".join(normalized_parts)

    # ── Structural validation ───────────────────────────────────────
    if not _validate_normalization(raw_text, normalized):
        logger.warning(
            "Normalization validation failed. "
            "Discarding normalized transcript and using raw transcript."
        )
        return None

    # ── Edit region logging ─────────────────────────────────────────
    changes = list(difflib.SequenceMatcher(None, raw_text, normalized).get_opcodes())
    correction_count = sum(1 for tag, *_ in changes if tag != "equal")
    logger.info(
        "Normalization: %d chars → %d chars | %d edit regions",
        len(raw_text), len(normalized), correction_count,
    )

    return normalized


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _call_normalization_api(text: str) -> str | None:
    """Single LLM call with retry logic. Returns normalized text or None."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                timeout=60.0,
            )
            response = client.chat.completions.create(
                model=NORMALIZATION_MODEL,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                messages=[
                    {"role": "system", "content": NORMALIZATION_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            normalized = response.choices[0].message.content or ""
            if not normalized.strip():
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Normalization returned empty — retry %d/%d",
                        attempt + 1, MAX_RETRIES,
                    )
                    continue
                return None
            return normalized
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Normalization API error (attempt %d/%d): %s — retrying…",
                    attempt + 1, MAX_RETRIES, e,
                )
                continue
            logger.warning("Normalization failed after %d retries: %s", MAX_RETRIES + 1, e)
            return None


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks on paragraph boundaries.
    Never splits inside speaker labels or timestamps.
    Falls back to sentence boundaries for very long paragraphs.
    """
    # Split on double newline (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If a single paragraph exceeds max_chars, split on sentences
        if para_len > max_chars:
            # Flush current chunk first
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            # Split long paragraph on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_chunk: list[str] = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) > max_chars and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_len = 0
                sent_chunk.append(sent)
                sent_len += len(sent) + 1
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
            continue

        # Add paragraph to current chunk if it fits
        if current_len + para_len + 2 <= max_chars:
            current.append(para)
            current_len += para_len + 2
        else:
            # Flush current and start new
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    # If somehow we produced no chunks, return the full text as one
    if not chunks:
        return [text]
    return chunks


def _validate_normalization(raw: str, normalized: str) -> bool:
    """
    Structural validation to prevent hallucination/summarisation.
    Returns True if normalized output is structurally sound.
    Checks are conditional — auto-pass if raw has no speakers or timestamps.
    """
    # Check 1: length ratio
    ratio = len(normalized) / max(len(raw), 1)
    if ratio < 0.85 or ratio > 1.15:
        logger.warning(
            "Normalization length ratio out of bounds: %.3f (raw=%d, norm=%d)",
            ratio, len(raw), len(normalized),
        )
        return False

    # Check 2: speaker label preservation (auto-PASS if raw has none)
    raw_speakers = len(re.findall(r"\[Speaker [A-Z]+\]", raw))
    norm_speakers = len(re.findall(r"\[Speaker [A-Z]+\]", normalized))
    if raw_speakers > 0:
        spk_ratio = norm_speakers / raw_speakers
        if spk_ratio < 0.9:
            logger.warning(
                "Speaker label preservation failed: raw=%d, norm=%d (ratio=%.2f)",
                raw_speakers, norm_speakers, spk_ratio,
            )
            return False

    # Check 3: timestamp preservation (auto-PASS if raw has none)
    raw_ts = len(re.findall(r"\[\d{2}:\d{2}\]", raw))
    norm_ts = len(re.findall(r"\[\d{2}:\d{2}\]", normalized))
    if raw_ts > 0:
        ts_ratio = norm_ts / raw_ts
        if ts_ratio < 0.9:
            logger.warning(
                "Timestamp preservation failed: raw=%d, norm=%d (ratio=%.2f)",
                raw_ts, norm_ts, ts_ratio,
            )
            return False

    # Check 4: paragraph preservation
    raw_paras = len([p for p in re.split(r"\n\n+", raw) if p.strip()])
    norm_paras = len([p for p in re.split(r"\n\n+", normalized) if p.strip()])
    if raw_paras > 0:
        para_ratio = norm_paras / raw_paras
        if para_ratio < 0.8:
            logger.warning(
                "Paragraph preservation failed: raw=%d, norm=%d (ratio=%.2f)",
                raw_paras, norm_paras, para_ratio,
            )
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Entity Re-Extraction — run on normalized transcript
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_PROMPT = """\
Extract named entities from this transcript. Return ONLY valid JSON:

{
  "persons": ["name1", "name2"],
  "organizations": ["org1", "org2"],
  "keywords": ["keyword1", "keyword2"],
  "main_ideas": ["idea1", "idea2"]
}

Rules:
- Persons: real people mentioned (not speakers, not roles).
- Organizations: companies, universities, institutions, research labs.
- Keywords: technologies, models, frameworks, products, concepts, regulations.
- Main ideas: 2-4 key themes or arguments discussed.
- Use canonical forms (e.g., "Yann LeCun" not "yan lecun").
- Return ONLY the JSON object — no markdown, no explanations."""


def re_extract_entities(normalized_transcript: dict) -> dict | None:
    """
    Re-extract entities from the normalized transcript text.
    This catches entities that the live ticker missed due to ASR errors
    in the raw transcript.

    Returns dict with {"persons": [...], "organizations": [...], "keywords": [...]}
    or None on failure.
    """
    if not LLM_API_KEY or not ENABLE_NORMALIZATION:
        return None

    full_text = normalized_transcript.get("full_text", "")
    if not full_text.strip():
        return None

    logger.info("Re-extracting entities from normalized transcript (%d chars)…", len(full_text))

    # Truncate to a reasonable size for entity extraction
    text_for_ner = full_text[:MAX_NORMALIZATION_CHARS]

    for attempt in range(MAX_RETRIES + 1):
        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60.0)
            response = client.chat.completions.create(
                model=NORMALIZATION_MODEL,
                temperature=TEMPERATURE,
                max_completion_tokens=1024,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                    {"role": "user", "content": text_for_ner},
                ],
            )
            raw = response.choices[0].message.content or ""
            # Strip markdown fences if present
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            import json
            entities = json.loads(raw)
            logger.info(
                "Re-extracted entities: %d persons, %d orgs, %d keywords",
                len(entities.get("persons", [])),
                len(entities.get("organizations", [])),
                len(entities.get("keywords", [])),
            )
            return entities
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Entity re-extraction failed (attempt %d/%d): %s — retrying…",
                              attempt + 1, MAX_RETRIES, e)
                continue
            logger.warning("Entity re-extraction failed after %d retries: %s", MAX_RETRIES + 1, e)
            return None
