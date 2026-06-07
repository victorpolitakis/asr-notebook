"""
strip_newlines.py — Produce a flat (newline-stripped) version of the
diarized transcript for WER evaluation.

WER metrics compare character sequences — sentence structure matters but
speaker labels and line breaks don't. This script strips speaker prefixes
and joins all lines with spaces, producing a clean flat text that can be
compared against ground truth.

Input:  results/normalized_diarized_transcript.txt  (speaker-prefixed, one turn/line)
Output: results/normalized_transcript_flat.txt       (no prefixes, single paragraph)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def strip_for_wer(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Remove speaker prefixes and join lines for WER evaluation.

    Consumes "Speaker A: text\\nSpeaker B: text\\n" and produces
    "text text text" (single line, space-separated).

    Args:
        input_path: Path to diarized transcript (default: results/normalized_diarized_transcript.txt)
        output_path: Path for flat output (default: results/normalized_transcript_flat.txt)

    Returns:
        Path to the flat transcript file.
    """
    if input_path is None:
        input_path = RESULTS_DIR / "normalized_diarized_transcript.txt"
    if output_path is None:
        output_path = RESULTS_DIR / "normalized_transcript_flat.txt"

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        # Fall back to reading normalized_transcript.txt directly
        fallback = RESULTS_DIR / "normalized_transcript.txt"
        if fallback.exists():
            input_path = fallback
            logger.warning("Falling back to: %s", fallback)
        else:
            raise FileNotFoundError(f"No diarized transcript found at {input_path}")

    text = input_path.read_text(encoding="utf-8")

    # Strip "Speaker X:" prefixes (CORRECT — no literal brackets)
    flat = re.sub(r"Speaker [A-Z]:\s*", "", text)

    # Join all lines into a single space-separated block
    flat = " ".join(line.strip() for line in flat.splitlines() if line.strip())

    output_path.write_text(flat, encoding="utf-8")
    logger.info(
        "Flat WER transcript: %d chars → %s",
        len(flat), output_path.name,
    )

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    strip_for_wer()
