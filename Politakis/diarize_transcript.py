"""
diarize_transcript.py — Convert transcript.json to speaker-prefixed text.

Reads Politakis/results/transcript.json, extracts per-segment speaker labels
from the ASR pipeline output, and produces a .txt file with one speaker turn
per line, prefixed with "Speaker A:", "Speaker B:", etc.

This is the structured input for the podcast script generation LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def diarize_transcript(
    transcript_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Convert transcript.json segments to speaker-prefixed text.

    Each line in the output is: "Speaker X: chunk text here"
    One line per speaker turn — newlines mark speaker changes.

    Args:
        transcript_path: Path to transcript.json (default: results/transcript.json)
        output_path: Path for output (default: results/normalized_diarized_transcript.txt)

    Returns:
        Path to the generated diarized transcript file.
    """
    if transcript_path is None:
        transcript_path = RESULTS_DIR / "transcript.json"
    if output_path is None:
        output_path = RESULTS_DIR / "normalized_diarized_transcript.txt"

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    lines: list[str] = []
    chunks = transcript.get("chunks", [])

    for chunk in chunks:
        speaker = chunk.get("speaker", "Speaker A")
        # Normalize to "Speaker X" format — handle whatever the ASR returns
        if isinstance(speaker, str) and speaker.startswith("SPEAKER_"):
            speaker = "Speaker " + speaker.replace("SPEAKER_", "")
        elif isinstance(speaker, str) and not speaker.startswith("Speaker"):
            speaker = f"Speaker {speaker}"

        text = chunk.get("text", "")
        if not text.strip():
            continue

        # Use normalized text if available, otherwise raw
        # The "text" field should already be the normalized version
        # after transcript_normalizer has run (it replaces full_text in place)
        clean = text.strip()
        if clean:
            lines.append(f"{speaker}: {clean}")

    output_text = "\n".join(lines)

    output_path.write_text(output_text, encoding="utf-8")
    logger.info(
        "Diarized transcript: %d speaker turns → %s (%d bytes)",
        len(lines), output_path.name, output_path.stat().st_size,
    )

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    diarize_transcript()
