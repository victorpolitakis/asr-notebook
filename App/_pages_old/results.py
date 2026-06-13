"""
pages/results.py — Transcript viewer, entity registry, multi-tier summaries.

Reads cached data from st.session_state (populated by the Upload page).
Never hits the Drive API directly — this page is pure display.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    transcript = st.session_state.get("transcript") or {}
    summary = st.session_state.get("summary") or {}
    pstate = st.session_state.get("pipeline_state", "idle")

    st.markdown("## Results")
    st.caption("Transcript, entities, and summaries from the last processed file.")

    if pstate != "done" or (not transcript and not summary):
        _render_empty()
        return

    # ── Tab layout ──
    t1, t2, t3 = st.tabs(["Transcript", "Entities", "Summaries"])

    with t1:
        _render_transcript(transcript)

    with t2:
        _render_entities(summary)

    with t3:
        _render_summaries(summary)


# ── Empty state ────────────────────────────────────────────────────

def _render_empty() -> None:
    st.info("No results yet. Upload and transcribe an audio file on the **Upload** page first.")
    if st.button("Go to Upload →"):
        st.session_state["_page"] = "Upload"
        st.rerun()


# ── Transcript ─────────────────────────────────────────────────────

def _render_transcript(data: dict) -> None:
    chunks = data.get("chunks", [])
    duration = data.get("total_duration_sec", 0)
    langs = data.get("languages_detected", [])
    full_text = data.get("full_text", "")

    all_segments: list[dict] = []
    for chunk in chunks:
        for seg in chunk.get("segments", []):
            if seg.get("text", "").strip():
                all_segments.append({
                    "speaker": seg.get("speaker", "Speaker A"),
                    "text": seg["text"].strip(),
                    "start": seg.get("start", chunk.get("start_time_sec", 0)),
                })

    speakers = list({s["speaker"] for s in all_segments})

    # Metrics row
    if duration:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration", f"{duration:.0f}s")
        m2.metric("Segments", len(all_segments))
        m3.metric("Languages", ", ".join(langs) if langs else "en")
        m4.metric("Speakers", len(speakers))

    st.divider()

    # Download full text
    if full_text:
        st.download_button(
            "Download Transcript (.txt)", full_text,
            file_name="transcript.txt", mime="text/plain",
        )

    if not all_segments:
        st.caption("No speaker segments found.")
        if full_text:
            st.markdown(full_text)
        return

    # Speaker segments
    for seg in all_segments:
        ts = _fmt_time(seg["start"])
        st.markdown(
            f'<div class="speaker-seg">'
            f'<span class="label">{seg["speaker"]} · {ts}</span>'
            f'{seg["text"]}</div>',
            unsafe_allow_html=True,
        )


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ── Entities ───────────────────────────────────────────────────────

def _render_entities(data: dict) -> None:
    entities = data.get("entities", {})
    if not entities:
        st.caption("No entities extracted.")
        return

    persons = entities.get("persons", [])
    orgs = entities.get("organizations", [])
    keywords = entities.get("keywords", [])

    if persons:
        st.markdown("**People**")
        for p in persons[:15]:
            name = p if isinstance(p, str) else p.get("name", str(p))
            st.markdown(f'<span class="entity-chip">👤 {name}</span>', unsafe_allow_html=True)

    if orgs:
        st.markdown("**Organizations**")
        for o in orgs[:15]:
            name = o if isinstance(o, str) else o.get("name", str(o))
            st.markdown(f'<span class="entity-chip">🏢 {name}</span>', unsafe_allow_html=True)

    if keywords:
        st.markdown("**Keywords**")
        for kw in keywords[:20]:
            name = kw if isinstance(kw, str) else kw.get("name", str(kw))
            st.markdown(f'<span class="entity-chip">🔑 {name}</span>', unsafe_allow_html=True)

    if not persons and not orgs and not keywords:
        st.caption("No entities extracted.")


# ── Summaries ──────────────────────────────────────────────────────

def _render_summaries(data: dict) -> None:
    summaries = data.get("summaries", {})
    chapters = data.get("chapters", [])

    tiers = [
        ("TL;DR", "tldr"),
        ("Executive Summary", "executive"),
        ("Deep Dive", "deep_dive"),
    ]

    cols = st.columns(len(tiers))
    for idx, (title, key) in enumerate(tiers):
        text = summaries.get(key, "")
        with cols[idx]:
            st.markdown(f"**{title}**")
            if text:
                st.markdown(
                    f'<div class="summary-box">{text}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    f"Download", text,
                    file_name=f"{key}.txt",
                    mime="text/plain",
                    key=f"dl_{key}",
                )
            else:
                st.caption("Not available.")

    if chapters:
        st.divider()
        st.markdown("**Chapters**")
        for ch in chapters:
            ts = ch.get("timestamp", 0)
            title = ch.get("title", "—")
            st.markdown(f"`[{_fmt_time(ts)}]` {title}")
