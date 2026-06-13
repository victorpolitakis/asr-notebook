"""
pages/upload.py — Audio upload, Drive submission, manual status polling.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

_pipeline = Path(__file__).parent.parent.parent / "Pipeline"
if str(_pipeline) not in sys.path:
    sys.path.insert(0, str(_pipeline))

import config
import drive_bridge as db


def _connect_drive() -> bool:
    """Connect to Drive on user request. Returns True on success."""
    try:
        db.authenticate()
        db.init_drive_structure()
        st.session_state["drive_connected"] = True
        return True
    except Exception as exc:
        st.error(f"Drive connection failed: {exc}")
        return False


def _upload_and_submit(uploaded_file) -> bool:
    """Upload file to Drive and start job. Returns True on success."""
    job_id = db.generate_job_id()
    # Preserve original extension so the Colab watcher can identify format
    suffix = Path(uploaded_file.name).suffix or ".wav"

    st.session_state["active_job_id"] = job_id
    st.session_state["uploaded_filename"] = uploaded_file.name
    st.session_state["pipeline_state"] = "uploading"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        db.upload_file(tmp_path, config.DRIVE_INPUT, filename=f"{job_id}{suffix}")
        st.session_state["pipeline_state"] = "processing"
        return True
    except Exception as exc:
        st.session_state["pipeline_state"] = "error"
        st.error(f"Upload failed: {exc}")
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _check_status() -> None:
    """Poll Drive for job status. Updates pipeline_state and loads results when done."""
    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return
    try:
        status = db.read_status(job_id)
    except Exception:
        return
    if not status:
        return

    stage = status.get("stage", "")
    if stage == "done":
        st.session_state["pipeline_state"] = "done"
        _load_results(job_id)
    elif stage == "error":
        st.session_state["pipeline_state"] = "error"


def _load_results(job_id: str) -> None:
    """Load transcript + summary from Drive into session_state."""
    try:
        for f in db.list_files(f"{config.DRIVE_OUTPUT}/{job_id}"):
            if f["name"] == "transcript.json":
                st.session_state["transcript"] = db.read_json(f["id"])
            elif f["name"] == "summary_outputs.json":
                st.session_state["summary"] = db.read_json(f["id"])
    except Exception:
        pass


# ── Render ─────────────────────────────────────────────────────────────────

def render() -> None:
    drive_ok = st.session_state.get("drive_connected", False)
    pstate = st.session_state.get("pipeline_state", "idle")
    job_id = st.session_state.get("active_job_id")
    fname = st.session_state.get("uploaded_filename")

    st.markdown("## Sources")
    st.caption("Upload an audio file to transcribe and analyze.")

    # Drive gate
    if not drive_ok:
        st.warning("Drive not connected.")
        creds = Path(__file__).parent.parent / "credentials.json"
        if not creds.exists():
            st.error("Missing `App/credentials.json`. Place your Google OAuth credentials file there.")
        else:
            st.info("`credentials.json` found — click below to authenticate.")
        if st.button("Connect Google Drive", key="pg_drive_connect"):
            if _connect_drive():
                st.rerun()
        return

    # File uploader — disabled during active processing
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        key="wav_uploader",
        disabled=pstate == "processing",
    )

    if uploaded is not None and pstate in ("idle", "done", "error"):
        if st.button("Transcribe", type="primary"):
            if _upload_and_submit(uploaded):
                st.rerun()

    # Job card
    if job_id and fname:
        _render_job_card(job_id, fname, pstate)

    # Post-completion brief
    if pstate == "done":
        _render_done_brief()


def _render_job_card(job_id: str, fname: str, pstate: str) -> None:
    color = {"processing": "#ddb83d", "done": "#3dde8f", "error": "#ff5a5a",
             "uploading": "#ddb83d"}.get(pstate, "#666")
    st.markdown(
        f'<div class="job-card">'
        f'<span style="font-size:28px">🎙️</span>'
        f'<div>'
        f'<div style="color:#e0e0f0;font-weight:500;font-size:14px">{fname}</div>'
        f'<div style="color:#707088;font-size:11px;margin-top:4px;font-family:IBM Plex Mono,monospace">Job: {job_id}</div>'
        f'</div>'
        f'<div style="margin-left:auto;display:flex;align-items:center;gap:6px;">'
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color}"></span>'
        f'<span style="color:{color};font-size:12px;font-family:IBM Plex Mono,monospace">{pstate}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if pstate == "processing":
        if st.button("Check Status", key="check_status"):
            with st.spinner("Polling Drive..."):
                _check_status()
            st.rerun()
        st.caption("Colab is processing. Click above to manually check progress.")


def _render_done_brief() -> None:
    transcript = st.session_state.get("transcript") or {}
    summary = st.session_state.get("summary") or {}

    all_segments = []
    for chunk in transcript.get("chunks", []):
        for seg in chunk.get("segments", []):
            if seg.get("text", "").strip():
                all_segments.append(seg)

    duration = transcript.get("total_duration_sec", 0)
    langs = transcript.get("languages_detected", [])
    speakers = list({s.get("speaker", "A") for s in all_segments})

    st.divider()
    st.markdown("### Job Complete")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segments", len(all_segments))
    m2.metric("Duration", f"{duration:.0f}s")
    m3.metric("Languages", ", ".join(langs) if langs else "—")
    m4.metric("Speakers", len(speakers) or "—")

    tldr = (summary.get("summaries") or {}).get("tldr", "")
    if tldr:
        st.caption(f"TL;DR — {tldr[:300]}{'…' if len(tldr) > 300 else ''}")

    if st.button("View Full Results →", use_container_width=True):
        st.session_state["_page"] = "Results"
        st.rerun()
