"""
streamlit_app.py — AI Podcast Studio  (ECE22073)

Multi-page Streamlit app for the multilingual podcast pipeline.
All ML compute runs on Colab — this app is UI + Google Drive Bridge only.

Run:  cd App && streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Ensure Pipeline/ is on path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Pipeline"))

import config
import drive_bridge as db

logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Podcast Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Dark background */
[data-testid="stAppViewContainer"] { background: #161616; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #1f1f1f; }
[data-testid="stSidebar"] * { color: #ccc !important; }

/* Sidebar nav buttons — flat, no border, left-aligned */
[data-testid="stSidebar"] .stButton button {
    background: transparent; border: none; text-align: left;
    padding: 6px 12px; border-radius: 6px; color: #aaa; font-size: 13px; width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #1a1a1a; color: #fff;
}

/* File upload dropzone */
div[data-testid="stFileUploadDropzone"] {
    background: #1a1a1a !important; border: 1px dashed #333 !important; border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #1a1a1a; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #888; }
.stTabs [aria-selected="true"] { background: #2a2a2a !important; color: #fff !important; }

/* Text inputs / areas */
.stTextArea textarea { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; color: #ddd !important; }

/* Metric cards */
.stMetric { background: #1a1a1a; border-radius: 10px; padding: 16px; border: 1px solid #2a2a2a; }

/* Cards */
.source-card {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 14px;
}
.speaker-bubble-a {
    background: #1e2a3a; border-left: 3px solid #4a9eff;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 14px;
}
.speaker-bubble-b {
    background: #1a2a1a; border-left: 3px solid #4aff9e;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 14px;
}
.entity-chip {
    display: inline-block; background: #222; border: 1px solid #333;
    border-radius: 20px; padding: 2px 10px; margin: 2px; font-size: 12px; color: #aaa;
}
.status-badge-green { background: #1e3a1e; color: #4aff9e; border-radius: 20px; padding: 4px 12px; font-size: 12px; }
.status-badge-yellow { background: #3a361e; color: #ffe94a; border-radius: 20px; padding: 4px 12px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Auto-refresh for polling (ONLY when a job is actively processing)
# ═══════════════════════════════════════════════════════════════
# When idle/done/error: interval is huge (~11 days) → effectively no refresh.
# When processing: interval = LOCAL_POLL_INTERVAL_SEC seconds.
# This eliminates the "gray screen flicker" from perpetual re-execution.
_active_state = st.session_state.get("pipeline_state", "idle")
_active_job = st.session_state.get("active_job_id")
_refresh_ms = config.LOCAL_POLL_INTERVAL_SEC * 1000
if _active_state in ("idle", "done", "error", "stalled") or not _active_job:
    _refresh_ms = 999_999_999  # effectively disable
st_autorefresh(interval=_refresh_ms, key="poll_timer")

# ═══════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "transcript": None,
    "summary_outputs": None,
    "active_job_id": None,
    "active_job_type": None,
    "pipeline_state": "idle",
    "uploaded_filename": None,
    "chat_history": [],
    "saved_notes": [],
    "drive_connected": False,
    "last_compare_result": {},
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ═══════════════════════════════════════════════════════════════
# Drive connection (one-time init via session_state)
# ═══════════════════════════════════════════════════════════════
# authenticate() opens a browser when there is no cached token — which
# blocks and gets cancelled by st_autorefresh's periodic rerun, creating
# a new tab every cycle.  To avoid this we **only** call authenticate()
# at module level when token.json already exists (so it returns instantly
# from the file cache).  If there is no token yet the user clicks the
# "Connect Drive" button in the sidebar.

if "drive_ready" not in st.session_state:
    st.session_state["drive_ready"] = True
    st.session_state["drive_connected"] = False

    token_path = Path(__file__).parent / "token.json"
    if token_path.exists():
        try:
            db.authenticate()
            db.init_drive_structure()
            st.session_state["drive_connected"] = True
        except Exception as e:
            logger.warning("Drive init failed: %s", e)

def _is_drive_ok() -> bool:
    return bool(st.session_state.get("drive_connected", False))


def _poll_job_status() -> None:
    """Update pipeline state from Colab status.json. Called each render cycle."""
    try:
        job_id = st.session_state.get("active_job_id")
        state = st.session_state.get("pipeline_state")
        if not job_id or state in ("done", "error", "idle"):
            return
        status = db.read_status(job_id)
        if status:
            stage = status.get("stage", "")
            if stage in ("done",):
                st.session_state["pipeline_state"] = "done"
            elif stage in ("error",):
                st.session_state["pipeline_state"] = "error"
    except Exception:
        pass  # Never crash on poll — Colab may not be running yet


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

def _sidebar() -> None:
    st.sidebar.markdown("<b style='font-size:16px'>ECE22073</b>", unsafe_allow_html=True)
    st.sidebar.markdown("<span style='font-size:12px;color:#888'>AI Audio Pipeline</span>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border-color:#1f1f1f;margin:12px 0'>", unsafe_allow_html=True)

    current = st.session_state.get("_current_page", "Upload")

    # ── Section: SOURCES ──
    st.sidebar.markdown("<span style='font-size:10px;text-transform:uppercase;color:#666;letter-spacing:1px'>Sources</span>", unsafe_allow_html=True)
    nav_items = {"Upload": "📤 Upload & Transcribe"}
    for label, display in nav_items.items():
        bg = "#1e3a5f" if current == label else ""
        color = "#fff" if current == label else ""
        st.sidebar.markdown(
            f'<div style="background:{bg};border-radius:6px;padding:2px 0">',
            unsafe_allow_html=True)
        if st.sidebar.button(display, key=f"nav_{label}", use_container_width=True):
            st.session_state["_current_page"] = label
            st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # ── Section: ANALYZE ──
    st.sidebar.markdown("<span style='font-size:10px;text-transform:uppercase;color:#666;letter-spacing:1px'>Analyze</span>", unsafe_allow_html=True)
    for label in ["Notebook", "Summaries", "Accuracy Check"]:
        icons = {"Notebook": "📝", "Summaries": "📊", "Accuracy Check": "🔬"}
        display = f"{icons.get(label, '')} {label.replace('Accuracy Check', 'Accuracy')}" if label == "Accuracy Check" else f"{icons.get(label, '')} {label}"
        bg = "#1e3a5f" if current == label else ""
        if st.sidebar.button(display, key=f"nav_{label}", use_container_width=True):
            st.session_state["_current_page"] = label
            st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # ── Section: CREATE ──
    st.sidebar.markdown("<span style='font-size:10px;text-transform:uppercase;color:#666;letter-spacing:1px'>Create</span>", unsafe_allow_html=True)
    bg = "#1e3a5f" if current == "Podcast" else ""
    if st.sidebar.button("🎧 Podcast", key="nav_Podcast", use_container_width=True):
        st.session_state["_current_page"] = "Podcast"
        st.rerun()

    st.sidebar.markdown("<hr style='border-color:#1f1f1f;margin:12px 0'>", unsafe_allow_html=True)

    # ── Drive + Pipeline status at bottom ──
    drive_ok = _is_drive_ok()
    color = "🟢" if drive_ok else "🔴"
    label = "Connected" if drive_ok else "Not connected"
    st.sidebar.markdown(f"{color} **Drive:** {label}")

    if not drive_ok:
        creds_ok = (Path(__file__).parent / "credentials.json").is_file() or Path("credentials.json").is_file()
        if creds_ok:
            st.sidebar.info("`credentials.json` found — click below to connect.")
        else:
            st.sidebar.warning(
                "Place `credentials.json` in `App/` and restart.\n\n"
                "Follow: Google Cloud Console → APIs & Services → Credentials → "
                "Create OAuth Client ID (Desktop)"
            )
        if st.sidebar.button("🔗 Connect Google Drive"):
            try:
                with st.spinner("Opening Google OAuth in your browser..."):
                    db.authenticate()
                    db.init_drive_structure()
                st.session_state["drive_connected"] = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")

    job_id = st.session_state.get("active_job_id")
    state = st.session_state.get("pipeline_state", "idle")
    state_icons = {
        "idle": "⚪", "uploading": "🔄", "processing": "🟡",
        "done": "🟢", "error": "🔴", "stalled": "🟠",
    }
    st.sidebar.markdown(f"{state_icons.get(state, '⚪')} **Pipeline:** {state}")

    if job_id:
        st.sidebar.markdown(f"**Job:** `{job_id}`")
    fname = st.session_state.get("uploaded_filename")
    if fname:
        st.sidebar.markdown(f"**File:** `{fname}`")

    if job_id and state not in ("done", "error", "idle", "stalled"):
        try:
            status = db.read_status(job_id)
            if status:
                stage_label = status.get("stage", "...").replace("_", " ").title()
                pct = status.get("progress_pct", 0)
                st.sidebar.progress(pct, text=f"{stage_label}: {int(pct * 100)}%")
                updated = status.get("updated_at", "")
                if updated:
                    last_ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - last_ts).total_seconds()
                    if age > config.STALL_TIMEOUT_SEC:
                        st.session_state["pipeline_state"] = "stalled"
                        st.sidebar.warning("⚠️ Colab may have disconnected.")
            else:
                st.sidebar.caption("⏳ Awaiting Colab... (notebook not started yet)")
        except Exception:
            st.sidebar.caption("⏳ Polling Drive...")

    if state == "done":
        st.sidebar.success("✅ Complete!")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _load_results(job_id: str) -> dict[str, Any]:
    folder = f"{config.DRIVE_OUTPUT}/{job_id}"
    results: dict[str, Any] = {}
    try:
        for f in db.list_files(folder):
            if f["name"] == "transcript.json":
                results["transcript"] = db.read_json(f["id"])
            elif f["name"] == "summary_outputs.json":
                results["summary_outputs"] = db.read_json(f["id"])
    except Exception as e:
        logger.warning("Failed to load results for %s: %s", job_id, e)
    return results


# ═══════════════════════════════════════════════════════════════
# PAGE 1: Upload & Transcribe
# ═══════════════════════════════════════════════════════════════

def _page_upload() -> None:
    st.markdown("## Sources")
    st.caption("Add audio files to transcribe and analyze.")

    uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"], key="wav_uploader")

    _cur_state = st.session_state.get("pipeline_state", "idle")
    if uploaded and _cur_state in ("idle", "done", "error") and st.button("Transcribe", type="primary"):
        if not st.session_state.get("drive_connected"):
            st.error("Drive not connected. Check credentials.json in App/")
            return

        job_id = db.generate_job_id()
        st.session_state["active_job_id"] = job_id
        st.session_state["active_job_type"] = "asr"
        st.session_state["pipeline_state"] = "uploading"
        st.session_state["uploaded_filename"] = uploaded.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            db.upload_file(tmp_path, config.DRIVE_INPUT, filename=f"{job_id}.wav")
            os.unlink(tmp_path)
            st.session_state["pipeline_state"] = "processing"
            st.success(f"Job `{job_id}` submitted. Processing on Colab...")
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")
            st.session_state["pipeline_state"] = "error"

    # ── Source card after upload ───────────────────────────────────
    job_id = st.session_state.get("active_job_id")
    fname = st.session_state.get("uploaded_filename", "")
    state = st.session_state.get("pipeline_state", "idle")
    if job_id and fname:
        state_badge = {"processing": "status-badge-yellow", "done": "status-badge-green"}.get(state, "status-badge-yellow")
        st.markdown(f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;
        padding:16px;margin-top:16px;display:flex;align-items:center;gap:16px;">
          <div style="font-size:32px">🎙️</div>
          <div>
            <div style="color:#fff;font-weight:500;font-size:14px">{fname}</div>
            <div style="color:#666;font-size:12px;margin-top:4px">Job: {job_id}</div>
          </div>
          <div style="margin-left:auto">
            <span class="{state_badge}">{state}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        if state == "processing":
            status = None
            try:
                status = db.read_status(job_id)
            except Exception:
                pass
            pct = status.get("progress_pct", 0) if status else 0.5
            st.progress(pct)

    if job_id:
        with st.expander("Raw status", expanded=False):
            try:
                status = db.read_status(job_id)
                if status:
                    st.json(status)
                else:
                    st.caption("⏳ Awaiting Colab...")
            except Exception as e:
                st.caption(f"⏳ Awaiting Colab... (error: {e})")

    state = st.session_state.get("pipeline_state")
    if state == "done" and st.session_state.get("active_job_id"):
        results = _load_results(st.session_state["active_job_id"])
        transcript = results.get("transcript", {})

        if transcript:
            st.divider()
            st.markdown("**Transcript**")
            chunks = transcript.get("chunks", [])
            langs = transcript.get("languages_detected", [])
            total_dur = transcript.get("total_duration_sec", 0)

            all_segments = []
            for chunk in chunks:
                segs = chunk.get("segments", [])
                if segs:
                    for seg in segs:
                        all_segments.append({
                            "speaker": seg.get("speaker", "Speaker A"),
                            "text": seg.get("text", ""),
                            "start": seg.get("start", chunk.get("start_time_sec", 0)),
                        })
                elif chunk.get("full_text", "").strip():
                    all_segments.append({
                        "speaker": "Speaker A",
                        "text": chunk.get("full_text", ""),
                        "start": chunk.get("start_time_sec", 0),
                    })

            all_speakers = set(s["speaker"] for s in all_segments) or {"Speaker A"}

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Segments", len(all_segments))
            m2.metric("Duration", f"{total_dur:.0f}s")
            m3.metric("Languages", ", ".join(langs) if langs else "en")
            m4.metric("Speakers", len(all_speakers))

            for seg in all_segments:
                if seg["text"].strip():
                    css_class = "speaker-bubble-a" if "A" in seg["speaker"] else "speaker-bubble-b"
                    ts = f"{seg['start']:.1f}s"
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<span style="opacity:0.5;font-size:11px">{seg["speaker"]} · {ts}</span><br>'
                        f'{seg["text"]}</div>',
                        unsafe_allow_html=True,
                    )

        summary_data = results.get("summary_outputs", {})
        if summary_data:
            st.divider()
            st.markdown("**Summary**")
            summaries = summary_data.get("summaries", {})
            for level in ["tldr", "executive", "deep_dive"]:
                if level in summaries:
                    with st.expander(level.replace("_", " ").title()):
                        st.write(summaries[level])


# ═══════════════════════════════════════════════════════════════
# PAGE 2: Notebook Workspace
# ═══════════════════════════════════════════════════════════════

def _page_notebook() -> None:
    st.markdown("## Notebook Workspace")

    state = st.session_state.get("pipeline_state")
    job_id = st.session_state.get("active_job_id")

    results = {}
    transcript = {}
    summary_data = {}
    all_segments = []
    entities = {}

    if state == "done" and job_id:
        results = _load_results(job_id)
        transcript = results.get("transcript", {})
        summary_data = results.get("summary_outputs", {})
        for chunk in transcript.get("chunks", []):
            for seg in chunk.get("segments", []):
                if seg.get("text", "").strip():
                    all_segments.append({
                        "speaker": seg.get("speaker", "Speaker A"),
                        "text": seg["text"].strip(),
                        "start": seg.get("start", chunk.get("start_time_sec", 0)),
                    })
        entities = summary_data.get("entities", {}) if summary_data else {}

    col_sources, col_chat, col_studio = st.columns([1, 2, 1], gap="medium")

    # ── Left: Sources ──────────────────────────────────────────────
    with col_sources:
        st.markdown("**Sources**")
        if all_segments:
            for seg in all_segments[:30]:
                border_color = "#4a9eff" if "A" in seg["speaker"] else "#4aff9e"
                st.markdown(
                    f'<div style="background:#1a1a1a;border-left:3px solid {border_color};'
                    f'border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:13px">'
                    f'<strong>{seg["speaker"]}</strong> '
                    f'<span style="opacity:0.4">[{seg["start"]:.0f}s]</span><br>'
                    f'<span>{seg["text"][:60]}{"…" if len(seg["text"])>60 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="source-card" style="color:#666">'
                'No sources yet. Upload an audio file to get started.</div>',
                unsafe_allow_html=True)

        # Add a Source expander
        with st.expander("Add a source", expanded=False):
            tab_url, tab_pdf = st.tabs(["URL", "PDF"])
            with tab_url:
                url = st.text_input("Web page URL", placeholder="https://...")
                if st.button("Fetch URL") and url.strip():
                    try:
                        import requests
                    except ImportError:
                        st.error("Run: pip install requests html2text")
                    else:
                        with st.spinner("Fetching..."):
                            try:
                                r = requests.get(url.strip(), timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                                r.raise_for_status()
                                try:
                                    import html2text
                                    h = html2text.HTML2Text()
                                    h.ignore_links = True; h.ignore_images = True; h.body_width = 0
                                    md = h.handle(r.text)[:8000]
                                except ImportError:
                                    import re as _re
                                    raw = _re.sub(r"<[^>]+>", " ", r.text)
                                    md = _re.sub(r"\s+", " ", raw).strip()[:8000]
                                st.session_state.setdefault("extra_sources", []).append({
                                    "type": "url", "title": url.strip()[:60], "content": md,
                                })
                                st.success("Fetched"); st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
            with tab_pdf:
                pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
                if pdf_file and st.button("Extract PDF"):
                    try:
                        import fitz
                    except ImportError:
                        st.error("Run: pip install pymupdf")
                    else:
                        with st.spinner("Extracting..."):
                            try:
                                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                                text = "\n".join(page.get_text() for page in doc)[:8000]
                                doc.close()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                            else:
                                st.session_state.setdefault("extra_sources", []).append({
                                    "type": "pdf", "title": pdf_file.name[:60], "content": text,
                                })
                                st.success("Extracted"); st.rerun()

        for i, src in enumerate(st.session_state.get("extra_sources", [])):
            tag = "🌐" if src["type"] == "url" else "📄"
            st.markdown(
                f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;'
                f'padding:6px 10px;margin:4px 0;font-size:12px">{tag} '
                f'<strong>{src["title"]}</strong></div>',
                unsafe_allow_html=True,
            )

    # ── Center: Chat ───────────────────────────────────────────────
    with col_chat:
        st.markdown("**Chat**")
        has_history = bool(st.session_state.get("chat_history", []))
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if not has_history and transcript:
            st.markdown("<span style='font-size:12px;color:#888'>Suggested questions:</span>", unsafe_allow_html=True)
            suggestions = ["Summarize this transcript", "Who are the speakers?", "What are the key topics?"]
            for s in suggestions:
                if st.button(s, key=f"sugg_{s[:20]}"):
                    st.session_state["chat_history"] = [{"role": "user", "content": s}]
                    st.rerun()

        query = st.chat_input("Ask about the transcript...")
        if query:
            st.session_state.setdefault("chat_history", []).append({"role": "user", "content": query})
            if transcript:
                try:
                    import sys as _sys
                    import json as _json
                    from pathlib import Path
                    _sys.path.insert(0, str(Path(__file__).parent.parent / "Pipeline"))
                    from summary_generator import _call_llm_sync, _QA_SYSTEM_PROMPT, query_transcript as _qt

                    extra_ctx = ""
                    for src in st.session_state.get("extra_sources", []):
                        extra_ctx += f"\n[Additional source: {src['title']}]\n{src['content']}\n"

                    if extra_ctx:
                        sys_prompt = _QA_SYSTEM_PROMPT + f"\n\nReference sources:\n{extra_ctx}"
                        payload = _json.dumps({
                            "question": query, "full_text": transcript.get("full_text", ""),
                            "persons": transcript.get("all_persons", []),
                            "orgs": transcript.get("all_organizations", []),
                            "keywords": transcript.get("all_keywords", []),
                        }, ensure_ascii=False)
                        answer = _call_llm_sync(sys_prompt, payload)
                    else:
                        answer = _qt(query, transcript)
                except Exception as e:
                    answer = f"Q&A not available: {e}"
            else:
                answer = "Upload a file and complete transcription on Colab first to enable Q&A."
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

    # ── Right: Studio ──────────────────────────────────────────────
    with col_studio:
        st.markdown("**Studio**")
        for label, icon in [("Podcast", "🎧"), ("Summaries", "📊"), ("Accuracy Check", "🔬")]:
            display = f"{icon} {label.replace('Accuracy Check', 'Accuracy')}"
            if st.button(display, key=f"studio_{label}", use_container_width=True):
                st.session_state["_current_page"] = label
                st.rerun()

        if entities:
            st.divider()
            st.markdown("<span style='font-size:11px;color:#888'>Entities</span>", unsafe_allow_html=True)
            for person in entities.get("persons", [])[:5]:
                name = person if isinstance(person, str) else person.get("name", "")
                st.markdown(f'<span class="entity-chip">👤 {name}</span>', unsafe_allow_html=True)
            for kw in entities.get("keywords", [])[:10]:
                name = kw if isinstance(kw, str) else kw.get("name", "")
                st.markdown(f'<span class="entity-chip">🔑 {name}</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown("<span style='font-size:11px;color:#888'>Notes</span>", unsafe_allow_html=True)
        new_note = st.text_area("", key="note_input", height=60, placeholder="Write a note...")
        if st.button("Save Note") and new_note.strip():
            st.session_state.setdefault("saved_notes", []).append({
                "text": new_note.strip(),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            st.rerun()
        for i, note in enumerate(st.session_state.get("saved_notes", [])):
            st.markdown(f"**{note['timestamp']}** {note['text'][:80]}{'…' if len(note['text'])>80 else ''}")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Summaries
# ═══════════════════════════════════════════════════════════════

def _page_summaries() -> None:
    st.markdown("## Summaries")
    state = st.session_state.get("pipeline_state")
    job_id = st.session_state.get("active_job_id")

    if state != "done" or not job_id:
        st.info("Complete an upload on Upload first.")
        return

    results = _load_results(job_id)
    sd = results.get("summary_outputs", {}) or {}
    if not sd:
        st.warning("No summary data.")
        return

    summaries = sd.get("summaries", {})
    tier_info = [("TL;DR", "tldr"), ("Executive Summary", "executive"), ("Deep Dive", "deep_dive")]
    cols = st.columns(3)
    for idx, (title, key) in enumerate(tier_info):
        with cols[idx]:
            st.markdown(f"**{title}**")
            text = summaries.get(key, "N/A")
            st.markdown(f'<div class="source-card">{text}</div>', unsafe_allow_html=True)
            st.download_button(f"Download {title}", text, file_name=f"{key}.txt", mime="text/plain")

    chapters = sd.get("chapters", [])
    if chapters:
        st.divider()
        st.markdown("**Chapters**")
        for ch in chapters:
            ts = ch.get("timestamp", 0)
            title = ch.get("title", "Chapter")
            st.markdown(f"`[{int(ts//60):02d}:{int(ts%60):02d}]` {title}")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: Podcast Studio
# ═══════════════════════════════════════════════════════════════

def _page_podcast() -> None:
    st.markdown("## Podcast Studio")

    if not st.session_state.get("drive_connected"):
        st.warning("Drive connection required.")
        return

    state = st.session_state.get("pipeline_state")
    job_id = st.session_state.get("active_job_id")

    tab1, tab2, tab3, tab4 = st.tabs(["Source", "Episode", "Speakers", "Generate"])

    with tab1:
        src_opt = st.radio(
            "Source", ["Full transcript", "TL;DR", "Executive Summary", "Deep Dive", "Custom text"],
            horizontal=True, key="pod_source",
        )
        source_text = ""
        if src_opt == "Custom text":
            source_text = st.text_area("Paste text", height=200)
        elif state == "done" and job_id:
            results = _load_results(job_id)
            if src_opt == "Full transcript":
                t_data = results.get("transcript", {})
                source_text = t_data.get("full_text", t_data.get("raw_full_text", ""))
            else:
                tier_map = {"TL;DR": "tldr", "Executive Summary": "executive", "Deep Dive": "deep_dive"}
                sd = results.get("summary_outputs", {}) or {}
                source_text = sd.get("summaries", {}).get(tier_map.get(src_opt, ""), "")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            tone = st.selectbox("Tone", ["casual", "academic", "debate", "interview"])
        with c2:
            length = st.selectbox("Length", ["short", "medium", "long"],
                                  format_func=lambda x: {"short": "~3m", "medium": "~7m", "long": "~15m"}[x])

    with tab3:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Speaker A**")
            a_name = st.text_input("Name", "Alex", key="a_name")
            a_desc = st.text_input("Description", "Curious interviewer", key="a_desc")
            a_tts = st.selectbox("TTS", ["kokoro", "dia", "bark", "xtts_v2", "f5_tts"], key="a_tts")
            if a_tts == "kokoro":
                a_voice: str | None = st.selectbox("Voice", ["af_heart", "af_nicole", "af_bella", "am_michael", "am_adam"], key="a_voice")
            else:
                vs = st.text_input("Voice preset", key="a_voice_v")
                a_voice = vs if vs.strip() else None
        with sc2:
            st.markdown("**Speaker B**")
            b_name = st.text_input("Name", "Sam", key="b_name")
            b_desc = st.text_input("Description", "Domain expert", key="b_desc")
            b_tts = st.selectbox("TTS", ["kokoro", "dia", "bark", "xtts_v2", "f5_tts"], key="b_tts", index=0)
            if b_tts == "kokoro":
                b_voice: str | None = st.selectbox("Voice", ["am_michael", "am_adam", "af_heart", "af_nicole"], key="b_voice")
            else:
                vs2 = st.text_input("Voice preset", key="b_voice_v")
                b_voice = vs2 if vs2.strip() else None

    with tab4:
        vram_a = {"kokoro": 2, "dia": 10, "bark": 8, "xtts_v2": 6, "f5_tts": 4}.get(a_tts, 6)
        vram_b = {"kokoro": 2, "dia": 10, "bark": 8, "xtts_v2": 6, "f5_tts": 4}.get(b_tts, 6)
        total_vram = vram_a + vram_b if a_tts != b_tts or a_tts != "dia" else vram_a
        vram_label = "🟢" if total_vram <= 14 else "🟠" if total_vram <= 16 else "🔴"
        st.info(f"{vram_label} Est. VRAM: **{total_vram:.0f} GB** (T4 = ~16 GB)")

        bc1, bc2 = st.columns(2)
        with bc1:
            preview_clicked = st.button("Script Preview", disabled=not source_text.strip())
        with bc2:
            generate_clicked = st.button("Generate Podcast", type="primary", disabled=not source_text.strip())

        if preview_clicked:
            st.info("Script preview requires Colab runtime. Source text: " + str(len(source_text.split())) + " words.")

        if generate_clicked:
            job_id = db.generate_job_id()
            pod_job = {
                "job_id": job_id,
                "source_text": source_text,
                "speaker_a": {"name": a_name, "description": a_desc, "tts_model": a_tts, "voice": a_voice},
                "speaker_b": {"name": b_name, "description": b_desc, "tts_model": b_tts, "voice": b_voice},
                "config": {"tone": tone, "length": length},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db.write_json(pod_job, config.DRIVE_INPUT_JOBS, f"{job_id}.json")
            st.session_state["active_job_id"] = job_id
            st.session_state["active_job_type"] = "podcast"
            st.session_state["pipeline_state"] = "processing"
            st.success(f"Podcast job `{job_id}` submitted!")
            st.rerun()

        with st.expander("Compare Models", expanded=False):
            compare_models = st.multiselect("Models", ["kokoro", "dia", "bark", "xtts_v2", "f5_tts"], default=["kokoro", "dia"])
            if st.button("Run Comparison") and compare_models and source_text.strip():
                st.info(f"Comparison would run sequentially on Colab ({len(compare_models)} models). Requires Colab runtime.")

    # Poll for podcast MP3
    if state == "done" and st.session_state.get("active_job_type") == "podcast":
        st.divider()
        st.markdown("**Your Podcast**")
        try:
            for f in db.list_files(config.DRIVE_OUTPUT_PODCASTS):
                if f["name"] == f"{st.session_state['active_job_id']}.mp3":
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        db.download_file(f["id"], tmp.name)
                        st.audio(tmp.name, format="audio/mp3")
                        with open(tmp.name, "rb") as mp3:
                            st.download_button("Download MP3", mp3.read(), file_name=f["name"])
                        os.unlink(tmp.name)
                    break
            else:
                st.info("Waiting for Colab...")
        except Exception as e:
            st.warning(f"Could not load podcast: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Page: Accuracy Check
# ═══════════════════════════════════════════════════════════════════════════

def _page_accuracy() -> None:
    st.markdown("## Accuracy Check")
    st.caption("Compare pipeline output against a reference transcript.")

    hyp = ""
    tr = st.session_state.get("transcript") or {}
    norm = tr.get("normalized_full_text", "") or tr.get("full_text", "")
    if norm and norm.strip():
        hyp = norm
        st.caption("Using normalized transcript from current job.")
    else:
        up_hyp = st.file_uploader("Pipeline output (.txt)", type=["txt"], key="acc_hyp")
        if up_hyp:
            hyp = up_hyp.read().decode("utf-8", errors="replace")

    up_ref = st.file_uploader("Reference / Ground Truth (.txt)", type=["txt"], key="acc_ref")
    ref = ""
    if up_ref:
        ref = up_ref.read().decode("utf-8", errors="replace")

    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Hypothesis (pipeline output)", value=hyp[:2000] + ("…" if len(hyp) > 2000 else ""),
                     height=200, key="acc_hyp_area", disabled=True)
    with col2:
        st.text_area("Reference (ground truth)", value=ref[:2000] + ("…" if len(ref) > 2000 else ""),
                     height=200, key="acc_ref_area", disabled=True)

    if st.button("Compare", disabled=not (hyp.strip() and ref.strip())):
        try:
            import jiwer
        except ImportError:
            st.error("Run: pip install jiwer rouge-score")
            return
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            st.error("Run: pip install rouge-score")
            return

        wer_val = round(jiwer.wer(ref, hyp), 4)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge = scorer.score(ref, hyp)
        r1 = round(rouge["rouge1"].fmeasure, 4)
        rl = round(rouge["rougeL"].fmeasure, 4)

        m1, m2, m3 = st.columns(3)
        m1.metric("WER", f"{wer_val:.4f}", delta=f"≤0.08" if wer_val <= 0.08 else ">0.08", delta_color="off")
        m2.metric("ROUGE-1 F1", f"{r1:.4f}")
        m3.metric("ROUGE-L F1", f"{rl:.4f}")

        import difflib
        diff = list(difflib.ndiff(ref.split(), hyp.split()))
        html_parts = []
        for token in diff:
            if token.startswith("- "):
                html_parts.append(f'<span style="color:#f85149;text-decoration:line-through">{token[2:]}</span>')
            elif token.startswith("+ "):
                html_parts.append(f'<span style="color:#4aff9e">{token[2:]}</span>')
            elif token.startswith("? "):
                pass
            else:
                html_parts.append(token[2:] if token.startswith("  ") else token)
        html = " ".join(html_parts)
        with st.expander("Word-Level Diff", expanded=True):
            st.markdown(
                f'<div style="font-family:monospace;line-height:1.8;max-height:400px;overflow-y:auto;'
                f'padding:1rem;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px">{html}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# Main routing
# ═══════════════════════════════════════════════════════════════

_PAGE_KEYS: dict[str, str] = {
    "Upload": "upload",
    "Notebook": "notebook",
    "Summaries": "summaries",
    "Podcast": "podcast",
    "Accuracy Check": "accuracy",
}
_PAGE_FUNCS: dict[str, Any] = {
    "upload": _page_upload,
    "notebook": _page_notebook,
    "summaries": _page_summaries,
    "podcast": _page_podcast,
    "accuracy": _page_accuracy,
}

_poll_job_status()
_sidebar()
current = _PAGE_KEYS.get(st.session_state.get("_current_page", ""), "upload")
_PAGE_FUNCS.get(current, _page_upload)()
