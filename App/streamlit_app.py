"""
ECE22073 — AI Podcast Studio
Single-file Streamlit UI. All ML compute runs on Colab via Google Drive.
Run: cd App && streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Must be the absolute first Streamlit call ──────────────────────────────
st.set_page_config(page_title="ECE22073", page_icon="🎙️", layout="wide")

# ── Add Pipeline/ and App/ to import path ─────────────────────────────────
_here = Path(__file__).parent
for _p in (_here.parent / "Pipeline", _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
import drive_bridge as db
import comparison_metrics as cm


# ══════════════════════════════════════════════════════════════════════════
# CSS  — dark theme, no `all:unset`, no `header{visibility:hidden}`
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Share+Tech+Mono&display=swap');

:root {
  --bg:          #070604;
  --bg2:         #110e09;
  --bg3:         #1c1609;
  --border:      #2e2510;
  --border-hi:   #4a3a18;
  --amber:       #e8a520;
  --amber-dim:   #7a5c0a;
  --amber-glow:  rgba(232,165,32,0.12);
  --green:       #2ee89b;
  --red:         #e84a2e;
  --text:        #d4c4a0;
  --text-dim:    #7a6d54;
  --text-faint:  #352c1e;
  --fn-head:     'Bebas Neue', 'Impact', 'Arial Narrow', sans-serif;
  --fn-body:     'Courier Prime', 'Courier New', monospace;
  --fn-mono:     'Share Tech Mono', 'Courier New', monospace;
}

/* ═══════════════════════════════════════════════════════════
   BASE
═══════════════════════════════════════════════════════════ */
html, body, [class*="css"] { font-family: var(--fn-body) !important; }

#MainMenu, footer { visibility: hidden; }
[data-testid="stStatusWidget"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }

/* ═══════════════════════════════════════════════════════════
   APP BACKGROUND — warm grid
═══════════════════════════════════════════════════════════ */
.stApp {
  background-color: var(--bg) !important;
  background-image:
    linear-gradient(rgba(232,165,32,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(232,165,32,0.028) 1px, transparent 1px) !important;
  background-size: 48px 48px !important;
}

/* ═══════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #030201 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"]::before {
  content: '';
  display: block;
  height: 2px;
  background: linear-gradient(90deg, var(--amber) 0%, transparent 80%);
  margin-bottom: 4px;
}

/* ═══════════════════════════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════════════════════════ */
h1, h2, h3 {
  font-family: var(--fn-head) !important;
  font-weight: 400 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  line-height: 1.05 !important;
}
h2 {
  font-size: 3rem !important;
  color: var(--amber) !important;
  padding-bottom: 12px !important;
  background-image: linear-gradient(90deg, var(--border-hi) 0%, transparent 55%) !important;
  background-size: 100% 1px !important;
  background-repeat: no-repeat !important;
  background-position: 0 100% !important;
  margin-bottom: 1.2rem !important;
}
h3 {
  font-size: 1.4rem !important;
  color: var(--text) !important;
  letter-spacing: 0.1em !important;
}
[data-testid="stMarkdownContainer"] p {
  font-family: var(--fn-body);
  color: var(--text-dim);
  line-height: 1.8;
}
[data-testid="stMarkdownContainer"] strong { color: var(--text); }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
  font-family: var(--fn-mono) !important;
  font-size: 10px !important;
  color: var(--text-faint) !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ═══════════════════════════════════════════════════════════
   METRICS
═══════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: 2px solid var(--amber-dim);
  border-radius: 2px;
  padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
  font-family: var(--fn-mono) !important;
  color: var(--amber-dim) !important;
  font-size: 9px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.14em !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--fn-head) !important;
  color: var(--amber) !important;
  font-size: 2.2rem !important;
  letter-spacing: 0.05em !important;
}

/* ═══════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════ */
[data-testid="stBaseButton-primary"] {
  background: var(--amber) !important;
  color: #070604 !important;
  border: none !important;
  border-radius: 2px !important;
  font-family: var(--fn-mono) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  transition: box-shadow 0.2s ease, opacity 0.15s ease !important;
}
[data-testid="stBaseButton-primary"]:hover {
  box-shadow: 0 0 20px rgba(232,165,32,0.55) !important;
  opacity: 0.9 !important;
}
[data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: var(--fn-mono) !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--amber-dim) !important;
  color: var(--text) !important;
}

/* ═══════════════════════════════════════════════════════════
   INPUTS / FILE UPLOADER
═══════════════════════════════════════════════════════════ */
textarea, input[type="text"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 2px !important;
  font-family: var(--fn-body) !important;
}
textarea:focus, input:focus {
  border-color: var(--amber-dim) !important;
  box-shadow: 0 0 0 1px rgba(232,165,32,0.2) !important;
  outline: none !important;
}
[data-testid="stFileUploadDropzone"] {
  background: var(--bg2) !important;
  border: 1px dashed var(--amber-dim) !important;
  border-radius: 2px !important;
  transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color: var(--amber) !important;
  box-shadow: 0 0 28px rgba(232,165,32,0.08) !important;
}

/* ═══════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
  background: var(--bg2);
  border-radius: 2px;
  padding: 3px;
  border: 1px solid var(--border);
  gap: 2px;
}
[data-baseweb="tab"] {
  color: var(--text-dim);
  font-size: 10px;
  font-family: var(--fn-mono);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  border-radius: 1px;
  padding: 5px 16px;
}
[aria-selected="true"] {
  background: rgba(232,165,32,0.1) !important;
  color: var(--amber) !important;
}

/* ═══════════════════════════════════════════════════════════
   ALERTS / SELECT / RADIO
═══════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: 2px !important;
  border-left-width: 2px !important;
  font-family: var(--fn-body);
  font-size: 13px;
  background: var(--bg2) !important;
}
[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}
[data-baseweb="select"] > div {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 2px;
  font-family: var(--fn-body);
}
[data-baseweb="radio"] label {
  font-size: 11px;
  color: var(--text-dim);
  font-family: var(--fn-mono);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ═══════════════════════════════════════════════════════════
   PROGRESS / DATAFRAME / DIVIDER
═══════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--amber-dim), var(--amber)) !important;
}
[data-testid="stDataFrame"] { font-size: 12px; font-family: var(--fn-mono); }
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.4rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════
   SCROLLBARS
═══════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

/* ═══════════════════════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════════════════════ */
@keyframes fade-up {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stMainBlockContainer"] {
  animation: fade-up 0.38s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes amber-pulse {
  0%, 100% { box-shadow: 0 0 0   0   rgba(232,165,32,0); }
  50%       { box-shadow: 0 0 24px 2px rgba(232,165,32,0.22); }
}

/* ═══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════════════════════ */
.seg {
  background: var(--bg2);
  border-left: 2px solid var(--border-hi);
  border-radius: 0 2px 2px 0;
  padding: 10px 14px;
  margin-bottom: 4px;
  font-size: 13px;
  line-height: 1.8;
  color: var(--text-dim);
  font-family: var(--fn-body);
}
.seg-a { border-left-color: var(--amber-dim); }
.seg-b { border-left-color: #2a5a3a; }
.seg .lbl {
  display: block;
  font-family: var(--fn-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--amber-dim);
  margin-bottom: 5px;
}
.chip {
  display: inline-block;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 1px;
  padding: 2px 8px;
  margin: 2px 3px 2px 0;
  font-size: 10px;
  color: var(--text-faint);
  font-family: var(--fn-mono);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.jcard {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 10px;
}
.jcard.is-processing {
  border-color: var(--amber-dim);
  animation: amber-pulse 2.4s ease-in-out infinite;
}
.diff-view {
  font-family: var(--fn-mono);
  font-size: 12px;
  line-height: 2;
  max-height: 420px;
  overflow-y: auto;
  padding: 14px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 2px;
  white-space: pre-wrap;
  word-break: break-word;
}
/* Hide the "add more files" (+) button — single upload only */
[data-testid="stFileUploaderFile"] ~ button,
[data-testid="stFileUploaderFile"] + button,
[data-testid="stFileUploader"] li ~ button,
[data-testid="stFileUploader"] ul ~ button,
[data-testid="stFileUploader"] [data-testid="stBaseButton-minimal"],
[data-testid="stFileUploader"] button[aria-label="Add file"],
[data-testid="stFileUploader"] button[aria-label="add file"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════
   CHAT
═══════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border);
  border-radius: 2px;
  margin-bottom: 6px;
}
[data-testid="stChatMessage"] p { color: var(--text) !important; font-family: var(--fn-body) !important; font-size: 13px; }
[data-testid="stChatMessage"][data-testid*="user"] { border-left: 2px solid var(--amber-dim); }
[data-testid="stChatMessage"][data-testid*="assistant"] { border-left: 2px solid #2a5a3a; }
[data-testid="stChatInput"] { background: var(--bg2) !important; border-top: 1px solid var(--border) !important; }
[data-testid="stChatInput"] textarea {
  background: var(--bg) !important;
  border: 1px solid var(--border-hi) !important;
  color: var(--text) !important;
  font-family: var(--fn-body) !important;
  font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Session state — all keys declared once here
# ══════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "uploaded_filename": None,
    "active_job_id":     None,
    "pipeline_state":    "idle",   # idle | uploading | processing | done | error
    "pipeline_error":    None,
    "transcript":        None,
    "summary":           None,
    "acc_result":        None,
    "acc_gt":            None,
    "history_items":     None,
    "drive_connected":   False,
    "_page":             "Upload",
    "poll_miss_count":   0,
    "chat_history":      [],
    "model_transcripts": {},
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_job() -> None:
    for _k in ("active_job_id", "pipeline_state", "pipeline_error",
               "uploaded_filename", "transcript", "summary", "poll_miss_count", "model_transcripts"):
        st.session_state[_k] = _DEFAULTS[_k]
    st.query_params.clear()


# ══════════════════════════════════════════════════════════════════════════
# Drive helpers
# ══════════════════════════════════════════════════════════════════════════

def _drive_connect_silent() -> None:
    """Connect silently if token.json is already cached. No-op if already connected."""
    if st.session_state.drive_connected:
        return
    if (_here / "token.json").exists():
        try:
            db.authenticate()
            st.session_state.drive_connected = True
        except Exception:
            pass


def _load_results(job_id: str) -> None:
    """Download transcript.json + summary_outputs.json + model-specific transcripts + accuracy from Drive into session_state."""
    try:
        st.session_state.model_transcripts = {}
        for f in db.list_files(f"{config.DRIVE_OUTPUT}/{job_id}"):
            if f["name"] == "transcript.json":
                st.session_state.transcript = db.read_json(f["id"])
            elif f["name"] == "summary_outputs.json":
                st.session_state.summary = db.read_json(f["id"])
            elif f["name"].startswith("transcript_") and f["name"].endswith(".json"):
                model_name = f["name"][11:-5]
                st.session_state.model_transcripts[model_name] = db.read_json(f["id"])
            elif f["name"] == "accuracy_result.json":
                acc_data = db.read_json(f["id"])
                st.session_state.acc_gt     = acc_data.get("gt_text", "")
                st.session_state.acc_result = acc_data.get("results")
    except Exception:
        pass


_POLL_MISS_LIMIT = 6  # 6 × 15 s = 90 s before auto-reset to idle

def _poll_job(job_id: str) -> None:
    """Read status.json from Drive. Transitions pipeline_state and loads results on done.
    Auto-resets to idle after POLL_MISS_LIMIT consecutive polls with no status found
    (handles stale job IDs left in the URL from a broken/mismatched session).
    """
    try:
        s = db.read_status(job_id)
        if not s:
            st.session_state.poll_miss_count += 1
            if st.session_state.poll_miss_count >= _POLL_MISS_LIMIT:
                _reset_job()
            return
        st.session_state.poll_miss_count = 0
        stage = s.get("stage", "")
        if stage == "done":
            st.session_state.pipeline_state = "done"
            _load_results(job_id)
        elif stage == "error":
            st.session_state.pipeline_state = "error"
            st.session_state.pipeline_error = s.get("error")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# Boot sequence (runs on every Streamlit rerun)
# ══════════════════════════════════════════════════════════════════════════

_drive_connect_silent()

# Restore job from URL after hard refresh (state lost, URL survives)
if not st.session_state.active_job_id:
    _qp_jid = st.query_params.get("job_id")
    if _qp_jid:
        st.session_state.active_job_id     = _qp_jid
        st.session_state.uploaded_filename = st.query_params.get("fname", "")
        # Default to processing so the card shows while we figure out real state.
        # The polling fragment will update to done/error on its first tick.
        st.session_state.pipeline_state = "processing"
        if st.session_state.drive_connected:
            try:
                _s = db.read_status(_qp_jid)
                if _s:
                    _stage = _s.get("stage", "")
                    if _stage == "done":
                        st.session_state.pipeline_state = "done"
                        _load_results(_qp_jid)
                    elif _stage == "error":
                        st.session_state.pipeline_state = "error"
                        st.session_state.pipeline_error = _s.get("error")
                    # else: leave as "processing" — fragment will poll
            except Exception:
                pass  # leave as "processing" — fragment will retry


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

def _dot(color: str, size: int = 6) -> str:
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;'
            f'border-radius:50%;background:{color};margin-right:7px;vertical-align:middle;'
            f'box-shadow:0 0 5px {color}90"></span>')


with st.sidebar:
    # ── Brand ──
    st.markdown(
        '<div style="padding:10px 0 20px">'
        '<div style="font-family:\'Bebas Neue\',\'Impact\',monospace;font-size:26px;'
        'letter-spacing:0.18em;color:#e8a520;line-height:1">ECE22073</div>'
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:9px;color:#352c1e;'
        'letter-spacing:0.22em;text-transform:uppercase;margin-top:3px">AI Audio Pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Navigation ──
    # type="primary" on the active page gives a clear visual active state without JS
    _cur_page = st.session_state._page
    _nav_items = [("Upload", "Upload"), ("Results", "Results"), ("Accuracy", "Accuracy Check"), ("History", "History")]
    for _key, _label in _nav_items:
        if st.button(
            _label, key=f"nav_{_key}",
            type="primary" if _cur_page == _key else "secondary",
            use_container_width=True,
        ):
            st.session_state._page = _key
            st.rerun()

    st.markdown('<hr style="margin:14px 0">', unsafe_allow_html=True)

    # ── Drive connection ──
    _drive_ok = st.session_state.drive_connected
    st.markdown(
        f'{_dot("#2ee89b" if _drive_ok else "#e84a2e")}'
        f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#7a6d54;'
        f'text-transform:uppercase;letter-spacing:0.08em">'
        f'{"Drive connected" if _drive_ok else "Drive not connected"}</span>',
        unsafe_allow_html=True,
    )

    if not _drive_ok:
        _has_creds = (_here / "credentials.json").exists()
        if not _has_creds:
            st.caption("Missing `App/credentials.json`")
        if st.button("Connect Google Drive", key="sb_connect_drive", disabled=not _has_creds):
            try:
                with st.spinner("Opening OAuth…"):
                    db.authenticate()
                    db.init_drive_structure()
                st.session_state.drive_connected = True
                st.rerun()
            except Exception as _e:
                st.error(str(_e))

    # ── Pipeline state ──
    _ps = st.session_state.pipeline_state
    _ps_color = {
        "idle": "#352c1e", "uploading": "#e8a520", "processing": "#e8a520",
        "done": "#2ee89b", "error": "#e84a2e",
    }.get(_ps, "#352c1e")

    st.markdown(
        f'<div style="margin-top:10px">'
        f'{_dot(_ps_color)}'
        f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#7a6d54;'
        f'text-transform:uppercase;letter-spacing:0.08em">Pipeline: {_ps}</span></div>',
        unsafe_allow_html=True,
    )

    _fn = st.session_state.uploaded_filename
    if _fn:
        st.caption(f"↳ {_fn[:28]}{'…' if len(_fn) > 28 else ''}")

    # Progress bar during processing
    if st.session_state.pipeline_state == "processing" and st.session_state.active_job_id:
        try:
            _status_live = db.read_status(st.session_state.active_job_id)
            if _status_live:
                _pct  = _status_live.get("progress_pct", 0)
                _stg  = _status_live.get("stage", "").replace("_", " ")
                st.progress(_pct, text=f"{_stg} · {int(_pct * 100)}%")
        except Exception:
            pass

    # Reset/cancel button — always visible once a job is active
    if _ps != "idle":
        st.write("")
        _btn_lbl = "New Job" if _ps in ("done", "error") else "✕ Cancel"
        if st.button(_btn_lbl, key="sb_new_job", use_container_width=True):
            _reset_job()
            st.session_state._page = "Upload"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Shared utility
# ══════════════════════════════════════════════════════════════════════════

def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _fv(v: Any) -> str:
    """Format a metric value, returning '—' for None."""
    return f"{v}" if v is not None else "—"


def _segments_from_transcript(t: dict) -> list[dict]:
    segs = []
    for chunk in t.get("chunks", []):
        chunk_segs = chunk.get("segments", [])
        if chunk_segs:
            for seg in chunk_segs:
                if seg.get("text", "").strip():
                    segs.append({
                        "speaker": seg.get("speaker", "Speaker A"),
                        "text": seg["text"].strip(),
                        "start": seg.get("start", chunk.get("start_time_sec", 0)),
                    })
        else:
            # Fallback for older transcripts without per-segment data
            text = chunk.get("full_text", "").strip()
            if text:
                speakers = chunk.get("speakers_detected", [])
                segs.append({
                    "speaker": speakers[0] if speakers else "Speaker A",
                    "text": text,
                    "start": chunk.get("start_time_sec", 0),
                })
    return segs


# ══════════════════════════════════════════════════════════════════════════
# Page: Upload
# ══════════════════════════════════════════════════════════════════════════

def _page_upload() -> None:
    st.markdown("## Upload")

    if not st.session_state.drive_connected:
        st.warning("Drive not connected — use the sidebar button to authenticate first.")
        return

    _ps   = st.session_state.pipeline_state
    jid   = st.session_state.active_job_id
    fname = st.session_state.uploaded_filename

    # ── Active job: hide uploader, show card ──
    if jid:
        _render_job_card(jid, fname or "—", _ps)
        if _ps == "processing":
            st.markdown(
                f'<p style="font-family:var(--fn-mono);font-size:10px;color:var(--text-faint);'
                f'letter-spacing:0.06em;text-transform:uppercase">'
                f'Uploaded. Colab is transcribing — auto-updates every '
                f'<span style="text-transform:none">{config.LOCAL_POLL_INTERVAL_SEC} s</span>.</p>',
                unsafe_allow_html=True,
            )
        elif _ps == "done":
            _render_upload_done_summary()
        elif _ps == "error":
            _err = st.session_state.pipeline_error
            if not _err:
                _err_msg = "No error details found. The Colab watcher may not be running — start Cell 6 in the notebook."
            elif "not running" in _err.lower() or "Pipeline returned False" in _err:
                _err_msg = f"Pipeline failed: {_err}\n\nMake sure the Colab watcher (Cell 6) is active."
            else:
                _err_msg = f"Pipeline error: {_err}"
            st.error(_err_msg)
            if st.button("↑ New Upload", key="btn_new_upload_err"):
                _reset_job()
                st.rerun()
        return

    # ── Idle: show uploader ──
    st.caption("Select ASR models to compare and then choose an audio file.")
    selected_models = st.multiselect(
        "ASR Models to compare",
        ["Whisper Turbo", "Whisper Large v3", "Nvidia Canary", "Nvidia Parakeet", "Qwen ASR", "Nemotron"],
        default=["Whisper Turbo"],
        key="selected_models_picker"
    )
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )
    if uploaded:
        with st.spinner(f"Uploading {uploaded.name} to Drive…"):
            _upload_and_submit(uploaded, selected_models)
        st.rerun()


def _upload_and_submit(f: Any, selected_models: list[str]) -> None:
    jid = db.generate_job_id()
    ext = Path(f.name).suffix or ".wav"
    st.session_state.active_job_id     = jid
    st.session_state.uploaded_filename = f.name
    st.session_state.pipeline_state    = "uploading"
    st.query_params["job_id"] = jid
    st.query_params["fname"]  = f.name

    name_map = {
        "Whisper Turbo": "whisper-turbo",
        "Whisper Large v3": "whisper-large-v3",
        "Nvidia Canary": "canary",
        "Nvidia Parakeet": "parakeet",
        "Qwen ASR": "qwen",
        "Nemotron": "nemotron"
    }
    model_keys = [name_map[m] for m in selected_models if m in name_map]
    if not model_keys:
        model_keys = ["whisper-turbo"]

    # Clear any stale audio files left in the input folder from previous broken
    # runs so the Colab watcher doesn't pick them up instead of the new file.
    # Only delete files older than 20 minutes to avoid deleting currently transcribing files.
    _AUDIO_EXTS = {".wav", ".mp3", ".m4a"}
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        for _stale in db.list_files(config.DRIVE_INPUT):
            if Path(_stale.get("name", "")).suffix.lower() in _AUDIO_EXTS:
                ct_str = _stale.get("createdTime")
                if ct_str:
                    try:
                        ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                        age_minutes = (now - ct).total_seconds() / 60.0
                        if age_minutes > 20:
                            db.delete_file(_stale["id"])
                    except Exception:
                        pass
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(f.read())
        tmp_path = tmp.name
    try:
        db.upload_file(tmp_path, config.DRIVE_INPUT, filename=f"{jid}{ext}")
        st.session_state.pipeline_state = "processing"
        try:
            db.write_json(
                {
                    "filename": f.name,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "selected_models": model_keys
                },
                f"{config.DRIVE_OUTPUT}/{jid}",
                "meta.json",
            )
        except Exception:
            pass
    except Exception as exc:
        st.session_state.pipeline_state = "error"
        st.error(f"Upload failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _render_job_card(jid: str, fname: str, ps: str) -> None:
    color  = {"processing": "#e8a520", "done": "#2ee89b", "error": "#e84a2e",
              "uploading": "#e8a520"}.get(ps, "#352c1e")
    symbol = {"processing": "▶", "done": "◼", "error": "◻", "uploading": "▶"}.get(ps, "◈")
    extra  = "is-processing" if ps in ("processing", "uploading") else ""
    st.markdown(
        f'<div class="jcard {extra}">'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:20px;'
        f'color:#7a5c0a;min-width:22px;text-align:center">{symbol}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="color:#d4c4a0;font-family:\'Courier Prime\',monospace;font-size:14px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{fname}</div>'
        f'<div style="color:#352c1e;font-size:9px;font-family:\'Share Tech Mono\',monospace;'
        f'margin-top:4px;letter-spacing:0.12em;text-transform:uppercase">JOB / {jid}</div>'
        f'</div>'
        f'<div style="color:{color};font-size:9px;font-family:\'Share Tech Mono\',monospace;'
        f'letter-spacing:0.18em;text-transform:uppercase;white-space:nowrap">{ps}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )



def _render_upload_done_summary() -> None:
    t = st.session_state.transcript or {}
    s = st.session_state.summary    or {}
    segs  = _segments_from_transcript(t)
    dur   = t.get("total_duration_sec", 0)
    langs = t.get("languages_detected", [])
    spks  = {seg["speaker"] for seg in segs}

    st.divider()
    st.markdown("### Done")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments",  len(segs))
    c2.metric("Duration",  _fmt_time(dur))
    c3.metric("Languages", ", ".join(langs) if langs else "en")
    c4.metric("Speakers",  len(spks) or "—")

    tldr = ((s.get("summaries") or {}).get("tldr") or "")
    if tldr:
        st.info(f"**TL;DR** — {tldr[:300]}{'…' if len(tldr) > 300 else ''}")

    c_r, c_n = st.columns(2)
    if c_r.button("View Full Results →", type="primary"):
        st.session_state._page = "Results"
        st.rerun()
    if c_n.button("↑ New Upload"):
        _reset_job()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Chat helper
# ══════════════════════════════════════════════════════════════════════════

def _results_chat(s: dict) -> None:
    context_parts: list[str] = []

    deep_dive = _flatten_summary((s.get("summaries") or {}).get("deep_dive"))
    if deep_dive:
        context_parts.append(f"SUMMARY:\n{deep_dive}")

    ents = s.get("entities") or {}
    ent_lines: list[str] = []
    for k, label in [("persons", "People"), ("organizations", "Organizations"), ("keywords", "Keywords")]:
        items = ents.get(k, [])
        if items:
            names = [i if isinstance(i, str) else i.get("name", "") for i in items[:20]]
            ent_lines.append(f"{label}: {', '.join(filter(None, names))}")
    if ent_lines:
        context_parts.append("ENTITIES:\n" + "\n".join(ent_lines))

    if not context_parts:
        st.caption("No summary or entity data available — run a transcription job first.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY in your environment to enable chat.")
        return

    system_prompt = (
        "You are a concise assistant answering questions about an audio recording. "
        "Use only the context below — do not invent information not present in it.\n\n"
        + "\n\n".join(context_parts)
    )

    # Declare container BEFORE chat_input so messages always render above the input box.
    chat_container = st.container()

    user_input = st.chat_input("Ask about the transcript…")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner(""):
            try:
                from openai import OpenAI as _OAI
                client = _OAI(api_key=api_key, base_url=config.LLM_BASE_URL)
                resp = client.chat.completions.create(
                    model=os.environ.get("LLM_MODEL", config.LLM_MODEL),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.chat_history,
                    ],
                    max_tokens=800,
                )
                answer = resp.choices[0].message.content or ""
            except Exception as exc:
                answer = f"Error: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if st.session_state.chat_history:
            if st.button("Clear", key="chat_clear"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Page: Results
# ══════════════════════════════════════════════════════════════════════════

def _page_results() -> None:
    t = st.session_state.transcript or {}
    s = st.session_state.summary    or {}

    if not t and not s:
        st.info("No results yet. Upload and transcribe a file on the **Upload** page first.")
        if st.button("← Go to Upload"):
            st.session_state._page = "Upload"
            st.rerun()
        return

    st.markdown("## Results")
    tab_tr, tab_ent, tab_sum, tab_chat = st.tabs(["Transcript", "Entities", "Summaries", "Chat"])

    with tab_tr:
        model_trans = st.session_state.model_transcripts or {}
        # Always include the main transcript (whisper-turbo from run_pipeline)
        all_transcripts: dict[str, dict] = {"Whisper Turbo": t} if t else {}
        for mk, mv in model_trans.items():
            all_transcripts[_DISPLAY_MAP.get(mk, mk.upper())] = mv

        if len(all_transcripts) > 1:
            model_tabs = st.tabs(list(all_transcripts.keys()))
            for m_tab, (mname, trans) in zip(model_tabs, all_transcripts.items()):
                with m_tab:
                    _results_transcript(trans, key=mname.replace(" ", "_").lower())
        else:
            _results_transcript(t)
    with tab_ent:
        _results_entities(s)
    with tab_sum:
        _results_summaries(s)
    with tab_chat:
        _results_chat(s)


def _results_transcript(t: dict, key: str = "main") -> None:
    segs  = _segments_from_transcript(t)
    dur   = t.get("total_duration_sec", 0)
    langs = t.get("languages_detected", [])
    spks  = {s["speaker"] for s in segs}
    full  = t.get("full_text", "")

    n_speakers = len(spks) if spks else (1 if full.strip() else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",  _fmt_time(dur))
    c2.metric("Segments",  len(segs))
    c3.metric("Languages", ", ".join(langs) if langs else "en")
    c4.metric("Speakers",  n_speakers)

    if full:
        st.download_button("⬇ transcript.txt", full, file_name="transcript.txt", key=f"dl_transcript_{key}")

    st.divider()

    if not segs:
        if full:
            st.markdown(f'<div class="seg seg-a">{full}</div>', unsafe_allow_html=True)
        else:
            st.caption("No transcript content found.")
        return

    if len(spks) > 1:
        for seg in segs:
            css_cls = "seg-a" if "A" in seg["speaker"] else "seg-b"
            st.markdown(
                f'<div class="seg {css_cls}">'
                f'<span class="lbl">{seg["speaker"]} · {_fmt_time(seg["start"])}</span>'
                f'{seg["text"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        combined = "<br>".join(seg["text"] for seg in segs)
        st.markdown(f'<div class="seg seg-a">{combined}</div>', unsafe_allow_html=True)


def _results_entities(s: dict) -> None:
    ents = s.get("entities") or {}
    if not ents:
        st.caption("No entities extracted.")
        return

    for label, key, icon in [
        ("People",        "persons",       "👤"),
        ("Organizations", "organizations", "🏢"),
        ("Keywords",      "keywords",      "🔑"),
    ]:
        items = ents.get(key, [])
        if not items:
            continue
        st.markdown(f"**{label}**")
        chips = " ".join(
            f'<span class="chip">{icon} {(i if isinstance(i, str) else i.get("name", ""))}</span>'
            for i in items[:25]
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.write("")


def _flatten_summary(raw: Any) -> str:
    """Convert any summary value (str | list | dict) to a plain display string."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        return "\n".join(f"• {item}" for item in raw if item)
    if isinstance(raw, dict):
        parts: list[str] = []
        if raw.get("overview"):
            parts.append(str(raw["overview"]))
        for k, lbl in [("bullet_points", "Key Points"),
                        ("key_takeaways",  "Key Takeaways"),
                        ("action_items",   "Action Items")]:
            items = raw.get(k) or []
            if items:
                parts.append(f"\n{lbl}:")
                parts.extend(f"  • {item}" for item in items)
        return "\n".join(parts)
    return ""


def _results_summaries(s: dict) -> None:
    sums = s.get("summaries") or {}
    chs  = s.get("chapters")  or []

    text = _flatten_summary(sums.get("deep_dive"))
    if text:
        st.text_area("", text, height=400, key="sum_deep_dive", label_visibility="collapsed")
        st.download_button("Download", text, file_name="deep_dive.txt", key="dl_deep_dive")
    else:
        st.caption("Summary not available.")

    if chs:
        st.divider()
        st.markdown("**Chapters**")
        for ch in chs:
            st.markdown(f"`[{_fmt_time(ch.get('timestamp', 0))}]` {ch.get('title', '')}")


# ══════════════════════════════════════════════════════════════════════════
# Page: Accuracy Check
# ══════════════════════════════════════════════════════════════════════════

def _page_accuracy() -> None:
    st.markdown("## Accuracy Check")
    st.caption("Compare pipeline output against a reference ground-truth transcript.")
    st.divider()
    _acc_single()


# ── Single comparison ──────────────────────────────────────────────────────

_DISPLAY_MAP = {
    "whisper-turbo":    "Whisper Turbo",
    "whisper-large-v3": "Whisper Large v3",
    "canary":           "Nvidia Canary",
    "parakeet":         "Nvidia Parakeet",
    "qwen":             "Qwen ASR",
    "nemotron":         "Nemotron",
}


def _acc_single() -> None:
    # Build hypotheses: main transcript (whisper-turbo) + any extra models
    hypotheses: dict[str, str] = {}  # {display_name: text}

    t_main = st.session_state.transcript or {}
    main_text = t_main.get("normalized_full_text") or t_main.get("full_text") or ""
    if main_text.strip():
        hypotheses["Whisper Turbo"] = main_text

    for model_key, trans in (st.session_state.model_transcripts or {}).items():
        text = trans.get("normalized_full_text") or trans.get("full_text") or ""
        if text.strip():
            hypotheses[_DISPLAY_MAP.get(model_key, model_key.upper())] = text

    if not hypotheses:
        st.info("No transcript loaded. Go to **Upload** or **History** to load a job first.")
        return

    up_ref = st.file_uploader("Ground Truth (.txt)", type=["txt"], key="acc_s_ref")
    if up_ref:
        ref = up_ref.read().decode("utf-8", errors="replace")
    else:
        ref = st.session_state.acc_gt or ""

    if not ref.strip():
        st.caption(f"Available transcripts: {', '.join(hypotheses.keys())}")
        return

    if st.button("Compare all models", type="primary"):
        with st.spinner("Computing metrics…"):
            results = {
                name: cm.compute_all_metrics(hyp, ref, label=name)
                for name, hyp in hypotheses.items()
            }
            st.session_state.acc_result = results
            st.session_state.acc_gt = ref
            # Persist to Drive so it survives tab switches and history reload
            job_id = st.session_state.get("active_job_id")
            if job_id:
                try:
                    db.write_json(
                        {"gt_text": ref, "results": results},
                        f"{config.DRIVE_OUTPUT}/{job_id}",
                        "accuracy_result.json",
                    )
                except Exception:
                    pass

    raw = st.session_state.acc_result
    # Guard against stale single-result dict from old session format
    results: dict[str, Any] = raw if isinstance(raw, dict) and all(
        isinstance(v, dict) and "wer" in v for v in raw.values()
    ) else {}
    if not results:
        return

    tab_names = list(results.keys()) + ["Summary"]
    tabs = st.tabs(tab_names)

    for i, (name, r) in enumerate(results.items()):
        with tabs[i]:
            _acc_render_single(r)

    with tabs[-1]:
        _acc_render_summary(results)


def _acc_render_single(r: dict) -> None:
    w     = r.get("wer",   {})
    rouge = r.get("rouge", {})
    b     = r.get("bleu",  {})

    st.divider()

    def _pct(v: Any) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    cols = st.columns(4)
    cols[0].metric("WER",      _pct(w.get("wer"))            if "error" not in w else "—")
    cols[1].metric("CER",      _pct(w.get("cer"))            if "error" not in w else "—")
    cols[2].metric("Norm WER", _pct(w.get("normalized_wer")) if "error" not in w else "—")
    cols[3].metric("BLEU",     _pct(b.get("bleu"))           if "error" not in b else "—")

    c2 = st.columns(3)
    c2[0].metric("ROUGE-1 F1", _pct(rouge.get("rouge1", {}).get("f1")))
    c2[1].metric("ROUGE-2 F1", _pct(rouge.get("rouge2", {}).get("f1")))
    c2[2].metric("ROUGE-L F1", _pct(rouge.get("rougeL", {}).get("f1")))

    with st.expander("BLEU Details"):
        if "error" in b:
            st.caption(f"Unavailable: {b['error']}")
        else:
            st.markdown(f"n-gram precisions: `{b.get('precisions', '—')}`  \n"
                        f"Brevity penalty: `{b.get('brevity_penalty', '—')}`  \n"
                        f"Hyp len: `{b.get('sys_len','—')}` · Ref len: `{b.get('ref_len','—')}`")

    with st.expander("Readability Comparison"):
        hr = r.get("hypothesis", {}).get("readability", {})
        rr = r.get("reference",  {}).get("readability", {})
        st.dataframe(pd.DataFrame([
            {"": "Hypothesis", **hr},
            {"": "Reference",  **rr},
        ]).set_index(""), use_container_width=True)

    diff_html = r.get("diff_word_html", "")
    if diff_html:
        st.markdown(f'<div class="diff-view">{diff_html}</div>', unsafe_allow_html=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dc1, dc2 = st.columns(2)
    dc1.download_button("JSON Report", cm.generate_report_json(r), f"report_{ts}.json",
                        "application/json", key=f"dl_s_json_{r.get('label','')}")
    dc2.download_button("TXT Report",  cm.generate_report_txt(r),  f"report_{ts}.txt",
                        "text/plain",       key=f"dl_s_txt_{r.get('label','')}")


def _acc_render_summary(results: dict[str, Any]) -> None:
    def _pct(v: Any) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    rows = []
    for name, r in results.items():
        w     = r.get("wer",   {})
        rouge = r.get("rouge", {})
        b     = r.get("bleu",  {})
        rows.append({
            "Model":     name,
            "WER":       _pct(w.get("wer"))            if "error" not in w else "—",
            "Norm WER":  _pct(w.get("normalized_wer")) if "error" not in w else "—",
            "ROUGE-L":   _pct((rouge.get("rougeL") or {}).get("f1")),
            "BLEU":      _pct(b.get("bleu"))           if "error" not in b else "—",
        })

    st.markdown("**Model Comparison**")
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Route
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# Page: History
# ══════════════════════════════════════════════════════════════════════════

def _page_history() -> None:
    st.markdown("## History")
    st.caption("Past jobs from Google Drive. Click **Refresh** to reload.")

    if not st.session_state.drive_connected:
        st.warning("Drive not connected — authenticate first from the sidebar.")
        return

    if st.button("↻ Refresh", key="hist_refresh"):
        st.session_state.history_items = None
        st.session_state._hist_loading = False
        st.rerun()

    # Two-pass loading to prevent the Upload page's fragment timer from racing
    # against the blocking Drive API call. Pass 1 is instant: this page becomes
    # the "last complete render", so any stale fragment fires show History content
    # (not the old Upload file uploader). Pass 2 does the actual load.
    if st.session_state.history_items is None:
        if not st.session_state.get("_hist_loading", False):
            st.session_state._hist_loading = True
            st.rerun()
        with st.spinner("Loading job history from Drive…"):
            try:
                st.session_state.history_items = db.list_job_history()
            except Exception as exc:
                st.error(f"Could not load history: {exc}")
                st.session_state.history_items = []
        st.session_state._hist_loading = False

    items = st.session_state.history_items or []

    if not items:
        st.info("No jobs found in the Drive output folder.")
        return

    for item in items:
        jid   = item["job_id"]
        stage = item["stage"]
        ts_raw = item.get("updated_at", "")
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts = dt.strftime("%d %b %Y · %H:%M UTC")
        except Exception:
            ts = ts_raw[:19] if ts_raw else "—"

        color  = {"done": "#2ee89b", "error": "#e84a2e"}.get(stage, "#e8a520")
        symbol = {"done": "◼", "error": "◻"}.get(stage, "▶")

        fname_raw = item.get("filename", "")
        label     = fname_raw if fname_raw else f"job: {jid}"
        sublabel  = f"JOB/{jid} · {ts}" if fname_raw else ts
        st.markdown(
            f'<div class="jcard" style="margin-top:8px">'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:16px;'
            f'color:#7a5c0a;min-width:18px;text-align:center">{symbol}</div>'
            f'<div style="flex:1;min-width:0">'
            f'<div style="color:#d4c4a0;font-family:\'Courier Prime\',monospace;font-size:14px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
            f'<div style="color:#352c1e;font-size:9px;font-family:\'Share Tech Mono\',monospace;'
            f'margin-top:3px;letter-spacing:0.1em;text-transform:uppercase">{sublabel}</div>'
            f'</div>'
            f'<div style="color:{color};font-size:9px;font-family:\'Share Tech Mono\',monospace;'
            f'letter-spacing:0.18em;text-transform:uppercase;white-space:nowrap;margin-left:12px">{stage}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if stage == "done":
            if st.button("Load Results →", key=f"hist_load_{jid}", type="primary"):
                with st.spinner("Loading results from Drive…"):
                    st.session_state.active_job_id    = jid
                    st.session_state.uploaded_filename = ""
                    st.session_state.pipeline_state   = "done"
                    st.session_state.acc_result = None
                    st.session_state.acc_gt     = None
                    _load_results(jid)
                    st.query_params["job_id"] = jid
                st.session_state._page = "Results"
                st.rerun()
        elif item.get("error"):
            st.caption(f"↳ {item['error'][:120]}")


# ══════════════════════════════════════════════════════════════════════════
# Auto-poll fragment + routing
# ══════════════════════════════════════════════════════════════════════════

# Only fire the poll fragment when the user is on the Upload page.
# Firing on other pages caused a race: the fragment's partial rerun froze
# the UI at the previous (Upload) render while History was still loading,
# making the file uploader bleed through into the History page.
_poll_interval: int | None = (
    config.LOCAL_POLL_INTERVAL_SEC
    if (st.session_state.pipeline_state == "processing"
        and st.session_state._page == "Upload")
    else None
)


@st.fragment(run_every=_poll_interval)
def _auto_poll() -> None:
    if st.session_state.pipeline_state == "processing" and st.session_state.active_job_id:
        _poll_job(st.session_state.active_job_id)
        if st.session_state.pipeline_state != "processing":
            st.rerun(scope="app")


_auto_poll()

_PAGES = {
    "Upload":   _page_upload,
    "Results":  _page_results,
    "Accuracy": _page_accuracy,
    "History":  _page_history,
}

_PAGES.get(st.session_state._page, _page_upload)()
