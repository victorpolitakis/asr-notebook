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
from streamlit_autorefresh import st_autorefresh


# ══════════════════════════════════════════════════════════════════════════
# CSS  — dark theme, no `all:unset`, no `header{visibility:hidden}`
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Hide Streamlit chrome — NOT the header element (breaks sidebar toggle) */
#MainMenu, footer { visibility: hidden; }
[data-testid="stStatusWidget"] { visibility: hidden; }

/* App chrome */
.stApp { background: #0d0d16; }
[data-testid="stSidebar"] { background: #080810 !important; border-right: 1px solid #181828 !important; }

/* Typography */
h1, h2, h3 { color: #e0e0f0 !important; font-weight: 300 !important; letter-spacing: -0.02em !important; }
[data-testid="stMarkdownContainer"] p { color: #a0a0b8; line-height: 1.7; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #12121e; border: 1px solid #1c1c2e;
    border-radius: 8px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #555 !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #dde0f0 !important; font-size: 22px !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #12121e; border: 1px dashed #2a2a40; border-radius: 10px;
}

/* Inputs / textareas */
textarea, input[type="text"] {
    background: #12121e !important;
    border: 1px solid #1c1c2e !important;
    color: #c0c0d8 !important;
    border-radius: 6px !important;
}
textarea:focus, input:focus {
    border-color: #7aa2f7 !important;
    box-shadow: 0 0 0 2px rgba(122,162,247,0.15) !important;
    outline: none !important;
}

/* Selectbox / radio */
[data-baseweb="select"] > div { background: #12121e; border: 1px solid #1c1c2e; border-radius: 6px; }
[data-baseweb="radio"] label { font-size: 13px; color: #a0a0b8; }

/* Tabs */
[data-baseweb="tab-list"] {
    background: #12121e; border-radius: 8px; padding: 3px;
    border: 1px solid #1c1c2e; gap: 2px;
}
[data-baseweb="tab"] { color: #555; font-size: 12px; border-radius: 6px; padding: 5px 16px; }
[aria-selected="true"] { background: rgba(122,162,247,0.12) !important; color: #7aa2f7 !important; }

/* Divider */
hr { border-color: #181828 !important; margin: 1.2rem 0 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px; border-left-width: 3px; }

/* Progress */
[data-testid="stProgress"] > div > div { background: #7aa2f7; }

/* Dataframe */
[data-testid="stDataFrame"] { font-size: 12px; }

/* ── Custom HTML blocks used in this app ── */
.seg {
    background: #12121e;
    border-left: 3px solid #2a2a48;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 4px;
    font-size: 13px;
    line-height: 1.7;
    color: #b0b0cc;
}
.seg .lbl {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 5px;
}
.chip {
    display: inline-block;
    background: #12121e;
    border: 1px solid #1c1c2e;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px 4px 2px 0;
    font-size: 11px;
    color: #888;
    font-family: 'IBM Plex Mono', monospace;
}
.jcard {
    background: #12121e;
    border: 1px solid #1c1c2e;
    border-radius: 10px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 12px;
}
.diff-view {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 2;
    max-height: 420px;
    overflow-y: auto;
    padding: 14px;
    background: #080810;
    border: 1px solid #1c1c2e;
    border-radius: 8px;
    white-space: pre-wrap;
    word-break: break-word;
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
    "transcript":        None,
    "summary":           None,
    "acc_result":        None,
    "drive_connected":   False,
    "_page":             "Upload",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_job() -> None:
    for _k in ("active_job_id", "pipeline_state", "uploaded_filename", "transcript", "summary"):
        st.session_state[_k] = _DEFAULTS[_k]


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
            db.init_drive_structure()
            st.session_state.drive_connected = True
        except Exception:
            pass


def _load_results(job_id: str) -> None:
    """Download transcript.json + summary_outputs.json from Drive into session_state."""
    try:
        for f in db.list_files(f"{config.DRIVE_OUTPUT}/{job_id}"):
            if f["name"] == "transcript.json":
                st.session_state.transcript = db.read_json(f["id"])
            elif f["name"] == "summary_outputs.json":
                st.session_state.summary = db.read_json(f["id"])
    except Exception:
        pass


def _poll_job(job_id: str) -> None:
    """Read status.json from Drive. Transitions pipeline_state and loads results on done."""
    try:
        s = db.read_status(job_id)
        if not s:
            return
        stage = s.get("stage", "")
        if stage == "done":
            st.session_state.pipeline_state = "done"
            _load_results(job_id)
        elif stage == "error":
            st.session_state.pipeline_state = "error"
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# Boot sequence (runs on every Streamlit rerun)
# ══════════════════════════════════════════════════════════════════════════

_drive_connect_silent()

_is_processing = st.session_state.pipeline_state == "processing"

# Auto-refresh only while a job is running
st_autorefresh(
    interval=config.LOCAL_POLL_INTERVAL_SEC * 1000 if _is_processing else 999_999_999,
    key="poll_timer",
)

# On each refresh cycle, poll Drive for job updates
if _is_processing and st.session_state.active_job_id:
    _poll_job(st.session_state.active_job_id)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

def _dot(color: str, size: int = 8) -> str:
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;'
            f'border-radius:50%;background:{color};margin-right:6px;vertical-align:middle"></span>')


with st.sidebar:
    # ── Brand ──
    st.markdown(
        '<div style="padding:6px 0 22px">'
        '<div style="font-size:15px;font-weight:500;color:#dde0f0">🎙️ ECE22073</div>'
        '<div style="font-size:10px;color:#3a3a52;font-family:IBM Plex Mono,monospace;margin-top:3px">'
        'AI Audio Pipeline</div></div>',
        unsafe_allow_html=True,
    )

    # ── Navigation ──
    # type="primary" on the active page gives a clear visual active state without JS
    _cur_page = st.session_state._page
    _nav_items = [("Upload", "Upload"), ("Results", "Results"), ("Accuracy", "Accuracy Check")]
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
        f'{_dot("#3dde8f" if _drive_ok else "#ff5a5a")}'
        f'<span style="font-size:12px;color:#666">{"Drive connected" if _drive_ok else "Drive not connected"}</span>',
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
        "idle": "#333", "uploading": "#ddb83d", "processing": "#ddb83d",
        "done": "#3dde8f", "error": "#ff5a5a",
    }.get(_ps, "#333")

    st.markdown(
        f'<div style="margin-top:10px">'
        f'{_dot(_ps_color)}'
        f'<span style="font-size:12px;color:#555">Pipeline: {_ps}</span></div>',
        unsafe_allow_html=True,
    )

    _fn = st.session_state.uploaded_filename
    if _fn:
        st.caption(f"↳ {_fn[:28]}{'…' if len(_fn) > 28 else ''}")

    # Progress bar during processing
    if _is_processing and st.session_state.active_job_id:
        try:
            _status_live = db.read_status(st.session_state.active_job_id)
            if _status_live:
                _pct  = _status_live.get("progress_pct", 0)
                _stg  = _status_live.get("stage", "").replace("_", " ")
                st.progress(_pct, text=f"{_stg} · {int(_pct * 100)}%")
        except Exception:
            pass

    # Reset button once job is in a terminal state
    if _ps in ("done", "error"):
        st.write("")
        if st.button("New Job", key="sb_new_job", use_container_width=True):
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
        for seg in chunk.get("segments", []):
            if seg.get("text", "").strip():
                segs.append({
                    "speaker": seg.get("speaker", "Speaker A"),
                    "text": seg["text"].strip(),
                    "start": seg.get("start", chunk.get("start_time_sec", 0)),
                })
    return segs


# ══════════════════════════════════════════════════════════════════════════
# Page: Upload & Transcribe
# ══════════════════════════════════════════════════════════════════════════

def _page_upload() -> None:
    st.markdown("## Sources")
    st.caption("Upload an audio file to send to Colab for transcription and analysis.")

    _ps = st.session_state.pipeline_state

    if not st.session_state.drive_connected:
        st.warning("Drive not connected — use the sidebar button to authenticate first.")
        return

    # ── File uploader ──
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        disabled=(_ps == "processing"),
    )

    if uploaded and _ps in ("idle", "done", "error"):
        if st.button("Transcribe", type="primary"):
            _upload_and_submit(uploaded)
            st.rerun()

    # ── Job card ──
    jid   = st.session_state.active_job_id
    fname = st.session_state.uploaded_filename
    if jid and fname:
        _render_job_card(jid, fname, _ps)

    # ── Completion summary ──
    if _ps == "done":
        _render_upload_done_summary()


def _upload_and_submit(f: Any) -> None:
    jid = db.generate_job_id()
    ext = Path(f.name).suffix or ".wav"
    st.session_state.active_job_id    = jid
    st.session_state.uploaded_filename = f.name
    st.session_state.pipeline_state   = "uploading"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(f.read())
        tmp_path = tmp.name
    try:
        db.upload_file(tmp_path, config.DRIVE_INPUT, filename=f"{jid}{ext}")
        st.session_state.pipeline_state = "processing"
    except Exception as exc:
        st.session_state.pipeline_state = "error"
        st.error(f"Upload failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _render_job_card(jid: str, fname: str, ps: str) -> None:
    color = {"processing": "#ddb83d", "done": "#3dde8f", "error": "#ff5a5a",
             "uploading": "#ddb83d"}.get(ps, "#666")
    st.markdown(
        f'<div class="jcard">'
        f'<span style="font-size:26px">🎙️</span>'
        f'<div style="flex:1">'
        f'<div style="color:#dde0f0;font-weight:500;font-size:14px">{fname}</div>'
        f'<div style="color:#444;font-size:11px;font-family:IBM Plex Mono,monospace;margin-top:3px">job: {jid}</div>'
        f'</div>'
        f'<div style="color:{color};font-size:12px;font-family:IBM Plex Mono,monospace">{ps}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if ps == "processing":
        if st.button("↻ Check Status", key="btn_check_status"):
            with st.spinner("Polling Drive…"):
                _poll_job(jid)
            st.rerun()
        st.caption("Colab is processing. Auto-refreshes every 15 s.")


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

    if st.button("View Full Results →", type="primary"):
        st.session_state._page = "Results"
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
    tab_tr, tab_ent, tab_sum = st.tabs(["Transcript", "Entities", "Summaries"])

    with tab_tr:
        _results_transcript(t)
    with tab_ent:
        _results_entities(s)
    with tab_sum:
        _results_summaries(s)


def _results_transcript(t: dict) -> None:
    segs  = _segments_from_transcript(t)
    dur   = t.get("total_duration_sec", 0)
    langs = t.get("languages_detected", [])
    spks  = {s["speaker"] for s in segs}
    full  = t.get("full_text", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",  _fmt_time(dur))
    c2.metric("Segments",  len(segs))
    c3.metric("Languages", ", ".join(langs) if langs else "en")
    c4.metric("Speakers",  len(spks))

    if full:
        st.download_button("⬇ transcript.txt", full, file_name="transcript.txt")

    st.divider()

    if not segs:
        st.caption("No speaker segments found.")
        if full:
            st.text(full[:5000])
        return

    for seg in segs:
        border = "#7aa2f7" if "A" in seg["speaker"] else "#9ece6a"
        st.markdown(
            f'<div class="seg" style="border-left-color:{border}">'
            f'<span class="lbl">{seg["speaker"]} · {_fmt_time(seg["start"])}</span>'
            f'{seg["text"]}</div>',
            unsafe_allow_html=True,
        )


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


def _results_summaries(s: dict) -> None:
    sums = s.get("summaries") or {}
    chs  = s.get("chapters")  or []

    cols = st.columns(3)
    for col, (title, key) in zip(cols, [("TL;DR", "tldr"), ("Executive", "executive"), ("Deep Dive", "deep_dive")]):
        text = sums.get(key, "")
        with col:
            st.markdown(f"**{title}**")
            if text:
                st.text_area("", text, height=220, key=f"sum_{key}", label_visibility="collapsed")
                st.download_button("Download", text, file_name=f"{key}.txt", key=f"dl_{key}")
            else:
                st.caption("Not available.")

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

def _acc_single() -> None:
    # Hypothesis: auto-load from current job transcript if available
    t = st.session_state.transcript or {}
    hyp = t.get("normalized_full_text") or t.get("full_text") or ""
    if hyp.strip():
        st.caption("Hypothesis: using normalized transcript from the current job.")
    else:
        up_hyp = st.file_uploader("Hypothesis (.txt)", type=["txt"], key="acc_s_hyp")
        hyp = up_hyp.read().decode("utf-8", errors="replace") if up_hyp else ""

    up_ref = st.file_uploader("Reference / Ground Truth (.txt)", type=["txt"], key="acc_s_ref")
    ref = up_ref.read().decode("utf-8", errors="replace") if up_ref else ""

    c1, c2 = st.columns(2)
    c1.text_area("Hypothesis (preview)", hyp[:1500] + ("…" if len(hyp) > 1500 else ""),
                 height=140, key="acc_s_hyp_prev", disabled=True)
    c2.text_area("Reference (preview)",  ref[:1500] + ("…" if len(ref)  > 1500 else ""),
                 height=140, key="acc_s_ref_prev", disabled=True)

    if st.button("Compare", type="primary", disabled=not (hyp.strip() and ref.strip())):
        with st.spinner("Computing metrics…"):
            st.session_state.acc_result = cm.compute_all_metrics(hyp, ref)

    if r := st.session_state.acc_result:
        _acc_render_single(r)


def _acc_render_single(r: dict) -> None:
    w     = r.get("wer",   {})
    rouge = r.get("rouge", {})
    b     = r.get("bleu",  {})

    st.divider()

    cols = st.columns(4)
    cols[0].metric("WER",      _fv(w.get("wer"))            if "error" not in w else "—")
    cols[1].metric("CER",      _fv(w.get("cer"))            if "error" not in w else "—")
    cols[2].metric("Norm WER", _fv(w.get("normalized_wer")) if "error" not in w else "—")
    cols[3].metric("BLEU",     _fv(b.get("bleu"))           if "error" not in b else "—")

    c2 = st.columns(3)
    c2[0].metric("ROUGE-1 F1", _fv(rouge.get("rouge1", {}).get("f1")))
    c2[1].metric("ROUGE-2 F1", _fv(rouge.get("rouge2", {}).get("f1")))
    c2[2].metric("ROUGE-L F1", _fv(rouge.get("rougeL", {}).get("f1")))

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

    diff_mode = st.radio("Diff granularity", ["Word", "Character"], horizontal=True, key="acc_diff_g")
    diff_html = r.get("diff_word_html" if diff_mode == "Word" else "diff_char_html", "")
    if diff_html:
        st.markdown(f'<div class="diff-view">{diff_html}</div>', unsafe_allow_html=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dc1, dc2 = st.columns(2)
    dc1.download_button("JSON Report", cm.generate_report_json(r), f"report_{ts}.json",
                        "application/json", key="dl_s_json")
    dc2.download_button("TXT Report",  cm.generate_report_txt(r),  f"report_{ts}.txt",
                        "text/plain",       key="dl_s_txt")


# ══════════════════════════════════════════════════════════════════════════
# Route
# ══════════════════════════════════════════════════════════════════════════

_PAGES = {
    "Upload":   _page_upload,
    "Results":  _page_results,
    "Accuracy": _page_accuracy,
}

_PAGES.get(st.session_state._page, _page_upload)()
