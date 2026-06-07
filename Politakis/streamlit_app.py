"""
streamlit_app.py — AI Audio Assistant  (ECE22073 Project 8)

Run:
    cd Politakis/
    streamlit run streamlit_app.py

Architecture:
  Left sidebar   → file upload, pipeline trigger, summary level, entity stats
  Main area      → metrics banner · chapters · entities · summary · Q&A chat
  All heavy work stays in run_pipeline.py / the module stack;
  this file is purely presentation + thin glue.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

# ── Page config — MUST be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="AI Audio Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR      = Path(__file__).parent / "results"
SUMMARY_FILE     = RESULTS_DIR / "summary_outputs.json"
TRANSCRIPT_FILE  = RESULTS_DIR / "transcript.json"
QUALITY_FILE     = RESULTS_DIR / "quality_metrics.json"
LATENCY_FILE     = RESULTS_DIR / "processing_time_analysis.json"

# ── Lazy module loader ───────────────────────────────────────────────────────
def _mod(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:
        st.error(f"Missing module **{name}**: `{e}`  \nRun `pip install -r requirements.txt`")
        st.stop()


# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "transcript":        None,
    "entity_registry":   None,
    "summary_outputs":   None,
    "pipeline_running":  False,
    "pipeline_done":     False,
    "uploaded_filename": None,
    "audio_bytes":       None,
    "chat_history":      [],
    "summary_level":     "TL;DR",
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body { font-family: 'Inter', sans-serif !important; }

.stApp { background: linear-gradient(135deg,#0d1117 0%,#161b22 60%,#0d1117 100%); color:#e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#13151a 0%,#1c1f26 100%);
    border-right: 1px solid #30363d;
}

/* Cards */
.card {
    background: rgba(22,27,34,.9);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.35);
}
.card h3 { color:#58a6ff; margin: 0 0 .6rem; font-size:1rem; }

/* Entity chips */
.chip       { display:inline-block; border-radius:20px; padding:3px 11px; margin:3px; font-size:.78rem; font-weight:500; }
.chip-blue  { background:rgba(88,166,255,.14); border:1px solid rgba(88,166,255,.4); color:#79c0ff; }
.chip-green { background:rgba(63,185,80,.12);  border:1px solid rgba(63,185,80,.4);  color:#56d364; }
.chip-purple{ background:rgba(188,140,255,.12);border:1px solid rgba(188,140,255,.4);color:#d2a8ff; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:.7rem;
         font-weight:600; background:rgba(88,166,255,.2); color:#58a6ff; margin-left:5px; }

/* Chapter row */
.ch-row   { display:flex; gap:1rem; padding:.6rem 0; border-bottom:1px solid #21262d; align-items:flex-start; }
.ch-ts    { font-family:monospace; font-size:.78rem; color:#58a6ff; min-width:110px; padding-top:2px; }
.ch-title { font-weight:600; color:#e6edf3; font-size:.92rem; }
.ch-sum   { color:#8b949e; font-size:.8rem; margin-top:2px; }

/* Chat bubbles */
.bubble-user { background:rgba(88,166,255,.18); border-radius:12px 12px 2px 12px;
               padding:.65rem 1rem; margin:.45rem 0; text-align:right; color:#cdd9e5; font-size:.88rem; }
.bubble-bot  { background:rgba(30,35,45,.9); border:1px solid #30363d;
               border-radius:12px 12px 12px 2px; padding:.65rem 1rem;
               margin:.45rem 0; color:#e6edf3; font-size:.88rem; line-height:1.55; }

/* Metric gate badge */
.gate-ok   { color:#3fb950; font-weight:700; }
.gate-fail { color:#f85149; font-weight:700; }

/* Main title */
h1.title { font-size:2.1rem; font-weight:700;
           background:linear-gradient(90deg,#58a6ff,#bc8cff);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Buttons */
.stButton>button { background:linear-gradient(135deg,#238636,#2ea043);
                   color:white; border:none; border-radius:8px; font-weight:600; }
.stButton>button:hover { background:linear-gradient(135deg,#2ea043,#3fb950);
                          box-shadow:0 0 10px rgba(46,160,67,.45); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _chips(items: list[dict], cls: str) -> str:
    return " ".join(
        f'<span class="chip {cls}">{e["name"]}'
        f'<span class="badge">×{e["count"]}</span></span>'
        for e in items
    )


def _gate(val: float, thr: float, lower: bool = True) -> str:
    ok = val <= thr if lower else val >= thr
    arrow = "≤" if lower else "≥"
    icon  = "✅" if ok else "❌"
    cls   = "gate-ok" if ok else "gate-fail"
    return f'<span class="{cls}">{icon} {val:.4f} ({arrow}{thr})</span>'


# ── Pipeline runner ──────────────────────────────────────────────────────────
def _run_pipeline(audio_bytes: bytes, filename: str) -> None:
    asr = _mod("asr_pipeline")
    llm = _mod("llm_integration")
    te  = _mod("topic_extraction")
    sg  = _mod("summary_generator")

    st.session_state["pipeline_running"] = True
    ph  = st.empty()          # status placeholder
    bar = st.progress(0)

    try:
        # Save to temp path (pipeline needs a file, not raw bytes)
        tmp = RESULTS_DIR / f"_upload_{filename}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(audio_bytes)

        # ── Stage 1+2: ASR ────────────────────────────────────────────────
        ph.info("⏳ **Stage 1 & 2 / 4** — VAD chunking + Whisper transcription…")
        bar.progress(8)
        chunks: list[dict] = list(asr.transcribe_file(str(tmp)))
        bar.progress(42)

        speech_chunks = sum(1 for c in chunks if c.get("is_speech"))
        total_dur     = sum(c.get("duration_sec", 0) for c in chunks)

        # ── Stage 3: LLM ticker ───────────────────────────────────────────
        ph.info("⏳ **Stage 3 / 4** — Pass-1 Live Ticker (NER + entity extraction)…")
        transcript: dict = llm.process_asr_stream_sync(chunks, source_file=str(tmp))
        bar.progress(68)

        # ── Stage 3b: Entity registry ─────────────────────────────────────
        ph.info("⏳ **Stage 3b / 4** — Building entity registry…")
        entity_registry: dict = te.build_entity_registry(transcript)
        bar.progress(78)

        # ── Stage 4: Summary ──────────────────────────────────────────────
        ph.info("⏳ **Stage 4 / 4** — Pass-2 summary generation (TL;DR / Exec / Deep Dive)…")
        summary_outputs: dict = sg.generate_summary(transcript, entity_registry)
        bar.progress(100)

        # ── Store results ─────────────────────────────────────────────────
        st.session_state.update({
            "transcript":        transcript,
            "entity_registry":   entity_registry,
            "summary_outputs":   summary_outputs,
            "pipeline_done":     True,
            "uploaded_filename": filename,
            "audio_bytes":       audio_bytes,
            "chat_history":      [],
        })

        ph.success(
            f"✅ Pipeline complete — {len(chunks)} chunks "
            f"({speech_chunks} speech, {total_dur:.1f}s audio)"
        )

    except Exception as exc:
        ph.error(f"❌ Pipeline error: {exc}")
        logger.exception("Pipeline failed")
    finally:
        st.session_state["pipeline_running"] = False
        bar.empty()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎙️ AI Audio Assistant")
    st.caption("ECE22073 · Project 8 · Victor Politakis")
    st.divider()

    # File upload
    up = st.file_uploader(
        "Upload Podcast / Audio",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="MP3, WAV, M4A, OGG, FLAC supported",
    )

    run_btn = st.button(
        "▶  Run Pipeline",
        disabled=up is None or st.session_state["pipeline_running"],
        use_container_width=True,
    )

    # Audio player
    if up is not None:
        st.audio(up, format=f"audio/{up.name.rsplit('.',1)[-1]}")
    elif st.session_state.get("audio_bytes"):
        st.audio(st.session_state["audio_bytes"])

    # Trigger pipeline
    if run_btn and up is not None:
        if up.name != st.session_state["uploaded_filename"]:
            _run_pipeline(up.read(), up.name)
            st.rerun()

    # Auto-load saved results on fresh start
    if not st.session_state["pipeline_done"]:
        sg = _mod("summary_generator")
        cached = sg.load_summary_outputs()
        if cached:
            st.session_state.update({
                "summary_outputs":  cached,
                "entity_registry":  cached.get("entities", {}),
                "pipeline_done":    True,
                "transcript":       st.session_state.get("transcript") or {},
            })
            # Also load transcript if saved
            if TRANSCRIPT_FILE.exists():
                with open(TRANSCRIPT_FILE) as fh:
                    st.session_state["transcript"] = json.load(fh)

    st.divider()

    if st.session_state["pipeline_done"]:
        st.markdown("### 📄 Summary Level")
        level = st.selectbox(
            "Detail level",
            ["TL;DR", "Executive Summary", "Deep Dive"],
            index=["TL;DR", "Executive Summary", "Deep Dive"].index(
                st.session_state["summary_level"]
            ),
            label_visibility="collapsed",
        )
        st.session_state["summary_level"] = level

        # Quick stats
        er = st.session_state["entity_registry"] or {}
        so = st.session_state["summary_outputs"] or {}
        st.divider()
        st.markdown("### 🔍 Entity Stats")
        c1, c2 = st.columns(2)
        c1.metric("Persons",  len(er.get("persons", [])))
        c2.metric("Orgs",     len(er.get("organizations", [])))
        c1.metric("Keywords", len(er.get("keywords", [])))
        c2.metric("Chapters", len(so.get("chapters", [])))

        # Language badges
        tr    = st.session_state.get("transcript") or {}
        langs = tr.get("languages_detected", [])
        if langs:
            _FLAG = {"en":"🇬🇧","el":"🇬🇷","es":"🇪🇸","de":"🇩🇪","fr":"🇫🇷",
                     "it":"🇮🇹","pt":"🇵🇹","nl":"🇳🇱","pl":"🇵🇱","ru":"🇷🇺",
                     "zh":"🇨🇳","ja":"🇯🇵","ko":"🇰🇷","ar":"🇸🇦"}
            st.divider()
            st.markdown("### 🌍 Languages Detected")
            badges = " ".join(
                f'<span class="chip chip-blue">{_FLAG.get(l,"🌐")} {l.upper()}</span>'
                for l in langs
            )
            st.markdown(badges, unsafe_allow_html=True)

        # Download results
        st.divider()
        if SUMMARY_FILE.exists():
            st.download_button(
                "⬇️  Download Results JSON",
                data=SUMMARY_FILE.read_bytes(),
                file_name="summary_outputs.json",
                mime="application/json",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="title">🎙️ AI Audio Assistant</h1>', unsafe_allow_html=True)
st.caption("Multilingual podcast transcription · NER · Hierarchical summarisation · Q&A")

# ── Evaluation metric banner ─────────────────────────────────────────────────
if QUALITY_FILE.exists() and LATENCY_FILE.exists():
    try:
        qm = json.loads(QUALITY_FILE.read_text())
        pt = json.loads(LATENCY_FILE.read_text())
        wer   = qm.get("wer",           {}).get("wer",        None)
        rouge = qm.get("rouge",         {}).get("rouge1_f1",  None)
        rec   = qm.get("topic_recall",  {}).get("recall",     None)
        lat   = pt.get("latency",       {}).get("ratio",       None)
        if all(v is not None for v in [wer, rouge, rec, lat]):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("WER",          f"{wer:.4f}",  delta="✅ ≤0.08"  if wer  <= 0.08 else "❌ >0.08",  delta_color="off")
            m2.metric("ROUGE-1 F1",   f"{rouge:.4f}", delta="✅ ≥0.40" if rouge >= 0.40 else "❌ <0.40",  delta_color="off")
            m3.metric("Topic Recall", f"{rec:.4f}",  delta="✅ ≥0.80"  if rec   >= 0.80 else "❌ <0.80",  delta_color="off")
            m4.metric("Latency",      f"{lat:.3f}×", delta="✅ ≤1.0×"  if lat   <= 1.00 else "❌ >1.0×",  delta_color="off")
    except Exception:
        pass

st.divider()

# Landing screen when no results yet
if not st.session_state["pipeline_done"]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>🎵 Upload Audio</h3>'
                    '<p style="color:#8b949e">MP3, WAV, M4A, OGG, FLAC. '
                    'Any length — stream-based processing handles hours of audio.</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>🧠 AI Analysis</h3>'
                    '<p style="color:#8b949e">Whisper ASR · Silero VAD · pyannote '
                    'diarization · GPT-4 NER in real-time 2-min ticker windows.</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>💬 Ask Anything</h3>'
                    '<p style="color:#8b949e">Chat with the full transcript. '
                    'No RAG — the entire podcast fits in a 128k-token context window.</p></div>',
                    unsafe_allow_html=True)
    st.info("👈 Upload an audio file in the sidebar, then click **▶ Run Pipeline**.")
    st.stop()


# ── Results available ────────────────────────────────────────────────────────
so:   dict = st.session_state["summary_outputs"] or {}
er:   dict = st.session_state["entity_registry"]  or {}
sums: dict = so.get("summaries", {})
level: str = st.session_state["summary_level"]


# ── YouTube Chapters ─────────────────────────────────────────────────────────
with st.expander("📺 YouTube Chapters", expanded=True):
    chapters: list[dict] = so.get("chapters", [])
    if chapters:
        for ch in chapters:
            ts = f"{_fmt(ch.get('start_sec',0))} – {_fmt(ch.get('end_sec',0))}"
            st.markdown(
                f'<div class="ch-row">'
                f'<div class="ch-ts">{ts}</div>'
                f'<div><div class="ch-title">Chapter {ch.get("index","")}: {ch.get("title","")}</div>'
                f'<div class="ch-sum">{ch.get("summary","")}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No chapters generated yet.")


# ── Named Entities ───────────────────────────────────────────────────────────
with st.expander("🔬 Named Entities & Keywords", expanded=False):
    tp, to, tk, ti = st.tabs(["👤 Persons", "🏢 Organisations", "🔑 Keywords", "💡 Main Ideas"])

    with tp:
        persons = er.get("persons", [])
        if persons:
            st.markdown(_chips(persons, "chip-blue"), unsafe_allow_html=True)
        else:
            st.info("No persons detected.")

    with to:
        orgs = er.get("organizations", [])
        if orgs:
            st.markdown(_chips(orgs, "chip-green"), unsafe_allow_html=True)
        else:
            st.info("No organisations detected.")

    with tk:
        kws = er.get("keywords", [])
        if kws:
            st.markdown(_chips(kws, "chip-purple"), unsafe_allow_html=True)
        else:
            st.info("No keywords detected.")

    with ti:
        ideas = er.get("main_ideas", [])
        if ideas:
            for i, idea in enumerate(ideas, 1):
                st.markdown(f"**{i}.** {idea}")
        else:
            st.info("No main ideas extracted.")


# ── Summary ──────────────────────────────────────────────────────────────────
st.markdown("### 📝 Summary")

if level == "TL;DR":
    tldr = sums.get("tldr", "")
    st.markdown(
        f'<div class="card">'
        f'<h3>⚡ TL;DR</h3>'
        f'<p style="font-size:1.1rem;font-weight:500;line-height:1.65;color:#e6edf3">'
        f'{tldr or "TL;DR not yet generated."}</p></div>',
        unsafe_allow_html=True,
    )

elif level == "Executive Summary":
    exec_txt = sums.get("executive", "")
    st.markdown(
        f'<div class="card"><h3>📊 Executive Summary</h3>'
        f'<div style="line-height:1.8;color:#cdd9e5;white-space:pre-wrap">'
        f'{exec_txt or "Not yet generated."}</div></div>',
        unsafe_allow_html=True,
    )

elif level == "Deep Dive":
    dd: dict = sums.get("deep_dive", {})
    if dd:
        st.markdown(
            f'<div class="card"><h3>🔭 Deep Dive — Overview</h3>'
            f'<p style="line-height:1.75;color:#cdd9e5">{dd.get("overview","")}</p></div>',
            unsafe_allow_html=True,
        )
        cl, cr = st.columns(2)
        with cl:
            if dd.get("bullet_points"):
                st.markdown("#### 📌 Key Points")
                for b in dd["bullet_points"]:
                    st.markdown(f"- {b}")
            if dd.get("key_takeaways"):
                st.markdown("#### 💡 Key Takeaways")
                for t in dd["key_takeaways"]:
                    st.markdown(f"- {t}")
        with cr:
            if dd.get("action_items"):
                st.markdown("#### ✅ Action Items")
                for a in dd["action_items"]:
                    st.markdown(f"- {a}")
            segs = er.get("segment_summaries", [])
            if segs:
                st.markdown("#### 🗂 Segment Summaries")
                for i, s in enumerate(segs, 1):
                    st.markdown(f"**Segment {i}:** {s}")
    else:
        st.info("Deep Dive not yet generated.")


# ── Q&A Chat ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💬 Ask About the Podcast")
st.caption("Answers are grounded in what was actually said — no hallucination, no RAG.")

for msg in st.session_state["chat_history"]:
    css = "bubble-user" if msg["role"] == "user" else "bubble-bot"
    icon = "🧑" if msg["role"] == "user" else "🤖"
    st.markdown(
        f'<div class="{css}">{icon} {msg["content"]}</div>',
        unsafe_allow_html=True,
    )

question = st.chat_input("Ask a question about the podcast…")
if question and question.strip():
    transcript: dict = st.session_state.get("transcript") or {}
    st.session_state["chat_history"].append({"role": "user", "content": question.strip()})

    with st.spinner("Thinking…"):
        sg = _mod("summary_generator")
        answer: str = sg.query_transcript(question.strip(), transcript)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # Persist to qa_logs
    sg.append_qa_log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question":  question.strip(),
        "answer":    answer,
        "model":     getattr(sg, "LLM_MODEL", "unknown"),
    })
    st.rerun()
