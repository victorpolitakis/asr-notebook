"""
pages/accuracy.py — Single + bulk transcript comparison against ground truth.

Uses comparison_metrics.py for all computations. Results are cachable
in session_state. Downloadable JSON/TXT reports.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure Pipeline/ and App/ are importable
_app_dir = Path(__file__).parent.parent
_pipeline_dir = _app_dir.parent / "Pipeline"
for _d in (_app_dir, _pipeline_dir):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import comparison_metrics as cm


def render() -> None:
    st.markdown("## Accuracy Check")
    st.caption("Compare pipeline output against a reference transcript.")

    mode = st.radio(
        "Mode", ["Single Comparison", "Bulk Comparison"],
        horizontal=True, key="acc_mode",
    )

    if mode == "Single Comparison":
        _single()
    else:
        _bulk()


# ── Single Comparison ──────────────────────────────────────────────

def _single() -> None:
    ref = _file_to_text(st.file_uploader("Reference / Ground Truth (.txt)", type=["txt"], key="acc_ref"))

    # Hypothesis: either from pipeline or uploaded
    transcript = st.session_state.get("transcript") or {}
    auto_hyp = transcript.get("normalized_full_text") or transcript.get("full_text") or ""
    if auto_hyp.strip():
        st.caption("Using normalized transcript from current job.")
        hyp = auto_hyp
    else:
        hyp = _file_to_text(st.file_uploader("Pipeline output (.txt)", type=["txt"], key="acc_hyp"))

    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Hypothesis", hyp[:1500] + ("…" if len(hyp) > 1500 else ""), height=160, key="hyp_area", disabled=True)
    with c2:
        st.text_area("Reference", ref[:1500] + ("…" if len(ref) > 1500 else ""), height=160, key="ref_area", disabled=True)

    if st.button("Compare", type="primary", disabled=not (hyp.strip() and ref.strip())):
        with st.spinner("Computing..."):
            st.session_state["acc_result"] = cm.compute_all_metrics(hyp, ref)

    result = st.session_state.get("acc_result")
    if result:
        _render_single_result(result)


def _render_single_result(result: dict) -> None:
    w = result.get("wer", {})
    r1 = result.get("rouge", {}).get("rouge1", {})
    r2 = result.get("rouge", {}).get("rouge2", {})
    rL = result.get("rouge", {}).get("rougeL", {})
    b = result.get("bleu", {})

    st.divider()
    st.markdown("### Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WER", f"{w.get('wer', '—')}" if "error" not in w else "—")
    c2.metric("CER", f"{w.get('cer', '—')}" if "error" not in w else "—")
    c3.metric("Norm. WER", f"{w.get('normalized_wer', '—')}" if "error" not in w else "—")
    bleu_v = f"{b.get('bleu', '—')}" if "error" not in b else "—"
    c4.metric("BLEU", bleu_v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROUGE-1 F1", f"{r1.get('f1', '—')}" if r1 else "—")
    c2.metric("ROUGE-2 F1", f"{r2.get('f1', '—')}" if r2 else "—")
    c3.metric("ROUGE-L F1", f"{rL.get('f1', '—')}" if rL else "—")
    c4.metric("Brev. Penalty", f"{b.get('brevity_penalty', '—')}" if isinstance(b.get('brevity_penalty'), (int, float)) else "—")

    # Readability
    with st.expander("Readability Comparison"):
        hyp_r = result.get("hypothesis", {}).get("readability", {})
        ref_r = result.get("reference", {}).get("readability", {})
        df = pd.DataFrame([
            {"": "Hypothesis", "Words": hyp_r.get("word_count"), "Chars": hyp_r.get("char_count"),
             "Sentences": hyp_r.get("sentence_count"), "Read time": f"{hyp_r.get('estimated_reading_time_sec', 0):.0f}s"},
            {"": "Reference", "Words": ref_r.get("word_count"), "Chars": ref_r.get("char_count"),
             "Sentences": ref_r.get("sentence_count"), "Read time": f"{ref_r.get('estimated_reading_time_sec', 0):.0f}s"},
        ]).set_index("")
        st.dataframe(df, use_container_width=True)

    # Diff
    st.markdown("### Diff")
    diff_html = result.get("diff_word_html", "")
    if diff_html:
        st.markdown(
            f'<div class="diff-container">{diff_html}</div>',
            unsafe_allow_html=True,
        )

    # Downloads
    st.divider()
    dc1, dc2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with dc1:
        st.download_button("Download JSON Report", cm.generate_report_json(result),
                           file_name=f"comparison_{ts}.json", mime="application/json")
    with dc2:
        st.download_button("Download TXT Report", cm.generate_report_txt(result),
                           file_name=f"comparison_{ts}.txt", mime="text/plain")


# ── Bulk Comparison ────────────────────────────────────────────────

def _bulk() -> None:
    ref = _file_to_text(st.file_uploader("Reference / Ground Truth (.txt)", type=["txt"], key="acc_bulk_ref"))

    st.divider()
    st.caption("Add one or more hypothesis files to compare against the reference.")

    entries = st.session_state.setdefault("acc_bulk_entries", [])
    if not entries:
        entries.append({"label": "Model 1", "hypothesis": "", "filename": ""})

    # Use a fixed key per entry that persists by index.
    # To avoid reindex bugs, we use a "stable_id" assigned on creation.
    for i, entry in enumerate(entries):
        eid = entry.setdefault("_id", f"entry_{i}_{datetime.now().timestamp()}")
        with st.container():
            ec1, ec2, ec3 = st.columns([2, 5, 1])
            with ec1:
                entry["label"] = st.text_input(
                    "Model name", value=entry["label"],
                    key=f"acc_label_{eid}", placeholder="e.g. faster-whisper turbo",
                )
            with ec2:
                up = st.file_uploader(
                    "Hypothesis (.txt)", type=["txt"],
                    key=f"acc_hyp_{eid}", label_visibility="collapsed",
                )
                if up is not None:
                    entry["hypothesis"] = up.read().decode("utf-8", errors="replace")
                    entry["filename"] = up.name
                if entry.get("filename") and up is None:
                    st.caption(f"Loaded: {entry['filename']}")
            with ec3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✕", key=f"acc_rm_{eid}"):
                    entries[:] = [e for e in entries if e.get("_id") != eid]
                    st.rerun()

    if st.button("＋ Add Another Model", use_container_width=True):
        entries.append({"label": f"Model {len(entries) + 1}", "hypothesis": "", "filename": ""})
        st.rerun()

    st.divider()

    valid = [e for e in entries if e.get("hypothesis", "").strip()]
    can_compare = bool(ref.strip()) and bool(valid)

    if st.button("Compare All", type="primary", disabled=not can_compare):
        with st.spinner(f"Comparing {len(valid)} models..."):
            st.session_state["acc_bulk_results"] = cm.compare_models(ref, valid)

    results = st.session_state.get("acc_bulk_results")
    if not results:
        return

    _render_bulk_results(results)


def _render_bulk_results(results: list[dict]) -> None:
    st.divider()
    st.markdown("### Bulk Results")

    try:
        df = cm.bulk_results_to_dataframe(results)
    except Exception as exc:
        st.error(f"Failed to build table: {exc}")
        return

    if "Error" in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row.get("Error")):
                st.warning(f"{row['Model']}: {row['Error']}")

    display = df[df.get("Error", pd.Series(dtype=object)).isna()] if "Error" in df.columns else df
    if display.empty:
        st.info("No valid comparisons.")
        return

    numeric = [c for c in display.columns if c != "Model" and display[c].dtype in ("float64", "float32", "int64")]
    def _highlight(s):
        if s.name == "Model" or s.name not in numeric:
            return [""] * len(s)
        best = s.idxmin() if "WER" in s.name or "CER" in s.name else s.idxmax()
        return ["color:#4aff9e;font-weight:500" if i == best and pd.notna(s.iloc[i]) else "" for i in range(len(s))]

    styled = display.style.apply(_highlight, subset=numeric)
    styled = styled.format({c: "{:.4f}" for c in numeric if display[c].dtype in ("float64", "float32")}, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Downloads
    dc1, dc2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with dc1:
        combined = json.dumps({"evaluated_at": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2, ensure_ascii=False)
        st.download_button("Download JSON", combined, file_name=f"bulk_{ts}.json", mime="application/json")
    with dc2:
        txt = "BULK COMPARISON\n" + "=" * 50 + "\n\n"
        for r in results:
            txt += cm.generate_report_txt(r) + "\n\n"
        st.download_button("Download TXT", txt, file_name=f"bulk_{ts}.txt", mime="text/plain")


# ── Helpers ────────────────────────────────────────────────────────

def _file_to_text(uploaded) -> str:
    if uploaded is not None:
        return uploaded.read().decode("utf-8", errors="replace")
    return ""
