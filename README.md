# ECE22073 — AI Audio Assistant: Multilingual Podcast Summarizer

Real-time pipeline for long-form audio: VAD chunking → multi-model ASR → speaker diarization → LLM-based NER, summarization and chapter detection → TTS podcast generation. Runs at **~6× real-time** on a T4 GPU.

**Author:** Πολιτάκης Βίκτωρ  
**Supervisor:** Παναγιώτης Ζέρβας — Τμήμα ΗΜΜΥ, Πανεπιστήμιο Πελοποννήσου — Εαρινό 2026

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victoras136/asr-notebook/blob/main/Pipeline/notebook.ipynb)

---

## How it works

The system separates the lightweight UI (Streamlit) from heavy GPU compute (Google Colab) using Google Drive as a job queue. The user drops an audio file in the browser; a Colab runtime picks it up, runs the full pipeline, and writes results back to Drive. Streamlit polls for updates automatically.

```
User browser
    │  drag-drop audio
    ▼
Streamlit App  ──── Google Drive API v3 ────►  ece22073/input/{job_id}/
                                                     │  job.json + audio
                                                     ▼
                                          colab_job_watcher.py  (polls every 10 s)
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                      ▼                      ▼
                       audio_processor.py     models_registry.py     llm_integration.py
                       Silero VAD chunking    up to 6 ASR models     Ticker Window (120 s)
                       −20 dBFS · 16 kHz      in parallel            4 concurrent LLM calls
                              │
                              ▼
                    diarize_transcript.py        summary_generator.py
                    pyannote 3.1                 chapters · TL;DR
                    Speaker A / B / C            Executive · Deep Dive
                              │
                              ▼
                    podcast_pipeline.py (optional)
                    GPT script → Kokoro / Dia / Bark / XTTS-v2 / F5-TTS → MP3
                              │
                              ▼
                    ece22073/output/{job_id}/
                    status.json · transcript.json · summary_outputs.json
                    {model}_transcript.json · {podcast_id}.mp3
                              │
                    Streamlit polls Drive every 15 s
                              │
                              ▼
                    Results Page — Transcript · Entities · Chapters · Summaries · Chat Q&A
```

---

## Repository layout

```
asr-notebook/
├── App/
│   ├── streamlit_app.py          # Single-file Streamlit UI (dark amber theme)
│   ├── comparison_metrics.py     # WER / ROUGE / BLEU diff viewer
│   ├── requirements.txt          # UI-only deps (no ML — ML runs on Colab)
│   ├── Dockerfile                # Builds from GitHub; no local clone needed
│   └── docker-compose.yml
├── Pipeline/
│   ├── config.py                 # Drive paths, polling intervals, job schema
│   ├── drive_bridge.py           # Google Drive API v3 wrapper (upload / download / poll)
│   ├── colab_job_watcher.py      # Main Colab entry point — dispatches ASR + Podcast jobs
│   ├── audio_processor.py        # Silero VAD, normalization, chunking
│   ├── asr_pipeline.py           # Per-chunk ASR + confidence filtering
│   ├── models_registry.py        # 6 ASR model loaders + AVAILABLE_MODELS dict
│   ├── diarize_transcript.py     # pyannote speaker diarization
│   ├── llm_integration.py        # AccumulatedTranscript + Ticker Window
│   ├── topic_extraction.py       # Topic extraction called by Ticker Window
│   ├── transcript_normalizer.py  # LLM-based proper noun correction
│   ├── summary_generator.py      # Pass-2: chapters + 3-level summaries + Q&A backend
│   ├── podcast_pipeline.py       # Script generation + TTS synthesis + MP3 concat
│   └── notebook.ipynb            # Colab notebook (run cells 1–4)
├── Benchmarks/
│   ├── evaluate_real_pipeline.py
│   ├── benchmark_all.py
│   └── benchmark_{canary,parakeet,normalize,no_vad,lang_lock}.py
├── Samples/
│   └── sample_podcasts/          # Test audio files
└── Results/                      # Pipeline output files (local runs)
```

---

## Performance

Benchmarked on a T4 GPU. Three models, two audio conditions:

| Model | WER — 3 min clean | WER — 10 min noisy (TED) | ROUGE-L — clean | ROUGE-L — noisy |
|-------|:-----------------:|:------------------------:|:---------------:|:---------------:|
| Whisper Turbo (809 M) | **4.5%** | **6.4%** (+1.9 pp) | **97.0%** | **95.8%** |
| Whisper Large v3 (1550 M) | 7.2% | 16.6% (+9.4 pp) | 95.6% | 90.3% |
| Parakeet TDT 0.6B v3 | 6.7% | 12.8% (+6.1 pp) | 96.0% | 91.2% |

- **Throughput:** 782 s audio processed in 578 s → **6× real-time** (target was ≤ 5×)
- **Summary quality:** ROUGE-1 = 0.40 (spec met); Topic Recall 0.42–0.50
- **Memory footprint:** 1.25–1.93 GB after int8 quantisation + per-chunk audio release
- **VAD chunk size tradeoff:** 30 s chunks → 6× RT, WER 8%; 10 s chunks → 7× RT, WER 6% (default: 30 s)

> **Recommended default: Whisper Turbo** — lowest WER in both conditions, least noise degradation (+1.9 pp vs +9.4 pp for Large v3). Speaker Diarization (pyannote) adds ~5–6 s overhead per chunk, consuming ~25% of total wall time.

---

## Audio processing details

`audio_processor.py` prepares every file the same way before ASR:

| Step | Value |
|------|-------|
| Loudness normalization | −20 dBFS |
| Sample rate | 16 kHz mono |
| VAD model | Silero VAD v5 — stateful RNN, 512-sample frames |
| Chunk target | 25–30 s of continuous speech |
| Silence threshold | ≥ 0.5 s gap to cut; gaps < 0.3 s discarded |
| Confidence filtering | Chunks below threshold are discarded |

---

## LLM integration — Ticker Window

`llm_integration.py` accumulates transcript chunks. Every 120 s of high-confidence speech, `AccumulatedTranscript` fires **four concurrent async LLM calls** via `asyncio.gather()`:

1. **Topic Extraction** (`topic_extraction.py`) — sliding ticker windows over segments
2. **Entity Recognition** — persons, organizations, keywords with deduplication
3. **Three-level Summarization** — TL;DR · Executive Summary · Deep Dive (JSON)
4. **Chapter Detection** — YouTube-style `{title, start_sec}` per topic group

**Transcript Normalization** (Pass-1.5): `transcript_normalizer.py` corrects cross-language proper nouns (e.g., *Ιαν Λε Κων* → *Yann LeCun*) using a strict LLM prompt. Anti-hallucination guard: if the output length ratio falls outside `0.85–1.15` or more than 10% of timestamps are lost, the change is rejected and the original is kept.

**Pass-2** (`summary_generator.py`) runs after all chunks complete, consuming the accumulated ticker results to produce the final `summary_outputs.json`:

```json
{
  "chapters": [{"index": 1, "title": "...", "start_sec": 0.0, "end_sec": 120.0, "summary": "..."}],
  "entities": {"persons": [], "organizations": [], "keywords": []},
  "summaries": {
    "tldr": "...",
    "executive": "...",
    "deep_dive": {"overview": "...", "bullet_points": [], "key_takeaways": [], "action_items": []}
  },
  "qa_logs": []
}
```

---

## Streamlit pages

| Page | What it does |
|------|-------------|
| **📤 Upload & Transcribe** | Drop `.wav`/`.mp3`/`.m4a`, choose ASR model(s), submit job — progress bar polls Drive every 15 s |
| **📊 Results** | Per-model transcript tabs · Entities panel · YouTube Chapters with timestamps · TL;DR / Executive / Deep Dive summaries · Chat Q&A backed by `query_transcript()` |
| **🎧 Podcast Studio** | Configure tone (casual / academic / debate / interview), length, and TTS model per speaker — generates two-speaker MP3 |
| **🕘 History** | Reload and inspect any past job from the Drive output folder |

---

## Local setup — Streamlit UI

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- Google Cloud project with Drive API enabled and an OAuth 2.0 Desktop client
- `OPENAI_API_KEY`

### Option A — Docker (recommended)

The image builds **directly from GitHub** — no local clone required:

```bash
# Create a working directory anywhere on your machine
mkdir ece22073 && cd ece22073

# Download only the two files you need
curl -O https://raw.githubusercontent.com/victoras136/asr-notebook/main/App/docker-compose.yml
curl -O https://raw.githubusercontent.com/victoras136/asr-notebook/main/App/.env.example

# Add your credentials
cp .env.example .env           # then edit .env → paste OPENAI_API_KEY
cp /path/to/credentials.json . # OAuth client downloaded from Google Cloud Console
touch token.json               # placeholder — overwritten on first login

# Start
docker compose up -d
```

Streamlit opens at [http://localhost:8501](http://localhost:8501).

To redeploy after a code update: `docker compose up -d --build`

### Option B — Native Python

```bash
git clone https://github.com/victoras136/asr-notebook.git
cd asr-notebook

# System deps (macOS)
brew install ffmpeg espeak-ng

pip install -r App/requirements.txt

# Place App/credentials.json (OAuth 2.0 Desktop client from Google Cloud Console)

cd App
streamlit run streamlit_app.py
```

### First OAuth flow

On first launch, click **🔗 Connect Google Drive** in the sidebar. A browser tab opens — approve the consent screen. A `token.json` is written to `App/`. All subsequent launches use the cached token.

---

## Colab setup — GPU backend

The Colab notebook handles all heavy compute: ASR, diarization, LLM calls, TTS.

1. Open: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victoras136/asr-notebook/blob/main/Pipeline/notebook.ipynb)
2. Add **Colab Secrets** (key icon in the left panel):

   | Secret | Purpose |
   |--------|---------|
   | `GITHUB_TOKEN` | GitHub PAT with `repo:read` scope — clones the repo |
   | `HF_TOKEN` | HuggingFace token — required for pyannote (gated model) and Qwen |
   | `OPENAI_API_KEY` | LLM calls (normalization, NER, summaries, podcast script) |

3. Run cells **1 → 4** in order: clone → install deps → Google Drive auth → start watcher
4. Cell 4 (`colab_job_watcher.py`) loops every 10 s and auto-stops after 5 min of idle to preserve Colab runtime

The watcher auto-processes any `.wav`, `.mp3`, or `.m4a` file that appears in `ece22073/input/`.

### Job state machine

```
uploading → asr → normalization → summary → [podcast_script → podcast_tts] → done
                                                                            ↘ error / stalled
```

A job is `stalled` if no status update arrives within 10 minutes (configurable via `STALL_TIMEOUT_SEC` in `config.py`). The Streamlit app can resume polling from Drive state even after a Colab disconnect.

---

## Google Drive folder structure

Created automatically on first run:

```
ece22073/
├── input/
│   ├── {job_id}/
│   │   ├── job.json          ← model selection, language hint, job config
│   │   └── audio.{ext}       ← uploaded audio file
│   ├── podcast_jobs/
│   │   └── {job_id}.json     ← 2-speaker podcast config
│   └── processed/            ← completed ASR jobs moved here
├── output/
│   ├── {job_id}/
│   │   ├── status.json                ← live progress + stage
│   │   ├── transcript.json            ← diarized transcript (primary model)
│   │   ├── summary_outputs.json       ← chapters + entities + summaries + Q&A logs
│   │   └── {model}_transcript.json   ← per-model transcript (multi-model jobs)
│   └── podcasts/
│       └── {podcast_id}.mp3
└── models/                   ← Drive-cached model weights (Kokoro, F5-TTS)
```

---

## Configuration

All constants live in `Pipeline/config.py` and are imported by every module — nothing is hardcoded elsewhere.

| Variable | Default | Where set | Purpose |
|----------|---------|-----------|---------|
| `OPENAI_API_KEY` | — | `.env` | All LLM calls |
| `HF_TOKEN` | — | `.env` / Colab Secret | pyannote + Qwen (gated models) |
| `LLM_MODEL` | `gpt-4o-mini` | `config.py` | NER, summaries, normalization, podcast script |
| `LLM_BASE_URL` | `None` (OpenAI) | `config.py` | Override for local LM Studio / Ollama |
| `POLL_INTERVAL_SEC` | `10` | `config.py` | Colab watcher loop |
| `LOCAL_POLL_INTERVAL_SEC` | `15` | `config.py` | Streamlit Drive refresh |
| `STALL_TIMEOUT_SEC` | `600` | `config.py` | Mark job stalled after N seconds of no update |
| `MIN_CHUNK_SEC` | `25` | `audio_processor.py` | VAD chunk target floor |
| `MAX_CHUNK_SEC` | `30` | `audio_processor.py` | VAD chunk hard ceiling |

---

## ASR models

Registered in `Pipeline/models_registry.py`. All unload themselves and clear CUDA/MPS cache after inference.

| Key | Model | Backend | Languages | Params |
|-----|-------|---------|-----------|--------|
| `whisper-turbo` | `openai/whisper-large-v3-turbo` | faster-whisper · CTranslate2 int8 | multilingual | 809 M |
| `whisper-large-v3` | `openai/whisper-large-v3` | faster-whisper · CTranslate2 int8 | multilingual | 1550 M |
| `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | Transformers · AutoModelForTDT | en-only | 600 M |
| `canary` | `nvidia/canary-1b-v2` | NeMo ASRModel | multilingual | 1 B |
| `nemotron` | `nvidia/stt_en_conformer_transducer_large_nemotron` | NeMo ASRModel | en-only | — |
| `qwen` | `Qwen/Qwen2-Audio-7B-Instruct` | Transformers · Qwen2Audio | multilingual | 7 B |

Canary and Qwen require `HF_TOKEN`. Nemotron, Canary, and Parakeet require CUDA.

---

## TTS models (Podcast Studio)

Registered in `Pipeline/podcast_pipeline.py`. All models are loaded lazily; only the selected model is loaded per session.

| Model | Size | Multi-speaker | Drive cache | Notes |
|-------|------|:-------------:|:-----------:|-------|
| Kokoro-82M | ~2 GB | ✅ (via style tensor) | ✅ | Fastest; RTF < 0.1 on T4 |
| Dia-1.6B | ~10 GB | ✅ (native) | ❌ (re-downloads) | Highest quality 2-speaker |
| Bark | ~8 GB | ✅ (via voice presets) | ❌ (re-downloads) | Expressive, slower |
| XTTS-v2 | — | ✅ | ❌ | Coqui TTS |
| F5-TTS | ~4 GB | ✅ | ✅ | Flow-matching TTS |

Podcast pipeline: GPT-4o-mini generates a two-speaker script → each line synthesized with the chosen TTS model → segments concatenated with 300 ms silence between turns → exported as 128 kbps MP3.

---

## Switching Google accounts

**Colab:** Runtime → *Disconnect and delete runtime* → re-run all cells → pick the new account in the auth browser prompt.

**Streamlit:**
1. Replace `App/credentials.json` with the new account's OAuth client *(see [GOOGLE_CREDENTIALS_SETUP.md](GOOGLE_CREDENTIALS_SETUP.md))*
2. Delete `App/token.json` (holds the cached token for the old account)
3. Restart Streamlit → sidebar → **🔗 Connect Google Drive** → sign in with the new account

> Both Colab and Streamlit **must use the same Google account**. If they differ, audio uploads land in one Drive and the Colab watcher polls a different one.

---

## Running the pipeline locally

```bash
# Full pipeline on a local audio file
python3 -c "
import sys; sys.path.insert(0, 'Pipeline')
from run_pipeline import run_pipeline
run_pipeline('Samples/sample_podcasts/bilingual_long.wav')
"
```

### Evaluation

```bash
# WER + ROUGE against ground truth
python3 -c "
import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline')
from evaluate_real_pipeline import run_real_evaluation
run_real_evaluation()
"

# Run all benchmarks
python3 -c "
import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline')
import benchmark_all
"
```
