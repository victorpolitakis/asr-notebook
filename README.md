# ECE22073 — Multilingual Podcast Summarizer

AI-powered pipeline: ASR transcription → speaker diarization → NER → multi-tier summarization → podcast generation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victoras136/asr-notebook/blob/main/Pipeline/notebook.ipynb)

## Architecture

```
audio_processor → asr_pipeline → llm_integration → transcript_normalizer
                → topic_extraction → summary_generator → podcast_pipeline
```

### Folder Structure

| Folder | Purpose |
|--------|---------|
| `Pipeline/` | Core processing — ASR, NER, summarization, Drive bridge, Colab watcher |
| `App/` | Streamlit UI, requirements, Docker config, credentials |
| `Benchmarks/` | Evaluation scripts and multi-model benchmarks |
| `Foundational/` | Real-time processor, sanity tests |
| `Samples/` | Test audio files and ground truth |
| `Results/` | Pipeline output files |

---

## Local Setup (Streamlit Dashboard)

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- Google Cloud Console project with Drive API enabled
- OpenAI API key

### Option 1 — Docker (recommended)

```bash
cd App

# One-time: create .env from template
cp .env.example .env
# Edit .env — paste your OPENAI_API_KEY

# Build and start
docker compose up
```

Streamlit opens at [http://localhost:8501](http://localhost:8501).

Code changes reflect instantly — no rebuild (volume mount). System deps (`ffmpeg`, `espeak-ng`) and Python packages are pre-installed in the image.

### Option 2 — Native Python

```bash
# System deps (macOS)
brew install ffmpeg espeak-ng

# Python deps
pip install -r App/requirements.txt

# OAuth credentials
# 1. Go to Google Cloud Console → APIs & Services → Credentials
# 2. Create OAuth Client ID (Desktop app)
# 3. Download as credentials.json → place in App/

# Start Streamlit
cd App
streamlit run streamlit_app.py
```

### First OAuth Flow

On first launch, click **🔗 Connect Google Drive** in the sidebar. A browser tab opens — approve the OAuth consent screen. A `token.json` file is written to `App/`. Subsequent launches use the cached token.

### Drive Folder Structure

The app creates this hierarchy on your Google Drive:

```
ece22073/
  input/              ← drop .wav/.mp3/.m4a files here
  input/podcast_jobs/ ← podcast generation configs
  output/{job_id}/    ← results per job (status.json, transcript.json, etc.)
  output/podcasts/    ← generated .mp3 files
```

---

## Colab Setup (GPU Backend)

The Colab notebook handles heavy ML compute on a T4 GPU.

1. [Open notebook in Colab](https://colab.research.google.com/github/victoras136/asr-notebook/blob/main/Pipeline/notebook.ipynb)
2. Set Colab Secrets:
   - `GITHUB_TOKEN` — GitHub PAT with repo read scope
   - `HF_TOKEN` — HuggingFace token for pyannote diarization
   - `OPENAI_API_KEY` — for LLM stages
3. Run cells 1–4 in order (clone → install → auth → watcher)
4. Cell 4 polls Drive every 10s and **auto-stops after 5 min of idle** (no new jobs) to preserve Colab runtime

The watcher auto-processes new `.wav`, `.mp3`, and `.m4a` files dropped in `ece22073/input/`.

### Switching Google Accounts (Colab)

To make Colab read/write a different Drive account:

1. **Runtime → Disconnect and delete runtime** — clears the cached Colab auth session
2. Re-run all cells from the top
3. When the auth cell runs, choose the new Google account in the browser prompt

To make the **Streamlit app** write to a different Drive account:

1. Replace `App/credentials.json` with the new account's OAuth credentials  
   *(see [GOOGLE_CREDENTIALS_SETUP.md](GOOGLE_CREDENTIALS_SETUP.md) to create them)*
2. Delete `App/token.json` — this holds the cached token for the old account
3. Restart Streamlit → sidebar → **Connect Google Drive** → sign in with the new account

> Both Colab and Streamlit must use the **same** Google account, otherwise uploads go to one Drive and the watcher polls a different one.

---

## Pipeline (Local)

```bash
cd /path/to/asr-notebook
python3 -c "import sys; sys.path.insert(0, 'Pipeline'); from run_pipeline import run_pipeline; run_pipeline('Samples/sample_podcasts/bilingual_long.wav')"
```

### Evaluation

```bash
python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"
```

### Run All Benchmarks

```bash
python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); sys.path.insert(0, 'Foundational'); from benchmark_all import <module>"
```

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | LLM stages (NER, summary, normalization) |
| `NORMALIZATION_MODEL` | `gpt-5.4-mini-2026-03-17` | Transcript cleanup model |
| `ENABLE_TRANSCRIPT_NORMALIZATION` | `true` | Feature flag for LLM cleanup |
| `MAX_NORMALIZATION_CHARS` | `8000` | Chunk threshold for long transcripts |
| `HF_TOKEN` | — | HuggingFace token for pyannote diarization |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Override for local Ollama |
| `LLM_MODEL` | `gpt-5.4-mini-2026-03-17` | Model for NER + summary |

---

## Pages (Streamlit)

| Page | What it does |
|------|-------------|
| 📤 Upload & Transcribe | Drop audio → Colab transcribes → view results |
| 📝 Notebook Workspace | Live notebook embedding |
| 📊 Summaries | TL;DR, executive summary, deep dive, entity views |
| 🎧 Podcast Studio | TTS podcast generation (Kokoro, Dia, Bark) |
| 🔍 Accuracy Check | WER/ROUGE diff viewer — compare output vs ground truth |

---

## ASR Models Tested

| Model | Backend | Greek | WER (benchmark) |
|-------|---------|-------|-----------------|
| faster-whisper turbo | CTranslate2 int8 | ✅ | 0.34 |
| faster-whisper large-v3 | CTranslate2 int8 | ✅ | 0.35 |
| Parakeet TDT 0.6B | Transformers CUDA | ✅ | pending |
| Canary 1B V2 | NeMo CUDA | ✅ | pending |
