# ECE22073 — Multilingual Podcast Summarizer

AI-powered pipeline: ASR transcription → speaker diarization → NER → multi-tier summarization → podcast generation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victoras136/ece22073-podcast/blob/main/notebook.ipynb)

## Architecture

```
audio_processor → asr_pipeline → llm_integration → transcript_normalizer
                → topic_extraction → summary_generator → podcast_pipeline
```

## Quick Start

```bash
pip install -r Politakis/requirements.txt
python3 setup_bilingual_test.py
python3 Politakis/run_pipeline.py <path_to_audio.wav>
```

## Streamlit Dashboard

```bash
cd Politakis && streamlit run streamlit_app.py
```

## Evaluation

```bash
python3 Politakis/evaluate_real_pipeline.py
```
