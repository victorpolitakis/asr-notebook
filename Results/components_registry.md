# Pipeline Components — Full Registry

> Terms only. **R** = έχει δημοσιευμένο research paper. **HF** = HuggingFace-documented (model card / blog).

---

## Python Libraries

### UI & Web
| Library | Type |
|---------|------|
| streamlit | HF |
| pandas | HF |
| plotly | HF |

### Google Drive Bridge
| Library | Type |
|---------|------|
| google-auth | HF |
| google-auth-oauthlib | HF |
| google-api-python-client | HF |
| googleapiclient | HF |

### Core ML
| Library | Type |
|---------|------|
| torch (PyTorch) | R |
| torchaudio | R |
| torchvision | R |
| numpy | R |
| numba | R |
| scipy | R |

### ASR
| Library | Type |
|---------|------|
| faster-whisper | R (Whisper paper, OpenAI 2022) — CTranslate2 backend |
| transformers (HuggingFace) | HF |
| accelerate | HF |

### Diarization
| Library | Type |
|---------|------|
| pyannote.audio | R (Bredin et al.) |
| soundfile | HF |

### Audio Processing
| Library | Type |
|---------|------|
| pydub | HF |
| ffmpeg (system dep) | HF |
| espeak-ng (system dep) | HF |

### LLM Client
| Library | Type |
|---------|------|
| openai | HF |

### NLP / Evaluation Metrics
| Library | Type |
|---------|------|
| rouge-score | R (Lin, 2004) |
| jiwer | R (WER standard) |
| sacrebleu | R (Post, 2018) |

### Monitoring
| Library | Type |
|---------|------|
| psutil | HF |

### Colab Fixes
| Library | Type |
|---------|------|
| nest_asyncio | HF |
| huggingface_hub | HF |

### TTS (optional, installed from git)
| Library | Source | Type |
|---------|--------|------|
| kokoro (>=0.9.4) | PyPI | HF |
| Dia-1.6B | git: nari-labs/dia | HF |
| Bark | git: suno-ai/bark | R (Suno) |
| XTTS-v2 | git: coqui-ai/TTS | R (Coqui) |
| F5-TTS | git: SWivid/F5-TTS | R |

### Sound Backend (TTS)
| Library | Type |
|---------|------|
| soundfile | HF |

---

## Models

### Production ASR Model
| Model | HuggingFace ID | Type |
|-------|---------------|------|
| faster-whisper turbo | openai/whisper-large-v3-turbo (via CTranslate2 int8) | R — Whisper (OpenAI, 2022) |
| faster-whisper large-v3 | openai/whisper-large-v3 (via CTranslate2 int8) | R — Whisper (OpenAI, 2022) |

### Alternative ASR Models (benchmarking)
| Model | HuggingFace ID | Type |
|-------|---------------|------|
| NVIDIA Parakeet TDT 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 | R — Parakeet-TDT (NVIDIA) |
| NVIDIA Canary 1B V2 | nvidia/canary-1b-v2 (via NeMo) | R — Canary (NVIDIA) |
| Qwen2-Audio-7B-Instruct | Qwen/Qwen2-Audio-7B-Instruct | R — Qwen2-Audio (Alibaba) |
| NVIDIA STT Conformer Transducer Large Nemotron | nvidia/stt_en_conformer_transducer_large_nemotron (via NeMo) | R — Conformer + Nemotron (NVIDIA) |
| faster-whisper tiny | openai/whisper-tiny (via CTranslate2 int8) | R — Whisper (OpenAI, 2022) — language detection only |

### Voice Activity Detection
| Model | Source | Type |
|-------|--------|------|
| Silero VAD v5 | snakers4/silero-vad (via torch.hub) | R — Silero VAD |

### Speaker Diarization
| Model | HuggingFace ID | Type |
|-------|---------------|------|
| pyannote 3.1 | pyannote/speaker-diarization-3.1 | R — pyannote (Bredin et al.) |

### LLM (via API)
| Model | Type |
|-------|------|
| gpt-5.4-mini-2026-03-17 | R — GPT-4o series (OpenAI) — normalization, NER, summarization |
| gpt-4o-mini | R — GPT-4o series (OpenAI) — underlying model |

### TTS Models (podcast generation)
| Model | Type | Tested |
|-------|------|--------|
| Kokoro-82M | HF | ✅ end-to-end |
| Dia-1.6B | HF (nari-labs) | ❌ |
| Bark | R (Suno) | ❌ |
| XTTS-v2 | R (Coqui) | ❌ |
| F5-TTS | R (SWivid) | ❌ |

---

## Evaluation Metrics Used
| Metric | Library | Type |
|--------|---------|------|
| WER (Word Error Rate) | jiwer | R |
| CER (Character Error Rate) | jiwer | R |
| ROUGE-1 / ROUGE-2 / ROUGE-L | rouge-score | R (Lin, 2004) |
| BLEU | sacrebleu | R (Post, 2018) |
| Readability | custom (comparison_metrics.py) | — |
| Topic Recall | custom (evaluate.py) | — |
| Latency Ratio | custom (evaluate.py) | — |
