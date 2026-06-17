# ECE22073 — Technical Terms Glossary

**All terms you should know for the oral examination, organized by domain.**

---

## 1. ASR & Speech Processing

| Term | Definition |
|------|-----------|
| **ASR (Automatic Speech Recognition)** | Speech-to-text — converting spoken audio into written text |
| **WER (Word Error Rate)** | Primary ASR accuracy metric. Formula: `(Substitutions + Insertions + Deletions) / Reference Words`. Target ≤ 0.08 |
| **Normalized WER** | WER after aggressive text normalization (lowercase, strip punctuation/speaker labels/symbols). Diagnostic metric to determine if errors come from formatting or genuine transcription failures. Our Norm WER (0.31) ≈ WER (0.30) → real errors, not formatting |
| **VAD (Voice Activity Detection)** | Algorithm that detects whether a segment of audio contains human speech or silence |
| **Silero VAD** | RNN-based VAD model from `torch.hub`, runs in <1ms per frame. Stateful — internal hidden states track speech patterns across frames. Must call `reset()` between files |
| **faster-whisper** | CTranslate2-backed implementation of OpenAI Whisper. 3-8× faster than vanilla Whisper/HF Transformers on Apple Silicon via int8 quantization |
| **CTranslate2** | Inference engine optimized for CPU — uses NEON/AMX instructions on Apple Silicon and Intel MKL on x86. Supports int8 quantization for memory/speed |
| **Whisper large-v3-turbo** | 809M parameter model. Finetuned version of large-v3 with decoder layers reduced from 32 → 4. Near-identical accuracy at 2× speed. Our production model |
| **int8 quantization** | Reduces model precision from 32-bit float to 8-bit integer. ~4× memory reduction, ~3× speedup, minimal accuracy loss |
| **beam_size** | Whisper decoding parameter. Number of alternative hypotheses explored in parallel. Higher = better accuracy, slower. We use 3 (production) and 5 (experiments) |
| **mel frames / mel spectrogram** | Audio representation Whisper uses internally. 100 mel frames per second of audio. Whisper expects exactly 3000 mel frames (30 seconds) per forward pass |
| **receptive field** | The maximum audio duration a model can process in one forward pass. Whisper: 30 seconds |
| **language detection** | Whisper auto-detects spoken language per segment. Returns `language` (ISO 639-1 code) and `language_probability` (confidence) |
| **TranscriptionInfo** | faster-whisper dataclass returned from `model.transcribe()`. Fields: `language`, `language_probability`, `duration`, `all_language_probs` (per-language probability distribution) |
| **hallucination filtering** | Discarding Whisper segments with low confidence (`avg_logprob < threshold`) or high `no_speech_prob`. Prevents garbled text from silence/noise |
| **audio normalization (dBFS)** | Scaling all audio to the same loudness level. Target: -20 dBFS. Formula: `loudness_delta = target_dBFS - current_dBFS`, then `audio.apply_gain(loudness_delta)` |
| **sample rate** | Audio samples per second. We resample everything to 16000 Hz (16 kHz) — required by both Silero VAD and Whisper |
| **mono audio** | Single-channel audio. We convert stereo to mono by averaging channels |
| **PCM (Pulse Code Modulation)** | Raw digital audio format. float32 samples in range [-1.0, 1.0] for ML models |

## 2. Diarization & Speakers

| Term | Definition |
|------|-----------|
| **Speaker Diarization** | Determining "who spoke when" in an audio recording. Answers: how many speakers, who spoke at what time |
| **PyAnnote (pyannote.audio)** | Industry-standard speaker diarization library. Uses 3 models: segmentation (voice activity per frame), embedding (speaker fingerprint), PLDA (clustering) |
| **PyAnnote pipeline** | `pyannote/speaker-diarization-3.1` — the specific model we use. Requires HuggingFace token (`HF_TOKEN`) for download |
| **DiarizeOutput** | Modern PyAnnote return type. Contains `.speaker_diarization` attribute for back-compat with `Annotation` API |
| **Annotation** | PyAnnote object representing speaker turns. Iterated with `.itertracks(yield_label=True)` |
| **Speaker overlap** | When multiple speakers talk simultaneously. We handle via maximum-overlap heuristic per segment |

## 3. LLM & NLP

| Term | Definition |
|------|-----------|
| **LLM (Large Language Model)** | Deep learning model trained on massive text corpora. We use OpenAI API (gpt-5.4-mini) for NER, summarization, and Q&A |
| **NER (Named Entity Recognition)** | Identifying and classifying named entities (persons, organizations, locations, products) in text |
| **Live Ticker** | Our pattern: every ~120 seconds of accumulated transcribed text triggers a background LLM call for NER + segment summarization. Async, non-blocking |
| **Entity Registry** | Deduplicated, frequency-ranked list of all entities found across the entire transcript. Updated incrementally as ticker calls complete |
| **Topic Recall** | Evaluation metric: `|extracted_topics ∩ reference_topics| / |reference_topics|`. Target ≥ 0.80. We achieve 0.42-0.50 |
| **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** | Summary quality metric. ROUGE-1: unigram overlap. ROUGE-L: longest common subsequence. Target ROUGE-1 ≥ 0.40. We achieve 0.40 |
| **Abstractive Summarization** | Generating new text that captures the essence (not extracting existing sentences). Three tiers: TL;DR, Executive, Deep Dive |
| **Extractive Summarization** | Selecting key sentences from the source. Not used in our pipeline |
| **Pass-1 / Pass-2** | Two-pass architecture. Pass-1: live ticker NER while audio streams. Pass-2: full summarization after stream ends |
| **System Prompt** | Instructions prepended to LLM input defining task, constraints, output format. Our normalization prompt is 800+ characters with explicit rules |
| **Temperature** | LLM generation parameter. 0 = deterministic (always same output). 0.3 = slightly creative. We use 0 for NER/normalization, 0.3 for summaries |
| **max_completion_tokens** | Maximum tokens the LLM can generate. Crucial finding: gpt-5.4-mini uses `max_completion_tokens` not `max_tokens` |
| **Hallucination (LLM)** | Model generating plausible but incorrect content. Our normalization layer has anti-hallucination guards (length ratio, speaker/timestamp preservation) |

## 4. Transcript Normalization

| Term | Definition |
|------|-----------|
| **Transcript Normalization** | LLM-based post-processing that corrects ASR errors in proper nouns. Runs after ASR, before entity extraction. Feature-flagged via `ENABLE_TRANSCRIPT_NORMALIZATION` |
| **Variant C (benchmark-proven)** | The normalization prompt design that scored best in benchmarks — explicitly lists 5 repair operations with concrete examples |
| **Entity Re-Extraction** | After normalization, a second LLM pass extracts fresh entities from the cleaned text. These replace (not merge with) the raw ticker entities |
| **Anti-hallucination validation** | 4 checks: length ratio (0.85-1.15), speaker label preservation (≥90%), timestamp preservation (≥90%), paragraph preservation (≥80%) |
| **difflib.SequenceMatcher** | Python library for comparing sequences. We use it to count "edit regions" between raw and normalized transcripts |
| **Fuzzy matching (rapidfuzz)** | Matching ASR-errors against canonical entity names using token-sort-ratio. Evaluated but ultimately replaced by LLM-based approach |
| **Greek-to-Latin transliteration** | Converting Greek-phonetic spellings to English canonical forms (Ιαν Λε Κων → Yann LeCun). Done by LLM normalization, not character-level mapping |

## 5. Machine Learning Concepts

| Term | Definition |
|------|-----------|
| **MPS (Metal Performance Shaders)** | Apple's GPU acceleration framework. PyTorch backend: `torch.device("mps")`. Used for pyannote diarization on Mac |
| **CUDA** | NVIDIA GPU acceleration. Used on Colab T4 GPU for faster-whisper and potential NVIDIA model inference |
| **float32 vs float16** | 32-bit vs 16-bit floating point. float16 uses half the memory and is 2-4× faster but not well supported on MPS. We use float32 on MPS |
| **NEON / AMX** | Apple Silicon CPU instruction sets. CTranslate2 uses these for high-performance int8 inference |
| **Encoder-Decoder** | Neural network architecture. Encoder processes input (audio → features), decoder generates output (features → text). Whisper and Canary use this |
| **Transformer** | Attention-based architecture. Whisper's encoder-decoder is Transformer-based |
| **FastConformer** | NVIDIA's optimized Conformer architecture. Used in Parakeet and Canary models. Convolution-augmented Transformer for speech |
| **CTC (Connectionist Temporal Classification)** | Loss function for sequence-to-sequence tasks where input/output alignment is unknown |
| **TDT (Token-and-Duration Transducer)** | Decoder architecture in Parakeet. Predicts both tokens and their durations |
| **RNN-T (Recurrent Neural Network Transducer)** | Sequence transduction model. Related to TDT |
| **SentencePiece Tokenizer** | Subword tokenization. Both Whisper and NVIDIA models use variants (BPE for Whisper, unified 8192/16384 vocab for Parakeet/Canary) |

## 6. Evaluation & Metrics

| Term | Definition |
|------|-----------|
| **jiwer** | Python library for WER computation. Handles text normalization (lowercase, punctuation removal, contractions expansion) |
| **rouge-score** | Python library for ROUGE computation. We use `RougeScorer(["rouge1", "rougeL"], use_stemmer=True)` |
| **rouge1_f1 vs rouge1** | The correct dictionary key in rouge-score output. Bug: older code used `rouge1`, current API returns `rouge1_f1` |
| **Latency Ratio** | Processing time / audio duration. ≤1.0 = faster than real-time. We achieve 0.74× (578s processing for 782s audio) |
| **RTFx (Real-Time Factor)** | Alternative latency metric. 1/ratio. Our RTFx ≈ 1.35× (processes 1.35 seconds of audio per second) |
| **Ground Truth (GT)** | Reference transcript used for WER computation. Stored in `Samples/sample_podcasts/{name}_gt.json` |
| **Hypothesis** | The ASR-produced transcript being evaluated against ground truth |
| **Tokenization** | Splitting text into tokens (words, subwords) for comparison. jiwer handles this automatically |
| **Stemming** | Reducing words to root form (e.g., "running" → "run"). Used in ROUGE computation (`use_stemmer=True`) |

## 7. Data Formats & Schemas

| Term | Definition |
|------|-----------|
| **SCHEMA 1 (Ticker)** | Per-window LLM output: `chunk_id`, `window_start`, `window_end`, `persons`, `organizations`, `keywords`, `main_ideas`, `segment_summary` |
| **SCHEMA 2 (Transcript)** | Full accumulated transcript: `source_file`, `total_duration_sec`, `chunks`, `ticker_results`, `full_text`, `all_persons`, `all_organizations`, `all_keywords`, `detected_languages`, `speaker_detected`, `language_distribution`, `language_switches` |
| **SummaryOutputs** | Final payload: `chapters`, `entities`, `summaries` (tldr, executive, deep_dive), `qa_logs` |
| **EntityRegistryDict** | Deduplicated entity lists with counts: `persons`, `organizations`, `keywords`, `main_ideas`, `segment_summaries` |
| **TypedDict** | Python type hint for dictionaries with known keys. Used for all cross-module schemas |
| **WAV (Waveform Audio File Format)** | Uncompressed audio. 44-byte header + raw PCM. Our native fallback parses this without ffmpeg |
| **MP3 (MPEG Audio Layer III)** | Compressed audio. Requires ffmpeg via pydub for loading |
| **JSONL (JSON Lines)** | One JSON object per line. Used for streaming chunk output to disk |
| **Manifest (NeMo)** | JSONL file with `audio_filepath`, `source_lang`, `target_lang`, `pnc`, `duration` fields. Required by Canary for language configuration |

## 8. Infrastructure & DevOps

| Term | Definition |
|------|-----------|
| **Google Colab** | Free Jupyter notebook environment with GPU (T4) runtime. Our GPU backend |
| **Streamlit** | Python web app framework. Our local UI with 5 pages |
| **Docker** | Container platform. Our Streamlit app runs in a `python:3.11-slim` container with ffmpeg + espeak-ng |
| **docker-compose** | Multi-container Docker orchestration. One command: `docker compose up` |
| **Volume Mount** | Docker feature mapping host directory → container directory. `../:/app` means code changes reflect instantly |
| **OAuth 2.0** | Authentication protocol. Streamlit uses Google OAuth (Desktop app) for Drive access |
| **Google Drive API** | REST API for Drive operations (upload, download, list, move). Used by `drive_bridge.py` |
| **Drive FUSE mount** | Filesystem view of Drive at `/content/drive/MyDrive/...`. BROKEN for API-uploaded files — we use API polling instead |
| **Polling** | Periodically checking for updates. Colab: 10s. Streamlit: 15s. With autorefresh disabled when idle |
| **Job ID** | Unique identifier per transcription/podcast job. 8-character hex string |
| **.env file** | Environment variable file. Contains `OPENAI_API_KEY`. Loaded by `transcript_normalizer.py` and `benchmark_all.py` |
| **Colab Secrets** | Google Colab's secret manager. Stores `GITHUB_TOKEN`, `HF_TOKEN`, `OPENAI_API_KEY` |
| **GitHub PAT** | Personal Access Token. Used by notebook Cell 1 to clone private repo |

## 9. NVIDIA ASR Ecosystem

| Term | Definition |
|------|-----------|
| **NeMo (NVIDIA NeMo Framework)** | NVIDIA's toolkit for conversational AI. Required for Canary and Parakeet models (`pip install nemo_toolkit[asr]`) |
| **Parakeet TDT 0.6B v3** | 600M parameter multilingual ASR model. FastConformer-TDT architecture. 25 European languages including Greek. Pure ASR (no translation). Uses `AutoModelForTDT` + `model.generate()` |
| **Canary 1B V2** | 978M parameter multitask ASR + AST model. FastConformer Encoder + Transformer Decoder. 25 languages. Must explicitly set `source_lang=target_lang` for ASR mode. Uses NeMo's `ASRModel.transcribe()` |
| **Canary 1B Flash** | 883M parameter model. Only 4 languages (en/de/es/fr). Does NOT support Greek — excluded from our benchmark |
| **AST (Automatic Speech Translation)** | Speech → translated text. Canary can do both ASR and AST depending on `source_lang` vs `target_lang` |
| **Granary Dataset** | NVIDIA's multilingual training corpus. 660,000 hours of pseudo-labeled data across 25 European languages |
| **Fleurs / MLS / CoVoST** | Standard multilingual ASR evaluation datasets. Parakeet Greek WER: 20.7% on Fleurs |
| **FastConformer** | Efficient Conformer variant with linearly scalable attention. Used in both Parakeet and Canary |
| **lhotse** | Audio data loading library used by NeMo. Version incompatibility caused Canary manifest API crashes on Colab |

## 10. TTS (Text-to-Speech) & Podcast Generation

| Term | Definition |
|------|-----------|
| **Kokoro-82M** | Lightweight TTS model (~2GB VRAM). Our primary podcast TTS engine. Multi-voice: `af_heart`, `af_nicole`, `am_michael`, etc. Synthesizes 155s audio in ~10s |
| **Dia-1.6B** | Native multi-speaker TTS model. Requires 10GB VRAM |
| **Bark** | OpenAI's TTS model. 8GB VRAM |
| **XTTS-v2** | Coqui TTS model. 6GB VRAM. CPML license |
| **F5-TTS** | Flow-matching TTS. 4GB VRAM |
| **espeak-ng** | Text-to-phoneme engine. Required by Kokoro |
| **VRAM (Video RAM)** | GPU memory. T4 Colab GPU has ~16GB. We estimate total VRAM for podcast generation in the UI |

## 11. Code Architecture Patterns

| Term | Definition |
|------|-----------|
| **Generator (Python)** | Function that yields values one at a time instead of returning a list. Used for streaming: `process_audio_file()` and `transcribe_file()` are generators |
| **asyncio** | Python async I/O library. Our LLM calls use `asyncio.create_task()` for non-blocking background execution |
| **`asyncio.run()`** | Creates event loop and runs coroutine. Can crash inside Streamlit callbacks → we use sync client instead |
| **Lazy Import** | Importing heavy dependencies only when needed. `_build_openai_client()` and `_get_diarization_pipeline()` use this pattern |
| **Feature Flag** | Environment variable that toggles behavior. `ENABLE_TRANSCRIPT_NORMALIZATION=true|false` |
| **sys.path.insert** | Adding directories to Python's module search path. Every file that imports from another directory needs this |
| **`__name__ == "__main__"` guard** | Python idiom: code only runs when file is executed directly, not when imported. Removed from library modules, kept in `run_pipeline.py` |
| **Session State** | Streamlit's per-user storage. We store `active_job_id`, `pipeline_state`, `chat_history`, `extra_sources`, etc. |

---

*This glossary covers every technical term referenced in the project. Be prepared to explain any of these in your own words during the oral examination.*
