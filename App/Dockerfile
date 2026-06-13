FROM python:3.11-slim

# ffmpeg for pydub audio handling, espeak-ng for TTS (if ever run locally)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy entire repo (build context = repo root when building from GitHub URL)
COPY . /repo

WORKDIR /repo/App

# Install UI-only deps — no torch, no transformers, no pyannote
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
