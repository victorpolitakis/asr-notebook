FROM python:3.11-slim

# System deps for pydub (ffmpeg) and kokoro (espeak-ng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
