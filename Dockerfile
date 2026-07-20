FROM python:3.10-slim

# System dependencies — ffmpeg is required for audio export format conversion (wav/ogg/flac/m4a/webm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Hugging Face Spaces expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

# Run as a non-root user (Hugging Face Spaces best practice)
RUN useradd -m -u 1000 user && chown -R user /app
USER user

CMD ["python", "app.py"]
