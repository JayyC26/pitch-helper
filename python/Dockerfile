# Pitch Helper on Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

# OpenCV (pulled in by mediapipe) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    PORT=7860 COOKIE_SECURE=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download the pose models at build time so the first analysis doesn't have to
RUN python -c "import pitch_helper as ph; [ph.get_model(k) for k in ('heavy', 'full', 'lite')]" \
    || echo "Model download skipped; it will download on first use."

EXPOSE 7860
# Single process (analysis jobs are tracked in memory), threaded so progress polling works
CMD ["python", "app.py"]
