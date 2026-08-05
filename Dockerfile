FROM python:3.11-slim

# System deps some pipeline libs need at build time (e.g. llm-guard's
# tokenizer/model deps, lxml-style parsing used by doc/report libs).
# Keep this list lean; add to it only if a build actually fails on it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as a non-root user by convention/requirement
# for the Docker SDK on the free tier — create one and own the app dir.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install deps first so Docker layer caching skips this step when only
# source files change (this is most of your rebuild time savings).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the pipeline source.
COPY . .

# HF Spaces (Docker SDK) always routes traffic to port 7860 inside the
# container — this is not configurable, so api.py's uvicorn must bind here.
ENV PORT=7860
EXPOSE 7860

RUN mkdir -p /tmp/rfp-pipeline-uploads && chown -R appuser:appuser /app /tmp/rfp-pipeline-uploads
USER appuser

# api.py is under backend/, per its own module docstring
# ("uvicorn backend.api:app --reload --port 8000") — same idea here,
# just the HF-mandated port and no --reload in production.
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
