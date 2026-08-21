FROM python:3.11-slim

# System dependencies required by optional parsing and scanner packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Docker Spaces run the application as a non-root user.
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install --no-cache-dir ".[full]"

# Hugging Face Docker Spaces route traffic to port 7860.
ENV PORT=7860
EXPOSE 7860

RUN mkdir -p /tmp/rfp-pipeline-uploads && chown -R appuser:appuser /app /tmp/rfp-pipeline-uploads
USER appuser

CMD ["uvicorn", "rfp.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
