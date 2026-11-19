# AegisRAG API image.
#
# Only the FastAPI service lives here. Model inference runs in the separate
# official Ollama container (see compose.yaml) and is reached over the private
# compose network at http://ollama:11434 -- it is never published to the host.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so edits to application code reuse the cached layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY pytest.ini ./
COPY tests/ ./tests/

# Writable location for the SQLite store, owned by the unprivileged user.
RUN useradd --create-home --uid 10001 aegis \
    && mkdir -p /data \
    && chown -R aegis:aegis /data /app
USER aegis

ENV DB_PATH=/data/aegisrag.sqlite3 \
    RECORDS_SEED_PATH=/app/data/synthetic_records.json \
    LLM_PROVIDER=ollama \
    LLM_BASE_URL=http://ollama:11434 \
    LLM_MODEL=qwen3:0.6b

EXPOSE 8000

# Reports "degraded" (HTTP 200) when the API is up but the model is not, so the
# container stays healthy while Ollama is still pulling its model.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
