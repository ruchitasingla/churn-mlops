# ---- builder: install dependencies into an isolated venv ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# API only needs the serving deps; MLflow is for training and lives in its own image.
COPY requirements.txt .
RUN grep -v '^mlflow' requirements.txt > requirements-serve.txt \
        && pip install --default-timeout=100 --retries 5 -r requirements-serve.txt

# ---- runtime: small image, non-root user ----
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src MODEL_DIR=/app/models
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
# The trained, quality-gated model is produced by CI (or `make train`) before the build.
COPY models ./models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
