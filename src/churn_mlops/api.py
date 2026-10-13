"""FastAPI inference service with Prometheus metrics."""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from . import __version__, config

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities (watch this for drift)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_ROC_AUC = Gauge("model_roc_auc", "ROC AUC of the model currently being served")

state: dict = {"model": None, "metrics": None}
STATIC_DIR = Path(__file__).parent / "static"


def load_model() -> None:
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"No model found at {config.MODEL_PATH}. Run `make train` first.")
    state["model"] = joblib.load(config.MODEL_PATH)
    if config.METRICS_PATH.exists():
        state["metrics"] = json.loads(config.METRICS_PATH.read_text())
        MODEL_ROC_AUC.set(state["metrics"]["roc_auc"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", version=__version__, lifespan=lifespan)


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, examples=[12])
    monthly_charges: float = Field(..., ge=0, le=500, examples=[79.5])
    total_charges: float = Field(..., ge=0, examples=[950.0])
    support_tickets: int = Field(..., ge=0, le=100, examples=[2])
    avg_monthly_usage_gb: float = Field(..., ge=0, le=5000, examples=[110.0])
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    payment_method: Literal["credit-card", "bank-transfer", "e-wallet", "mailed-check"]
    has_internet_backup: Literal["yes", "no"]


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    model_roc_auc: float | None = None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "version": __version__}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([customer.model_dump()])[config.FEATURES]
    proba = float(state["model"].predict_proba(row)[0, 1])
    will_churn = proba >= 0.5
    CHURN_PROBABILITY.observe(proba)
    PREDICTIONS.labels("churn" if will_churn else "stay").inc()
    return Prediction(
        churn_probability=round(proba, 4),
        will_churn=will_churn,
        model_roc_auc=(state["metrics"] or {}).get("roc_auc"),
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
