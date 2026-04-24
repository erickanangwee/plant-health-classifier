"""
Plant Leaf Health Classifier — FastAPI Microservice
═════════════════════════════════════════════════════

Endpoints
─────────
  GET  /health    liveness + readiness check
  GET  /classes   output class names and label map
  GET  /metrics   Prometheus metrics scrape endpoint
  POST /predict   upload a leaf image → HEALTHY / DISEASED

Request pipeline
────────────────
  1. Validate content type
  2. Decode image
  3. Leaf guard: reject non-leaf images (cosine similarity < threshold)
     └─ records: guard decision counter + similarity histogram
  4. Extract EfficientNet-B0 features
     └─ records: feature extraction duration
  5. Apply StandardScaler
  6. Run champion classifier → label + probabilities
     └─ records: classifier duration + prediction counter + confidence histogram
  7. Return structured JSON response

Observability
─────────────
  HTTP metrics    — auto-instrumented by prometheus_fastapi_instrumentator
  Custom ML metrics — defined in api/metrics.py, recorded in predict()
  Scrape endpoint — GET /metrics (text/plain; Prometheus format)
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
# from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from api.leaf_guard import LeafGuard
from api.metrics import (
    CLASSIFIER_DURATION,
    CONFIDENCE_HISTOGRAM,
    FEATURE_EXTRACTION_DURATION,
    GUARD_DECISIONS_TOTAL,
    INFERENCE_DURATION,
    LEAF_SIMILARITY_HISTOGRAM,
    MODEL_INFO,
    PREDICTIONS_TOTAL,
)
from api.model_loader import get_model, get_scaler
from api.schemas import (
    ClassesResponse,
    HealthResponse,
    PredictionResponse,
    RejectionDetail,
)

ACCEPTED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
LABELS         = {0: "HEALTHY", 1: "DISEASED"}

# Module-level singletons initialised in lifespan
_guard:     LeafGuard | None       = None
_transform: T.Compose | None       = None
_embedder:  torch.nn.Module | None = None


def _load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy objects once at startup and record model info metric."""
    global _guard, _transform, _embedder

    print("=== Plant Health Classifier API — Starting up ===")

    print("[startup] Loading champion classifier and scaler...")
    model = get_model()
    get_scaler()

    print("[startup] Initialising leaf guard...")
    _guard = LeafGuard()

    print("[startup] Building EfficientNet-B0 feature extractor...")
    p        = _load_params()
    img_size = p["features"]["image_size"]
    weights  = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    mob      = tvm.efficientnet_b0(weights=weights)
    mob.classifier = torch.nn.Identity()
    mob.eval()
    _embedder  = mob
    _transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Record static model info for Prometheus / Grafana 
    MODEL_INFO.info({
        "model_type":   type(model).__name__,
        "api_version":  "1.0.0",
        "backbone":     p["features"].get("backbone", "efficientnet_b0"),
    })

    print("=== API ready ===")
    yield
    print("=== API shutting down ===")


app = FastAPI(
    title="Plant Leaf Health Classifier",
    description=(
        "Upload any plant leaf image to get a binary health prediction: "
        "**HEALTHY** or **DISEASED** (disease / pest infestation detected). "
        "Non-leaf images are automatically rejected. "
        "Prometheus metrics are available at `/metrics`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 
import os

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Prometheus HTTP instrumentation 
# Instruments all routes automatically: request counts, latency histograms,
# in-progress gauges. Adds a GET /metrics endpoint in Prometheus text format.
# exclude_paths keeps /metrics and /health out of the latency histograms
# (they are not ML workloads and would skew p99 latency).
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True,
                          tags=["Observability"])


# ── Feature extraction 
@torch.no_grad()
def _extract_features(image: Image.Image) -> np.ndarray:
    """Return (1, 1280) float32 EfficientNet-B0 embedding."""
    x    = _transform(image).unsqueeze(0)
    feat = _embedder(x).squeeze().numpy()
    return feat.reshape(1, -1).astype(np.float32)


# ── Endpoints 

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    try:
        model  = get_model()
        loaded = model is not None
        mtype  = type(model).__name__ if loaded else None
    except Exception:
        loaded, mtype = False, None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_type=mtype,
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Meta"])
def classes():
    return ClassesResponse(
        classes=["HEALTHY", "DISEASED"],
        label_map={"HEALTHY": 0, "DISEASED": 1},
        description=(
            "HEALTHY — no disease or pest detected. "
            "DISEASED — disease or pest infestation present."
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        415: {"description": "Unsupported image format"},
        422: {"description": "Image rejected — not a plant leaf",
              "model": RejectionDetail},
    },
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image of a plant leaf."),
):
    # ── 1. Content type validation 
    if file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Accepted formats: {sorted(ACCEPTED_TYPES)}"
            ),
        )

    # ── 2. Decode image 
    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Start overall inference timer 
    t_inference_start = time.perf_counter()

    # ── 3. Leaf guard 
    is_leaf, sim_score = _guard.check(image)

    decision_label = "accepted" if is_leaf else "rejected"
    GUARD_DECISIONS_TOTAL.labels(decision=decision_label).inc()
    LEAF_SIMILARITY_HISTOGRAM.labels(decision=decision_label).observe(sim_score)

    if not is_leaf:
        raise HTTPException(
            status_code=422,
            detail=RejectionDetail(
                filename=file.filename or "unknown",
                rejected=True,
                reason=(
                    f"Image does not appear to be a plant leaf "
                    f"(similarity={sim_score:.3f}, "
                    f"threshold={_guard.threshold:.3f}). "
                    "Please upload a clear, close-up photo of a plant leaf."
                ),
                leaf_similarity=sim_score,
                threshold_used=_guard.threshold,
            ).model_dump(),
        )

    # ── 4. Feature extraction (timed) 
    t_embed_start = time.perf_counter()
    features      = _extract_features(image)
    FEATURE_EXTRACTION_DURATION.observe(time.perf_counter() - t_embed_start)

    # ── 5. Scale 
    scaler = get_scaler()
    if scaler is not None:
        features = scaler.transform(features)

    # ── 6. Classify (timed)
    model = get_model()
    t_clf_start = time.perf_counter()
    pred_int    = int(model.predict(features)[0])
    probas      = model.predict_proba(features)[0]
    CLASSIFIER_DURATION.observe(time.perf_counter() - t_clf_start)

    confidence = float(probas[pred_int])
    pred_label = LABELS[pred_int]

    # ── Record ML metrics 
    PREDICTIONS_TOTAL.labels(prediction=pred_label).inc()
    CONFIDENCE_HISTOGRAM.labels(prediction=pred_label).observe(confidence)
    INFERENCE_DURATION.observe(time.perf_counter() - t_inference_start)

    # ── 7. Respond 
    message = (
        "No disease or pest detected. Leaf appears healthy."
        if pred_int == 0
        else "Disease or pest infestation detected. Consider consulting an agronomist."
    )

    return PredictionResponse(
        filename=file.filename or "unknown",
        prediction=pred_label,
        confidence=round(confidence, 4),
        healthy_prob=round(float(probas[0]), 4),
        diseased_prob=round(float(probas[1]), 4),
        is_leaf=True,
        leaf_similarity=sim_score,
        model_used=type(model).__name__,
        message=message,
    )