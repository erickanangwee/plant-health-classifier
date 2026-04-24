"""
api/metrics.py
══════════════
Central registry of all custom Prometheus metrics for the Plant Health
Classifier API.

Design principles
─────────────────
- All metrics are module-level singletons — instantiated once on import,
  reused across every request.
- Metric names follow the Prometheus naming convention:
    <namespace>_<subsystem>_<name>_<unit>
  where namespace = "plantguard", subsystem = functional area.
- Every metric includes a docstring (help text) that describes what it
  measures in plain English — this surfaces in the /metrics endpoint and
  in Grafana's metric browser.
- Histograms use carefully chosen buckets rather than the default ones,
  because inference latency for EfficientNet-B0 on CPU clusters around
  50–500ms, which the default Prometheus buckets do not cover well.

Metric catalogue
────────────────
HTTP layer (auto-instrumented by prometheus-fastapi-instrumentator):
  http_requests_total                  — request counter by method/path/status
  http_request_duration_seconds        — latency histogram by method/path

ML inference layer (custom):
  plantguard_predictions_total         — prediction counter by outcome label
  plantguard_confidence_score          — confidence probability histogram
  plantguard_inference_duration_seconds— end-to-end inference time histogram
  plantguard_feature_extraction_seconds— EfficientNet embedding time histogram
  plantguard_classifier_seconds        — sklearn classifier time histogram

Leaf guard layer (custom):
  plantguard_guard_decisions_total     — accepted/rejected counter
  plantguard_leaf_similarity_score     — cosine similarity histogram

Model info (custom):
  plantguard_model_info                — gauge labelled with model type + version
"""

from prometheus_client import Counter, Histogram, Info #, Gauge

# ── Prediction outcomes 
PREDICTIONS_TOTAL = Counter(
    name="plantguard_predictions_total",
    documentation="Total number of classification predictions made, by outcome.",
    labelnames=["prediction"],   # values: HEALTHY, DISEASED
)

# ── Confidence score distribution 
CONFIDENCE_HISTOGRAM = Histogram(
    name="plantguard_confidence_score",
    documentation=(
        "Distribution of model confidence scores (predicted-class probability). "
        "Low scores signal uncertain predictions worth investigating."
    ),
    labelnames=["prediction"],
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

# ── End-to-end inference latency 
# Covers steps 3–6 in the request pipeline (guard + embed + scale + classify)
INFERENCE_DURATION = Histogram(
    name="plantguard_inference_duration_seconds",
    documentation=(
        "Wall-clock time for the full inference pipeline "
        "(leaf guard + feature extraction + classifier). "
        "Excludes HTTP overhead and image decoding."
    ),
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)

# ── Feature extraction latency (EfficientNet-B0)
FEATURE_EXTRACTION_DURATION = Histogram(
    name="plantguard_feature_extraction_seconds",
    documentation=(
        "Time spent running the EfficientNet-B0 forward pass to produce "
        "a 1,280-dim feature vector. High values indicate CPU contention."
    ),
    buckets=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0],
)

# ── Sklearn classifier latency 
CLASSIFIER_DURATION = Histogram(
    name="plantguard_classifier_seconds",
    documentation=(
        "Time spent running the champion sklearn classifier (predict + predict_proba). "
        "Should be <5ms for XGBoost/RandomForest on 1,280 features."
    ),
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5],
)

# ── Leaf guard decisions 
GUARD_DECISIONS_TOTAL = Counter(
    name="plantguard_guard_decisions_total",
    documentation=(
        "Total leaf guard decisions by outcome. "
        "'accepted' images proceed to classification; "
        "'rejected' images are returned as HTTP 422."
    ),
    labelnames=["decision"],   # values: accepted, rejected
)

# ── Leaf similarity score distribution 
LEAF_SIMILARITY_HISTOGRAM = Histogram(
    name="plantguard_leaf_similarity_score",
    documentation=(
        "Distribution of cosine similarity scores between uploaded image embeddings "
        "and the training-set centroid. Scores below the threshold trigger rejection."
    ),
    labelnames=["decision"],   # accepted or rejected — lets you compare distributions
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
)

# ── Model information gauge 
MODEL_INFO = Info(
    name="plantguard_model",
    documentation=(
        "Static information about the loaded champion model. "
        "Labels include model_type and api_version. "
        "Useful for correlating metric changes with model redeployments."
    ),
)