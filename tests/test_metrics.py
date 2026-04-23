"""
Smoke tests for the Prometheus metrics endpoint.

Verifies:
  - /metrics returns HTTP 200
  - /metrics content-type is the Prometheus text format
  - All custom metric families are present in the output
  - Making a prediction increments the counter
"""
import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


def _make_leaf_image(color=(60, 120, 40)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.91, 0.09]])

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    mock_guard = MagicMock()
    mock_guard.check.return_value = (True, 0.83)
    mock_guard.threshold = 0.68

    mock_extract = MagicMock(return_value=np.zeros((1, 1280), dtype=np.float32))

    with (
        patch("api.main.get_model",         return_value=mock_model),
        patch("api.main.get_scaler",        return_value=mock_scaler),
        patch("api.main.LeafGuard",         return_value=mock_guard),
        patch("api.main._guard",            mock_guard),
        patch("api.main._extract_features", mock_extract),
        patch("api.main._embedder",         MagicMock()),
        patch("api.main._transform",        MagicMock()),
    ):
        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)


EXPECTED_METRIC_FAMILIES = [
    "plantguard_predictions_total",
    "plantguard_confidence_score",
    "plantguard_inference_duration_seconds",
    "plantguard_feature_extraction_seconds",
    "plantguard_classifier_seconds",
    "plantguard_guard_decisions_total",
    "plantguard_leaf_similarity_score",
    "plantguard_model_info",
    "http_requests_total",
]


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type_is_prometheus_format(self, client):
        r = client.get("/metrics")
        ct = r.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_all_custom_metric_families_present(self, client):
        body = client.get("/metrics").text
        for metric_name in EXPECTED_METRIC_FAMILIES:
            assert metric_name in body, (
                f"Expected metric family '{metric_name}' not found in /metrics output"
            )

    def test_prediction_increments_counter(self, client):
        # Read baseline
        body_before = client.get("/metrics").text
        before = _extract_counter(body_before,
                                   'plantguard_predictions_total{prediction="HEALTHY"}')

        # Make a prediction (mock returns HEALTHY = label 0)
        client.post("/predict",
                    files={"file": ("leaf.jpg", _make_leaf_image(), "image/jpeg")})

        # Read after
        body_after = client.get("/metrics").text
        after = _extract_counter(body_after,
                                  'plantguard_predictions_total{prediction="HEALTHY"}')

        assert after == before + 1.0, (
            f"Expected counter to increment by 1 (was {before}, now {after})"
        )

    def test_guard_accepted_counter_increments(self, client):
        body_before = client.get("/metrics").text
        before = _extract_counter(body_before,
                                   'plantguard_guard_decisions_total{decision="accepted"}')

        client.post("/predict",
                    files={"file": ("leaf.jpg", _make_leaf_image(), "image/jpeg")})

        body_after = client.get("/metrics").text
        after = _extract_counter(body_after,
                                  'plantguard_guard_decisions_total{decision="accepted"}')
        assert after == before + 1.0

    def test_rejected_image_increments_rejected_counter(self, client):
        mock_guard = MagicMock()
        mock_guard.check.return_value = (False, 0.21)
        mock_guard.threshold = 0.68

        body_before = client.get("/metrics").text
        before = _extract_counter(
            body_before,
            'plantguard_guard_decisions_total{decision="rejected"}',
            default=0.0,
        )

        with patch("api.main._guard", mock_guard):
            client.post("/predict",
                        files={"file": ("not_a_leaf.jpg",
                                        _make_leaf_image(), "image/jpeg")})

        body_after = client.get("/metrics").text
        after = _extract_counter(
            body_after,
            'plantguard_guard_decisions_total{decision="rejected"}',
            default=0.0,
        )
        assert after == before + 1.0

    def test_model_info_gauge_present(self, client):
        body = client.get("/metrics").text
        assert "plantguard_model_info" in body
        # The Info metric should expose model_type label
        assert "model_type" in body


def _extract_counter(text: str, metric_line: str, default: float = 0.0) -> float:
    """Parse a counter value from a Prometheus text output line."""
    for line in text.splitlines():
        if line.startswith(metric_line):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return default