"""
Unit tests for the Plant Leaf Health Classifier API.
Run: pytest tests/ -v
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


def _make_image(color=(80, 140, 60), size=(224, 224)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])  # DISEASED
    mock_model.predict_proba.return_value = np.array([[0.08, 0.92]])

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    mock_guard = MagicMock()
    mock_guard.check.return_value = (True, 0.81)
    mock_guard.threshold = 0.68

    mock_extract = MagicMock(return_value=np.zeros((1, 1280), dtype=np.float32))

    with (
        patch("api.main.get_model", return_value=mock_model),
        patch("api.main.get_scaler", return_value=mock_scaler),
        patch("api.main.LeafGuard", return_value=mock_guard),
        patch("api.main._guard", mock_guard),
        patch("api.main._extract_features", mock_extract),
        patch("api.main._embedder", MagicMock()),
        patch("api.main._transform", MagicMock()),
    ):
        from fastapi.testclient import TestClient

        from api.main import app

        yield TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestClassesEndpoint:
    def test_returns_both_classes(self, client):
        data = client.get("/classes").json()
        assert "HEALTHY" in data["classes"]
        assert "DISEASED" in data["classes"]

    def test_label_map(self, client):
        data = client.get("/classes").json()
        assert data["label_map"]["HEALTHY"] == 0
        assert data["label_map"]["DISEASED"] == 1


class TestPredictEndpoint:
    def test_valid_leaf_returns_prediction(self, client):
        r = client.post(
            "/predict", files={"file": ("leaf.jpg", _make_image(), "image/jpeg")}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] in ("HEALTHY", "DISEASED")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["is_leaf"] is True
        assert "healthy_prob" in data
        assert "diseased_prob" in data

    def test_unsupported_type_returns_415(self, client):
        r = client.post(
            "/predict", files={"file": ("doc.pdf", b"%PDF", "application/pdf")}
        )
        assert r.status_code == 415

    def test_non_leaf_rejected_with_422(self, client):
        mock_guard = MagicMock()
        mock_guard.check.return_value = (False, 0.22)
        mock_guard.threshold = 0.68
        with patch("api.main._guard", mock_guard):
            r = client.post(
                "/predict", files={"file": ("dog.jpg", _make_image(), "image/jpeg")}
            )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["rejected"] is True
        assert "leaf_similarity" in detail

    def test_corrupt_image_returns_400(self, client):
        r = client.post(
            "/predict", files={"file": ("bad.jpg", b"notanimage", "image/jpeg")}
        )
        assert r.status_code == 400
