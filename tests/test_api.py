"""
FastAPI HTTP Endpoint Integration Tests.
Tests criteria:
- Health API (/health)
- Models API (/api/v1/models)
- Standard Production Detection API (/api/v1/ingredients/detect)
- Specific Model Detection API (/api/v1/models/{model}/detect)
- Invalid model error handling
- Legacy API compatibility (/api/ai/analyze-image)
- RAG Recipe Suggestion API (/api/v1/recipes/suggest & /api/ai/recipe-suggest)
- Invalid/empty image validation
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from run_ai import app
from detectors import get_detector


def create_test_image_file() -> io.BytesIO:
    """Helper to create dummy JPEG image buffer."""
    img = Image.new("RGB", (320, 320), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def client():
    """Setup HTTP client using TestClient with initialized app state."""
    from starlette.testclient import TestClient
    from detectors import get_detector
    from rag import get_rag_service

    det = get_detector("rtdetr")
    if not det.is_loaded:
        det.load()
    app.state.detector = det
    app.state.rag_service = get_rag_service()

    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Integration test suite for Food AI Service API."""

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_detector" in data
        assert "available_detectors" in data
        assert "yolo26" in data["available_detectors"]
        assert "rtdetr" in data["available_detectors"]
        assert "rfdetr" in data["available_detectors"]
        assert data["rag_recipes_count"] == 96

    def test_models_info_endpoint(self, client):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "active_model" in data
        assert "available_models" in data
        assert isinstance(data["available_models"], list)

    def test_standard_detection_endpoint(self, client):
        img_buf = create_test_image_file()
        response = client.post(
            "/api/v1/ingredients/detect",
            files={"image": ("test.jpg", img_buf, "image/jpeg")},
            params={"confidence": 0.2}
        )
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "detections" in data
        assert "inference_time_ms" in data
        assert isinstance(data["detections"], list)

        # If detections exist, check schema
        for det in data["detections"]:
            assert "label" in det
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4

    def test_specific_model_detect_endpoint(self, client):
        img_buf = create_test_image_file()
        response = client.post(
            "/api/v1/models/yolo26/detect",
            files={"image": ("test.jpg", img_buf, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "yolo26"

    def test_invalid_model_detect_endpoint(self, client):
        img_buf = create_test_image_file()
        response = client.post(
            "/api/v1/models/non_existent_detector/detect",
            files={"image": ("test.jpg", img_buf, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Unsupported detector" in response.json()["detail"]

    def test_empty_image_error_handling(self, client):
        response = client.post(
            "/api/v1/ingredients/detect",
            files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
        )
        assert response.status_code == 400

    def test_legacy_analyze_image_endpoint(self, client):
        img_buf = create_test_image_file()
        response = client.post(
            "/api/ai/analyze-image",
            files={"image": ("test.jpg", img_buf, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "ingredients" in data["data"]

    def test_rag_recipe_suggest_endpoint(self, client):
        payload = {
            "user_ingredients": ["trứng", "cà chua"],
            "top_k": 3
        }
        response = client.post("/api/v1/recipes/suggest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "best_recipe" in data
        assert data["best_recipe"] is not None

    def test_legacy_rag_recipe_suggest_endpoint(self, client):
        payload = {
            "user_ingredients": ["thịt gà"],
            "top_k": 3
        }
        response = client.post("/api/ai/recipe-suggest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "best_recipe" in data
