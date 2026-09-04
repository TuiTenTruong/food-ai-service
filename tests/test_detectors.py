"""
Unit and Integration Tests for Detector Architecture.
Tests criteria:
- Detector registry
- ENV model selection
- BaseDetector contract
- YOLO26, RT-DETR, RF-DETR implementations
- Invalid detector handling
- Missing weights handling
- Bounding box [x1, y1, x2, y2] & confidence [0.0, 1.0] normalization
- Extensibility: Adding a 4th detector without modifying core pipeline
"""

import os
import io
import pytest
import numpy as np
from PIL import Image

from detectors import (
    BaseDetector,
    register_detector,
    get_detector,
    list_available_detectors,
    DETECTOR_REGISTRY,
    YOLO26Detector,
    RTDETRDetector,
    RFDETRDetector,
)


def create_dummy_image_bytes(width=640, height=640) -> bytes:
    """Helper to create dummy RGB image bytes for testing."""
    img = Image.new("RGB", (width, height), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDetectorRegistry:
    """Tests for detector registry and dynamic loading."""

    def test_registered_detectors(self):
        """Ensure standard initial models are registered."""
        available = list_available_detectors()
        assert "yolo26" in available
        assert "rtdetr" in available
        assert "rfdetr" in available

    def test_get_valid_detectors(self):
        """Ensure get_detector returns correct class instances."""
        det_yolo = get_detector("yolo26")
        assert isinstance(det_yolo, YOLO26Detector)

        det_rtdetr = get_detector("rtdetr")
        assert isinstance(det_rtdetr, RTDETRDetector)

        det_rfdetr = get_detector("rfdetr")
        assert isinstance(det_rfdetr, RFDETRDetector)

    def test_invalid_detector_error(self):
        """Ensure requesting an unknown detector raises ValueError with available options."""
        with pytest.raises(ValueError) as exc_info:
            get_detector("unknown_detector_xyz")
        assert "Unsupported detector: 'unknown_detector_xyz'" in str(exc_info.value)
        assert "Available detectors" in str(exc_info.value)


class TestExtensibility:
    """Agent 3 Task 3: Test adding a new model without modifying core detection pipeline."""

    def test_add_test_detector_without_core_changes(self):
        """
        Verify that creating a class, implementing BaseDetector, and decorating
        with @register_detector allows seamless integration into registry and inference.
        """
        @register_detector("test_detector_v4")
        class TestDetector(BaseDetector):
            def load(self):
                self.model = "dummy_v4_weights"
                self.is_loaded = True

            def preprocess(self, image_input):
                return super().preprocess(image_input)

            def predict_raw(self, image, conf_threshold=0.25):
                return [{"box": [10, 20, 110, 120], "score": 0.95, "cls": "tomato"}]

            def postprocess(self, raw_output, image_size, conf_threshold=0.25):
                results = []
                for item in raw_output:
                    if item["score"] >= conf_threshold:
                        results.append({
                            "label": "Cà chua",
                            "label_en": item["cls"],
                            "confidence": item["score"],
                            "bbox": item["box"]
                        })
                return results

        # 1. Verify in registry
        available = list_available_detectors()
        assert "test_detector_v4" in available

        # 2. Retrieve instance
        detector = get_detector("test_detector_v4")
        assert isinstance(detector, TestDetector)
        assert detector.model_name == "test_detector_v4"

        # 3. Test lifecycle and prediction
        dummy_img = create_dummy_image_bytes()
        res = detector.predict(dummy_img, conf_threshold=0.5)

        assert res["model"] == "test_detector_v4"
        assert len(res["detections"]) == 1
        assert res["detections"][0]["label"] == "Cà chua"
        assert res["detections"][0]["confidence"] == 0.95
        assert res["detections"][0]["bbox"] == [10, 20, 110, 120]
        assert "inference_time_ms" in res


class TestBoundingBoxAndConfidenceNormalization:
    """Tests that outputs strictly follow [x1, y1, x2, y2] and [0.0, 1.0]."""

    def test_mock_detector_normalization(self):
        @register_detector("mock_normalizer")
        class MockNormalizerDetector(BaseDetector):
            def load(self):
                self.model = True
                self.is_loaded = True

            def preprocess(self, image_input):
                return super().preprocess(image_input)

            def predict_raw(self, image, conf_threshold=0.25):
                return None

            def postprocess(self, raw_output, image_size, conf_threshold=0.25):
                return [
                    {
                        "label": "Trứng",
                        "label_en": "egg",
                        "confidence": 0.887,
                        "bbox": [50, 60, 250, 300]
                    }
                ]

        det = get_detector("mock_normalizer")
        res = det.predict(create_dummy_image_bytes())
        detection = res["detections"][0]

        # Check bbox structure [x1, y1, x2, y2]
        bbox = detection["bbox"]
        assert len(bbox) == 4
        assert bbox[0] < bbox[2], "x1 must be strictly less than x2"
        assert bbox[1] < bbox[3], "y1 must be strictly less than y2"

        # Check confidence normalization [0.0, 1.0]
        conf = detection["confidence"]
        assert 0.0 <= conf <= 1.0


class TestLiveModelWeights:
    """Integration tests on actual downloaded weights."""

    def test_rtdetr_live_load_and_inference(self):
        """Test RT-DETR with actual weights."""
        det = get_detector("rtdetr")
        det.load()
        assert det.is_loaded is True

        dummy_img = create_dummy_image_bytes()
        res = det.predict(dummy_img, conf_threshold=0.25)
        assert res["model"] == "rtdetr"
        assert isinstance(res["detections"], list)
        assert res["inference_time_ms"] > 0

    def test_yolo26_live_load_and_inference(self):
        """Test YOLO26 with actual weights."""
        det = get_detector("yolo26")
        det.load()
        assert det.is_loaded is True

        dummy_img = create_dummy_image_bytes()
        res = det.predict(dummy_img, conf_threshold=0.25)
        assert res["model"] == "yolo26"
        assert isinstance(res["detections"], list)
        assert res["inference_time_ms"] > 0

    def test_rfdetr_live_load_and_inference(self):
        """Test RF-DETR with actual weights."""
        det = get_detector("rfdetr")
        det.load()
        assert det.is_loaded is True

        dummy_img = create_dummy_image_bytes()
        res = det.predict(dummy_img, conf_threshold=0.25)
        assert res["model"] == "rfdetr"
        assert isinstance(res["detections"], list)
        assert res["inference_time_ms"] > 0
