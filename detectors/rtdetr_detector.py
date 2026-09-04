"""
RT-DETR (Real-Time DEtection TRansformer) Detector.
Uses Ultralytics RTDETR implementation with custom weights for ingredient detection.
"""

import os
import logging
from typing import List, Dict, Any
from PIL import Image
import torch

from .base import BaseDetector
from .registry import register_detector
from .mapping import translate_label

logger = logging.getLogger(__name__)


def _configure_trusted_torch():
    """Ensure torch loads weights without strict weights_only blocking in PyTorch 2.6+"""
    if getattr(torch.load, "_food_ai_patched", False):
        return
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)
    patched_load._food_ai_patched = True
    torch.load = patched_load


@register_detector("rtdetr")
class RTDETRDetector(BaseDetector):
    """
    RT-DETR Detector for 32 Vietnamese Food Ingredients.
    """

    def __init__(self, model_name: str = "rtdetr", model_path: str = None, device: str = None):
        super().__init__(model_name, model_path, device)
        if not self.model_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.getenv(
                "RTDETR_MODEL_PATH",
                os.path.join(base_dir, "models_weights", "rtdetr_best.pt")
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.names = {}

    def load(self) -> None:
        """Load RT-DETR weights using Ultralytics."""
        _configure_trusted_torch()
        from ultralytics import RTDETR

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RT-DETR weight file not found at: {self.model_path}")

        logger.info(f"Loading RT-DETR model from {self.model_path} onto {self.device}...")
        self.model = RTDETR(self.model_path)
        self.names = getattr(self.model, "names", {})
        self.is_loaded = True
        logger.info(f"RT-DETR loaded successfully with {len(self.names)} classes.")

    def predict_raw(self, image: Image.Image, conf_threshold: float = 0.25) -> Any:
        """Execute RT-DETR inference on image."""
        results = self.model(image, conf=conf_threshold, device=self.device, verbose=False)
        return results[0] if results else None

    def postprocess(self, raw_output: Any, image_size: tuple, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Normalize bounding boxes to [x1, y1, x2, y2] and confidence to 0.0 - 1.0."""
        detections = []
        if raw_output is None or not hasattr(raw_output, "boxes") or raw_output.boxes is None:
            return detections

        boxes = raw_output.boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_threshold:
                continue

            cls_id = int(boxes.cls[i].item())
            label_en = self.names.get(cls_id, str(cls_id))
            label_vi = translate_label(label_en)

            # xyxy format in pixel coordinates
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

            detections.append({
                "label": label_vi,
                "label_en": label_en,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
