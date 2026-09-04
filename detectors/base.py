"""
Base interface for all Ingredient Detectors.
Defines standard lifecycle: load, preprocess, predict, postprocess.
Ensures uniform output format across YOLO26, RT-DETR, RF-DETR and future detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import io
import time


class BaseDetector(ABC):
    """
    Abstract base detector class.
    All ingredient detectors must inherit from this class.
    """

    def __init__(self, model_name: str, model_path: str = None, device: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into memory and target device.
        Must be called once during startup.
        """
        pass

    def preprocess(self, image_input: Union[bytes, Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert input bytes or ndarray to PIL RGB image.
        Can be overridden by subclass for detector-specific preprocessing.
        """
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # If BGR (OpenCV) convert to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                import cv2
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
            else:
                image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        return image

    @abstractmethod
    def predict_raw(self, image: Image.Image, conf_threshold: float = 0.25) -> Any:
        """
        Execute raw model inference on the preprocessed image.
        Returns raw predictions specific to the detector framework.
        """
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, image_size: tuple, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Convert raw predictions to standardized detections:
        [
            {
                "label": str,          # English label or Vietnamese translation
                "confidence": float,   # Normalized 0.0 to 1.0
                "bbox": [x1, y1, x2, y2] # Integer pixel coordinates
            }
        ]
        """
        pass

    def predict(self, image_input: Union[bytes, Image.Image, np.ndarray], conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        End-to-end inference pipeline:
        1. Preprocess image
        2. Run inference with timer
        3. Postprocess and format result
        """
        if not self.is_loaded or self.model is None:
            self.load()

        start_time = time.perf_counter()
        image = self.preprocess(image_input)
        raw_output = self.predict_raw(image, conf_threshold=conf_threshold)
        detections = self.postprocess(raw_output, image.size, conf_threshold=conf_threshold)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model": self.model_name,
            "detections": detections,
            "inference_time_ms": elapsed_ms
        }
