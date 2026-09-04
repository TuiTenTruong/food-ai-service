"""
Detectors module for Food AI Service.
Provides BaseDetector interface, dynamic Registry, and standard detectors
for YOLO26, RT-DETR, RF-DETR and future models.
"""

from .base import BaseDetector
from .registry import (
    register_detector,
    get_detector,
    list_available_detectors,
    DETECTOR_REGISTRY,
)
from .mapping import translate_label, INGREDIENT_CLASSES_EN, EN_TO_VI_MAPPING
from .yolo_detector import YOLO26Detector
from .rtdetr_detector import RTDETRDetector
from .rfdetr_detector import RFDETRDetector

__all__ = [
    "BaseDetector",
    "register_detector",
    "get_detector",
    "list_available_detectors",
    "DETECTOR_REGISTRY",
    "translate_label",
    "INGREDIENT_CLASSES_EN",
    "EN_TO_VI_MAPPING",
    "YOLO26Detector",
    "RTDETRDetector",
    "RFDETRDetector",
]
