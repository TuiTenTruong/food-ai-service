"""
Detector Registry & Factory.
Implements Registry Pattern + Strategy Pattern to dynamically register
and instantiate ingredient detectors.
"""

from typing import Dict, Type, List, Optional
import os
import logging
from .base import BaseDetector

logger = logging.getLogger(__name__)

# Global registry mapping model string key to detector class
DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_detector(name: str):
    """
    Decorator to register a detector class under a given name.
    Example:
        @register_detector("rtdetr")
        class RTDETRDetector(BaseDetector):
            ...
    """
    def decorator(cls: Type[BaseDetector]):
        key = name.lower().strip()
        if key in DETECTOR_REGISTRY:
            logger.warning(f"Overwriting detector registration for key: '{key}'")
        DETECTOR_REGISTRY[key] = cls
        return cls
    return decorator


def list_available_detectors() -> List[str]:
    """Return sorted list of all registered detector names."""
    return sorted(list(DETECTOR_REGISTRY.keys()))


def get_detector(name: Optional[str] = None, **kwargs) -> BaseDetector:
    """
    Factory function to retrieve and instantiate a detector by name.
    If name is None, reads INGREDIENT_MODEL from environment variables (default 'rtdetr').
    Raises ValueError with descriptive error message if detector is not supported.
    """
    # Ensure standard detectors are imported and registered
    _ensure_detectors_loaded()

    if not name:
        name = os.getenv("INGREDIENT_MODEL", "rtdetr")

    key = name.lower().strip()

    if key not in DETECTOR_REGISTRY:
        available = ", ".join(list_available_detectors())
        raise ValueError(
            f"Unsupported detector: '{name}'. Available detectors: {available}"
        )

    detector_cls = DETECTOR_REGISTRY[key]
    return detector_cls(model_name=key, **kwargs)


def _ensure_detectors_loaded():
    """
    Lazily import default detector implementations to populate registry.
    """
    try:
        from . import yolo_detector
    except ImportError as e:
        logger.debug(f"YOLO detector module import error: {e}")

    try:
        from . import rtdetr_detector
    except ImportError as e:
        logger.debug(f"RT-DETR detector module import error: {e}")

    try:
        from . import rfdetr_detector
    except ImportError as e:
        logger.debug(f"RF-DETR detector module import error: {e}")
