"""
API package for Food AI Service.
"""

from .detection_router import router as detection_router
from .rag_router import router as rag_router

__all__ = ["detection_router", "rag_router"]
