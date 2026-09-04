"""
Detection API Router.
Provides standardized detection endpoints according to project specifications:
- POST /api/v1/ingredients/detect (active model)
- GET /api/v1/models (active & available models)
- POST /api/v1/models/{model_name}/detect (model-specific)
- POST /api/ai/analyze-image (legacy backward-compatible endpoint for be_nckh)
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from detectors import (
    get_detector,
    list_available_detectors,
    DETECTOR_REGISTRY,
    BaseDetector,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class DetectionItem(BaseModel):
    label: str = Field(..., description="Tên nguyên liệu tiếng Việt chuẩn hóa")
    label_en: Optional[str] = Field(None, description="Tên nguyên liệu tiếng Anh gốc")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tự tin chuẩn hóa 0.0 - 1.0")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    model: str = Field(..., description="Tên model đã thực hiện nhận diện")
    detections: List[DetectionItem] = Field(..., description="Danh sách nguyên liệu phát hiện được")
    inference_time_ms: float = Field(..., description="Thời gian suy luận tính bằng mili-giây")


class ModelsInfoResponse(BaseModel):
    active_model: str
    available_models: List[str]
    loaded: bool


def _get_active_detector(request: Request) -> BaseDetector:
    """Retrieve the pre-loaded active detector instance from app state."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Active detector has not been initialized. Service is starting up or configuration is invalid."
        )
    return detector


@router.get("/api/v1/models", response_model=ModelsInfoResponse, tags=["Detectors"])
async def get_models_info(request: Request):
    """
    Get information about currently active detector and all available detectors in the registry.
    """
    try:
        active_detector = _get_active_detector(request)
        active_name = active_detector.model_name
        is_loaded = active_detector.is_loaded
    except Exception:
        active_name = "none"
        is_loaded = False

    return {
        "active_model": active_name,
        "available_models": list_available_detectors(),
        "loaded": is_loaded
    }


@router.post(
    "/api/v1/ingredients/detect",
    response_model=DetectionResponse,
    tags=["Detectors"]
)
async def detect_ingredients(
    request: Request,
    image: UploadFile = File(..., description="Tệp ảnh nguyên liệu (JPEG, PNG, WebP)"),
    confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Ngưỡng confidence tùy chọn")
):
    """
    Standard Production Detection API.
    Calls the active detector loaded at service startup via INGREDIENT_MODEL ENV.
    Returns detections with unified bounding box [x1, y1, x2, y2] and confidence [0.0, 1.0].
    """
    detector = _get_active_detector(request)

    # Read image content
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {str(e)}")

    # Use specified confidence threshold or default 0.25
    conf_thresh = confidence if confidence is not None else 0.25

    try:
        result = detector.predict(image_bytes, conf_threshold=conf_thresh)
        return result
    except Exception as exc:
        logger.error(f"Detection inference error on model {detector.model_name}: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")


@router.post(
    "/api/v1/models/{model_name}/detect",
    response_model=DetectionResponse,
    tags=["Detectors"]
)
async def detect_with_specific_model(
    model_name: str,
    image: UploadFile = File(..., description="Tệp ảnh nguyên liệu"),
    confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Ngưỡng confidence tùy chọn")
):
    """
    Model-specific detection endpoint.
    Allows testing a specific detector (e.g. yolo26, rtdetr, rfdetr) on demand.
    """
    model_key = model_name.lower().strip()
    available = list_available_detectors()
    if model_key not in DETECTOR_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported detector: '{model_name}'. Available detectors: {', '.join(available)}"
        )

    try:
        # Get detector (cached or new instance for specific model)
        detector = get_detector(model_key)
        if not detector.is_loaded:
            detector.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model '{model_name}': {str(e)}")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        conf_thresh = confidence if confidence is not None else 0.25
        return detector.predict(image_bytes, conf_threshold=conf_thresh)
    except Exception as exc:
        logger.error(f"Inference error with specific model {model_name}: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")


@router.post("/api/ai/analyze-image", deprecated=True, tags=["Legacy"])
async def analyze_image_legacy(
    request: Request,
    image: UploadFile = File(...),
    recipe_chunks: Optional[str] = Form(default=None)
):
    """
    Legacy endpoint maintained for backward compatibility with existing be_nckh Flask backend.
    Uses active detector and formats response to the legacy schema expected by be_nckh.
    """
    detector = _get_active_detector(request)

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {str(e)}")

    try:
        pred_result = detector.predict(image_bytes, conf_threshold=0.25)
        raw_detections = pred_result.get("detections", [])

        # Format detections for be_nckh
        legacy_ingredients = []
        for det in raw_detections:
            legacy_ingredients.append({
                "name": det["label"],
                "confidence": det["confidence"]
            })

        # Remove duplicates
        unique_ingredients = {item["name"]: item for item in legacy_ingredients}.values()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "ingredients": list(unique_ingredients),
                    "ai_suggestion": None
                },
                "message": "Phân tích hình ảnh thành công!"
            }
        )
    except Exception as e:
        logger.error(f"Legacy analyze-image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
