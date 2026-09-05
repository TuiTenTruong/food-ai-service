"""
Food AI Service - Production Application Entry Point.
Provides unified Detection (YOLO26, RT-DETR, RF-DETR) and RAG Recipe Suggestion services.
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Tránh UnicodeEncodeError trên Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from detectors import (
    get_detector,
    list_available_detectors,
    DETECTOR_REGISTRY,
    BaseDetector
)
from rag import get_rag_service, RAGService
from api import detection_router, rag_router

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("FoodAIService")


class CreateChatSessionRequest(BaseModel):
    title: Optional[str] = None


class SendChatMessageRequest(BaseModel):
    message: str
    recipes: Optional[List[Dict[str, Any]]] = None
    user_pantry: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan:
    1. Read INGREDIENT_MODEL from ENV (default: rtdetr).
    2. Check registry and load selected detector ONCE at startup.
    3. Initialize RAG Food Database and retriever ONCE at startup.
    4. Store instances in app.state for request handlers.
    """
    logger.info("========================================")
    logger.info(" STARTING FOOD AI SERVICE ")
    logger.info("========================================")

    # 1. Detectors Initialization
    model_name = os.getenv("INGREDIENT_MODEL", "rtdetr").lower().strip()
    available_detectors = list_available_detectors()
    logger.info(f"Configured INGREDIENT_MODEL: '{model_name}'")
    logger.info(f"Available Detectors: {available_detectors}")

    if model_name not in DETECTOR_REGISTRY:
        error_msg = (
            f"Unsupported detector: '{model_name}'. "
            f"Available detectors: {', '.join(available_detectors)}"
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

    try:
        active_detector: BaseDetector = get_detector(model_name)
        active_detector.load()
        app.state.detector = active_detector
        logger.info(f" Active detector '{model_name}' loaded successfully into memory.")
    except Exception as exc:
        logger.error(f" Failed to load detector '{model_name}': {exc}", exc_info=True)
        raise RuntimeError(f"Detector loading failed: {exc}")

    # 2. RAG Service Initialization
    try:
        rag_service: RAGService = get_rag_service()
        app.state.rag_service = rag_service
        recipe_count = len(rag_service.food_db.get_all_recipes())
        logger.info(f" RAG Food Database loaded with {recipe_count} recipes.")
    except Exception as exc:
        logger.error(f" Failed to initialize RAG Service: {exc}", exc_info=True)
        raise RuntimeError(f"RAG Service initialization failed: {exc}")

    # 3. Chatbot Service (lazy-loaded on demand when chat endpoints are called)
    app.state.chatbot = None

    # 4. Pre-load Qwen 2.5 on GPU if LLM_PROVIDER=qwen
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if llm_provider == "qwen":
        try:
            logger.info("⚡ Pre-loading Qwen 2.5 - 3B onto GPU memory...")
            from services.qwen_service import get_qwen_service
            qwen_srv = get_qwen_service()
            qwen_srv.load()
            app.state.qwen_service = qwen_srv
            logger.info(" Qwen 2.5 - 3B loaded and ready on GPU!")
        except Exception as exc:
            logger.warning(f"Could not pre-load Qwen at startup (will retry on request): {exc}")

    logger.info(" Food AI Service startup complete and ready for requests.")

    yield

    # Shutdown
    logger.info(" Shutting down Food AI Service...")
    if hasattr(app.state, "detector"):
        del app.state.detector
    if hasattr(app.state, "rag_service"):
        del app.state.rag_service
    if hasattr(app.state, "qwen_service"):
        del app.state.qwen_service
    logger.info(" Clean shutdown complete.")


app = FastAPI(
    title="Food AI Service",
    description="Modular AI Service for Ingredient Detection (YOLO26/RT-DETR/RF-DETR) and RAG Recipe Suggestion",
    version="3.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(detection_router)
app.include_router(rag_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint with detector and RAG status."""
    active_det = getattr(app.state, "detector", None)
    active_name = active_det.model_name if active_det else "uninitialized"
    is_loaded = active_det.is_loaded if active_det else False

    rag_srv = getattr(app.state, "rag_service", None)
    recipes_count = len(rag_srv.food_db.get_all_recipes()) if rag_srv else 0

    try:
        from services.llm_config import get_llm_settings, llm_settings_summary
        llm_info = llm_settings_summary(get_llm_settings())
    except Exception:
        llm_info = {"provider": os.getenv("LLM_PROVIDER", "gemini"), "model": os.getenv("LLM_MODEL", "gemini-2.5-flash")}

    return {
        "status": "healthy",
        "service": "Food AI Service",
        "version": "3.0.0",
        "active_detector": active_name,
        "detector_loaded": is_loaded,
        "available_detectors": list_available_detectors(),
        "rag_recipes_count": recipes_count,
        "llm": llm_info
    }


def _get_chatbot_instance():
    chatbot = getattr(app.state, "chatbot", None)
    if chatbot is None:
        try:
            from services.chatbot_service import get_chatbot_service
            app.state.chatbot = get_chatbot_service()
            chatbot = app.state.chatbot
        except Exception as e:
            logger.warning(f"Chatbot service unavailable: {e}")
            return None
    return chatbot


# ============== CHAT SESSION APIS (Backward Compatible) ==============

@app.post("/api/ai/chat/sessions", tags=["Chat"])
async def create_chat_session(request: CreateChatSessionRequest):
    try:
        chatbot = _get_chatbot_instance()
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")
        session = chatbot.create_session(title=request.title)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": session,
                "message": "Tạo chat session thành công"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Create session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/chat/sessions/{session_id}/messages", tags=["Chat"])
async def send_chat_message(session_id: str, request: SendChatMessageRequest):
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="message không được để trống")

        chatbot = _get_chatbot_instance()
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        result = chatbot.chat(
            session_id=session_id,
            message=request.message.strip(),
            recipes=request.recipes,
            user_pantry=request.user_pantry
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": result,
                "message": "Gửi tin nhắn thành công"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Send message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/chat/sessions/{session_id}/messages/stream", tags=["Chat"])
async def stream_chat_message(session_id: str, request: SendChatMessageRequest):
    """
    [Giải pháp Tối ưu 2] Server-Sent Events (SSE) chat stream.
    Truyền từng token thời gian thực về client.
    Client nhận: `data: {"chunk": "..."}\n\n`
    Kết thúc với: `data: [DONE]\n\n`
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="message không được để trống")

        chatbot = _get_chatbot_instance()
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service is not available")

        import json

        def sse_event_generator():
            try:
                for token_chunk in chatbot.stream_chat(
                    session_id=session_id,
                    message=request.message.strip(),
                    recipes=request.recipes,
                    user_pantry=request.user_pantry
                ):
                    if token_chunk:
                        payload = json.dumps({"chunk": token_chunk}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"data: {err_payload}\n\n"

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Stream message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "run_ai:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1
    )