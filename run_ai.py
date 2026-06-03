import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Khởi tạo các Services
from services.vision_service import VisionService
from services.chat_service import CookingLangChainService
from services.chatbot_service import get_chatbot_service
from services.rag_controller import get_rag_controller
from services.rag_schemas import RecipeSuggestRequest, RecipeSuggestResponse

load_dotenv()

# Biến toàn cục để lưu models
models = {}


class CreateChatSessionRequest(BaseModel):
    title: Optional[str] = None


class SendChatMessageRequest(BaseModel):
    message: str
    recipes: Optional[List[Dict[str, Any]]] = None
    user_pantry: Optional[List[str]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models 1 lần khi khởi động server
    print("====================================")
    print(" KHỞI ĐỘNG FOOD AI SERVICE ")
    print("====================================")
    
    models["vision"] = VisionService()
    models["ai_assistant"] = CookingLangChainService()
    models["chatbot"] = get_chatbot_service()
    models["rag_controller"] = get_rag_controller()
    
    print(" ✓ Models đã load xong và sẵn sàng!")
    
    yield
    
    # Shutdown: Dọn dẹp nếu cần
    models.clear()
    print(" Server đã tắt!")

app = FastAPI(
    title="Food AI Service",
    description="AI Service for food ingredient detection and recipe suggestion",
    version="2.0.0",
    lifespan=lifespan
)

# CORS - Cho phép Backend khác port gọi qua
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health_check():
    return {"status": "AI Service is active and running!"}


# ============== CHAT SESSION API ==============

@app.post('/api/ai/chat/sessions')
async def create_chat_session(request: CreateChatSessionRequest):
    try:
        session = models["chatbot"].create_session(title=request.title)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": session,
                "message": "Tạo chat session thành công"
            }
        )
    except Exception as e:
        print(f" [API] Create session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/ai/chat/sessions/{session_id}/messages')
async def send_chat_message(session_id: str, request: SendChatMessageRequest):
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="message không được để trống")

        result = models["chatbot"].chat(
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
        print(f" [API] Send message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/ai/analyze-image')
async def analyze_image(
    image: UploadFile = File(...),
    recipe_chunks: str = Form(default=None)
):
    try:
        # Đọc bytes từ file upload
        image_bytes = await image.read()
        
        # Bước 1: Computer Vision (YOLO + ResNet)
        detected_items = models["vision"].predict_image(image_bytes)
        
        if not detected_items:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "ingredients": [],
                        "ai_suggestion": None
                    },
                    "message": "Không nhận diện được nguyên liệu nào."
                }
            )

        # Bước 2: AI Reasoning (LangChain + OpenAI)
        ingredient_names = [item['name'] for item in detected_items]

        parsed_recipe_chunks = []
        if recipe_chunks:
            try:
                parsed_recipe_chunks = json.loads(recipe_chunks)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="recipe_chunks phải là JSON hợp lệ (mảng chuỗi hoặc object)."
                )

        suggestion = models["ai_assistant"].get_suggestion(
            ingredient_names,
            parsed_recipe_chunks
        )

        # Bước 3: Trả về kết quả JSON tổng hợp
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "ingredients": detected_items,
                    "ai_suggestion": suggestion
                },
                "message": "Phân tích hình ảnh và tư vấn thành công!"
            }
        )

    except Exception as e:
        print(f"Lỗi Server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== RAG RECIPE SUGGESTION API ==============

@app.post('/api/ai/recipe-suggest', response_model=RecipeSuggestResponse)
@app.post('/internal/rag/recipe-suggest', response_model=RecipeSuggestResponse)
async def rag_recipe_suggest(request: RecipeSuggestRequest):
    """
    Internal API: Gợi ý công thức món ăn bằng RAG
    
    Flow:
    1. Nhận danh sách nguyên liệu user có + recipes từ backend
    2. Build embeddings và index vector store
    3. Retrieve top K recipes phù hợp
    4. Dùng LLM sinh gợi ý cuối cùng
    
    Request body:
    {
        "user_ingredients": ["trứng", "cà chua", "hành lá"],
        "recipes": [
            {
                "id": "1",
                "name": "Trứng chiên cà chua",
                "description": "...",
                "steps": "...",
                "ingredients": [{"name": "trứng", "amount": "2 quả"}]
            }
        ],
        "top_k": 5
    }
    """
    try:
        print(f"\n{'='*60}")
        print(" [API] POST /internal/rag/recipe-suggest")
        print(f" - Ingredients: {request.user_ingredients}")
        print(f" - Recipes count: {len(request.recipes)}")
        print(f"{'='*60}\n")
        
        result = models["rag_controller"].suggest_recipe(request)
        
        return result
    
    except Exception as e:
        print(f" [API] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    # Đọc port từ file .env, mặc định là 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "run_ai:app",
        host='0.0.0.0',
        port=port,
        reload=False,  # Tắt reload khi chạy mô hình nặng
        workers=1  # 1 worker để tránh load model nhiều lần
    )