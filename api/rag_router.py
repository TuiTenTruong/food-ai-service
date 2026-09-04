"""
RAG Recipe Suggestion API Router.
Provides endpoints for retrieving and suggesting recipes from the food database:
- POST /api/v1/recipes/suggest (primary production endpoint)
- POST /api/ai/recipe-suggest (backward-compatible endpoint for be_nckh)
- POST /internal/rag/recipe-suggest (legacy internal endpoint)
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rag import get_rag_service, RAGService
from services.rag_schemas import RecipeInput, AlternativeRecipe, RecipeSuggestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class StandardRecipeSuggestRequest(BaseModel):
    """Request schema for Recipe Suggestion."""
    user_ingredients: List[str] = Field(
        ...,
        description="Danh sách nguyên liệu người dùng có",
        json_schema_extra={"example": ["trứng", "cà chua"]}
    )
    recipes: Optional[List[RecipeInput]] = Field(
        default=None,
        description="Tùy chọn danh sách recipes. Nếu không truyền, hệ thống sẽ tự động tra cứu từ Food Database 96 món."
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Số lượng recipes lấy tối đa"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tùy chọn lọc (độ khó, thời gian nấu...)"
    )
    query: Optional[str] = Field(
        default=None,
        description="Câu hỏi hoặc yêu cầu thêm của người dùng"
    )


def _get_rag_service(request: Request) -> RAGService:
    """Retrieve the RAG service instance from app state or singleton."""
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        service = get_rag_service()
    return service


@router.post(
    "/api/v1/recipes/suggest",
    response_model=RecipeSuggestResponse,
    tags=["RAG Recipes"]
)
@router.post(
    "/api/ai/recipe-suggest",
    response_model=RecipeSuggestResponse,
    tags=["RAG Recipes"],
    deprecated=True
)
@router.post(
    "/internal/rag/recipe-suggest",
    response_model=RecipeSuggestResponse,
    tags=["RAG Recipes"],
    deprecated=True
)
async def suggest_recipes(
    request_data: StandardRecipeSuggestRequest,
    request: Request
):
    """
    RAG Recipe Suggestion Endpoint.
    Gợi ý công thức món ăn từ cơ sở dữ liệu dựa trên nguyên liệu và yêu cầu người dùng.
    """
    rag_service = _get_rag_service(request)

    try:
        # If caller provides dynamic recipes, we can support temporary indexing or direct matching
        # However, default and recommended behavior is retrieving from the curated Food Database
        result = rag_service.suggest_recipe(
            user_ingredients=request_data.user_ingredients,
            preferences=request_data.preferences,
            top_k=request_data.top_k or 5,
            query=request_data.query
        )
        return result
    except Exception as exc:
        logger.error(f"[RAG API] Error during recipe suggestion: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi gợi ý công thức RAG: {str(exc)}"
        )
