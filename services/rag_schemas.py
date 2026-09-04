"""
Pydantic Schemas cho RAG Recipe Suggestion API
Định nghĩa input/output contract giữa backend chính và ai-service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============== INPUT SCHEMAS ==============

class IngredientItem(BaseModel):
    """Nguyên liệu của một recipe"""
    name: str = Field(..., description="Tên nguyên liệu")
    quantity: str = Field(default="", description="Số lượng, ví dụ: '2', '100'")
    unit: str = Field(default="", description="Đơn vị, ví dụ: 'quả', 'gram'")


class RecipeInput(BaseModel):
    """Thông tin một recipe từ backend"""
    id: str = Field(..., description="ID của recipe")
    name: str = Field(..., description="Tên món ăn")
    description: Optional[str] = Field(None, description="Mô tả món ăn")
    steps: Optional[str] = Field(None, description="Các bước nấu")
    ingredients: List[IngredientItem] = Field(default_factory=list, description="Danh sách nguyên liệu")
    image_url: Optional[str] = Field(None, description="URL ảnh món ăn")
    cook_time_minutes: Optional[int] = Field(None, description="Thời gian nấu (phút)")
    difficulty: Optional[str] = Field(None, description="Độ khó")
    servings: Optional[int] = Field(None, description="Số khẩu phần")


class RecipeSuggestRequest(BaseModel):
    """
    Request từ backend chính
    Chứa danh sách nguyên liệu user có và tất cả recipes để search
    """
    user_ingredients: List[str] = Field(
        ..., 
        description="Danh sách nguyên liệu người dùng hiện có",
        examples=[["trứng", "cà chua", "hành lá"]]
    )
    recipes: List[RecipeInput] = Field(
        ..., 
        description="Danh sách recipes từ database"
    )
    top_k: int = Field(
        default=5, 
        description="Số lượng recipes tối đa để retrieve",
        ge=1,
        le=20
    )
    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="Tùy chọn lọc của người dùng (độ khó, thời gian nấu, chế độ ăn...)"
    )



# ============== OUTPUT SCHEMAS ==============

class AlternativeRecipe(BaseModel):
    """Công thức thay thế"""
    id: str
    name: str
    matched_count: int = Field(..., description="Số nguyên liệu khớp")
    missing_count: int = Field(..., description="Số nguyên liệu thiếu")
    image_url: Optional[str] = Field(None, description="URL ảnh món ăn")
    cook_time_minutes: Optional[int] = Field(None, description="Thời gian nấu (phút)")
    difficulty: Optional[str] = Field(None, description="Độ khó")
    description: Optional[str] = Field(None, description="Mô tả ngắn")


class RecipeSuggestResponse(BaseModel):
    """
    Response trả về cho backend
    Chứa công thức được chọn và các thông tin liên quan
    """
    best_recipe: Optional[str] = Field(None, description="Tên món phù hợp nhất")
    recipe_id: Optional[str] = Field(None, description="ID của món được chọn")
    reason: Optional[str] = Field(None, description="Lý do chọn món này")
    image_url: Optional[str] = Field(None, description="URL ảnh món phù hợp nhất")
    cook_time_minutes: Optional[int] = Field(None, description="Thời gian nấu (phút)")
    difficulty: Optional[str] = Field(None, description="Độ khó")
    servings: Optional[int] = Field(None, description="Số khẩu phần")
    matched_ingredients: List[str] = Field(
        default_factory=list, 
        description="Nguyên liệu người dùng có mà recipe cần"
    )
    missing_ingredients: List[str] = Field(
        default_factory=list, 
        description="Nguyên liệu còn thiếu"
    )
    substitutions: List[str] = Field(
        default_factory=list, 
        description="Gợi ý thay thế cho nguyên liệu thiếu"
    )
    instructions: List[str] = Field(
        default_factory=list, 
        description="Các bước nấu ngắn gọn"
    )
    alternative_recipes: List[AlternativeRecipe] = Field(
        default_factory=list, 
        description="Các công thức thay thế khác"
    )
    

class ErrorResponse(BaseModel):
    """Response khi có lỗi"""
    error: str
    detail: Optional[str] = None
