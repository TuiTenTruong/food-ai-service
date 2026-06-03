"""
LLM Service - Gọi OpenAI Chat API để sinh gợi ý công thức
Xử lý prompt template và parse JSON response
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ============== PROMPT TEMPLATE ==============

RECIPE_SUGGESTION_PROMPT = """Bạn là một đầu bếp chuyên nghiệp và tư vấn ẩm thực.

## NGUYÊN LIỆU NGƯỜI DÙNG HIỆN CÓ:
{user_ingredients}

## CÁC CÔNG THỨC PHÙ HỢP ĐÃ TÌM ĐƯỢC:
{retrieved_recipes}

## YÊU CẦU:
Dựa vào nguyên liệu người dùng có và các công thức đã tìm được, hãy:

1. **Chọn công thức phù hợp nhất** - Ưu tiên món có nhiều nguyên liệu khớp nhất
2. **Giải thích lý do** - Tại sao món này phù hợp với nguyên liệu hiện có
3. **Liệt kê nguyên liệu còn thiếu** - Nếu có
4. **Gợi ý thay thế** - Nếu thiếu ít nguyên liệu, có thể thay bằng gì
5. **Viết lại cách nấu ngắn gọn** - Dễ hiểu, từng bước rõ ràng
6. **Gợi ý thêm** - 2-3 công thức thay thế khác nếu có

## RÀNG BUỘC QUAN TRỌNG:
- CHỈ sử dụng thông tin từ các công thức đã cung cấp
- KHÔNG bịa công thức mới ngoài danh sách
- Trả lời bằng tiếng Việt
- Nếu nguyên liệu gần giống (VD: "thịt gà" ≈ "gà") thì coi như khớp

## OUTPUT FORMAT:
Trả về JSON với cấu trúc sau (KHÔNG có markdown code block):

{{
    "best_recipe": "Tên món phù hợp nhất",
    "recipe_id": "ID của món",
    "reason": "Giải thích ngắn gọn tại sao chọn món này",
    "matched_ingredients": ["nguyên liệu 1", "nguyên liệu 2"],
    "missing_ingredients": ["nguyên liệu thiếu 1", "nguyên liệu thiếu 2"],
    "substitutions": ["có thể thay X bằng Y", "có thể bỏ qua Z"],
    "instructions": [
        "Bước 1: ...",
        "Bước 2: ...",
        "Bước 3: ..."
    ],
    "alternative_recipes": [
        {{"id": "...", "name": "Tên món 2", "matched_count": 3, "missing_count": 1}},
        {{"id": "...", "name": "Tên món 3", "matched_count": 2, "missing_count": 2}}
    ]
}}
"""


class LLMService:
    """
    Service gọi OpenAI Chat API
    Xử lý prompt và parse JSON response
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Model nhanh và rẻ, đủ tốt cho task này
        self.temperature = 0.3  # Thấp để output consistent
        print(f" [LLMService] Initialized with model: {self.model}")
    
    def _format_retrieved_recipes(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Format danh sách recipes retrieved thành text cho prompt
        """
        parts = []
        
        for i, item in enumerate(retrieved, 1):
            recipe = item['recipe']
            matched = item.get('matched_ingredients', [])
            missing = item.get('missing_ingredients', [])
            score = item.get('overlap_score', 0)
            
            recipe_text = f"""
### Công thức {i}: {recipe.get('name', 'Không rõ')}
- ID: {recipe.get('id', '')}
- Mô tả: {recipe.get('description', 'Không có mô tả')}
- Nguyên liệu khớp ({len(matched)}): {', '.join(matched) if matched else 'Không có'}
- Nguyên liệu thiếu ({len(missing)}): {', '.join(missing) if missing else 'Không có'}
- Độ khớp: {score:.0%}
- Cách làm: {recipe.get('steps', 'Không có hướng dẫn')}
"""
            parts.append(recipe_text.strip())
        
        return "\n\n".join(parts)
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON từ LLM response
        Xử lý cả trường hợp có markdown code block
        """
        # Loại bỏ markdown code block nếu có
        content = content.strip()
        
        # Pattern 1: ```json ... ```
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        else:
            # Pattern 2: ``` ... ```
            code_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if code_match:
                content = code_match.group(1)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f" [LLMService] JSON parse error: {e}")
            print(f" [LLMService] Raw content: {content[:500]}")
            return None
    
    def generate_suggestion(
        self, 
        user_ingredients: List[str],
        retrieved_recipes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Gọi LLM để sinh gợi ý công thức
        
        Args:
            user_ingredients: Danh sách nguyên liệu user có
            retrieved_recipes: Kết quả từ retrieval service
        
        Returns:
            Dict với thông tin gợi ý
        """
        if not retrieved_recipes:
            return {
                "best_recipe": None,
                "recipe_id": None,
                "reason": "Không tìm thấy công thức phù hợp với nguyên liệu của bạn",
                "matched_ingredients": [],
                "missing_ingredients": [],
                "substitutions": [],
                "instructions": [],
                "alternative_recipes": []
            }
        
        # Format prompt
        ingredients_str = ", ".join(user_ingredients)
        recipes_str = self._format_retrieved_recipes(retrieved_recipes)
        
        prompt = RECIPE_SUGGESTION_PROMPT.format(
            user_ingredients=ingredients_str,
            retrieved_recipes=recipes_str
        )
        
        print(f" [LLMService] Calling OpenAI with {len(user_ingredients)} ingredients and {len(retrieved_recipes)} recipes")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một đầu bếp chuyên nghiệp. Luôn trả lời bằng JSON hợp lệ, không có markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            print(f" [LLMService] Received response: {len(content)} chars")
            
            # Parse JSON
            result = self._parse_json_response(content)
            
            if result:
                return result
            else:
                # Fallback: trả về công thức đầu tiên với basic info
                best = retrieved_recipes[0]
                return {
                    "best_recipe": best['recipe'].get('name'),
                    "recipe_id": best['recipe'].get('id'),
                    "reason": "Đây là công thức khớp nhiều nguyên liệu nhất",
                    "matched_ingredients": best.get('matched_ingredients', []),
                    "missing_ingredients": best.get('missing_ingredients', []),
                    "substitutions": [],
                    "instructions": [best['recipe'].get('steps', 'Xem chi tiết trong công thức')],
                    "alternative_recipes": []
                }
        
        except Exception as e:
            print(f" [LLMService] Error: {e}")
            # Fallback error response
            return {
                "best_recipe": None,
                "recipe_id": None,
                "reason": f"Lỗi khi gọi AI: {str(e)}",
                "matched_ingredients": [],
                "missing_ingredients": [],
                "substitutions": [],
                "instructions": [],
                "alternative_recipes": []
            }


# Singleton
_llm_service = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
