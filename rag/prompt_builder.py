"""
Strict RAG Prompt Builder for Food AI Service.
Ensures zero-hallucination, grounds responses strictly in retrieved database recipes,
and handles no-result scenarios gracefully.
"""

from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """Bạn là trợ lý đầu bếp chuyên nghiệp thuộc hệ thống nghiên cứu công thức món ăn truyền thống Việt Nam.

QUY TẮC BẮT BUỘC:
1. Bạn CHỈ ĐƯỢC PHÉP gợi ý và hướng dẫn các món ăn có trong danh sách CÔNG THỨC TRÍCH XUẤT TỪ DATABASE được cung cấp dưới đây.
2. TUYỆT ĐỐI KHÔNG tự bịa đặt công thức hoặc giả vờ rằng thông tin đến từ cơ sở dữ liệu nếu nó không có trong context.
3. Nếu không có công thức nào phù hợp trong database, HÃY NÓI RÕ: "Cơ sở dữ liệu hiện chưa có công thức phù hợp với nguyên liệu của bạn." và lịch sự đề xuất các nguyên liệu phổ biến cần thêm.
4. Ưu tiên món ăn có nhiều nguyên liệu khớp nhất với danh sách người dùng đang có.
5. Luôn trả lời bằng tiếng Việt tự nhiên, ngắn gọn, súc tích và chuẩn xác.
"""


def build_recipe_suggestion_prompt(
    user_ingredients: List[str],
    retrieved_recipes: List[Dict[str, Any]],
    query: Optional[str] = None
) -> str:
    """
    Build structured prompt for LLM recipe suggestion.
    """
    if not retrieved_recipes:
        return f"""
NGUYÊN LIỆU NGƯỜI DÙNG: {', '.join(user_ingredients) if user_ingredients else 'Không có'}
CÂU HỎI / YÊU CẦU: {query or 'Gợi ý món ăn'}

CÔNG THỨC TRÍCH XUẤT TỪ DATABASE:
(Không tìm thấy công thức nào trong cơ sở dữ liệu khớp với yêu cầu này)

YÊU CẦU ĐẦU RA:
- Thông báo rõ ràng rằng cơ sở dữ liệu hiện không có món ăn nào phù hợp với các nguyên liệu trên.
- Đưa ra lời khuyên ngắn gọn về các nguyên liệu cơ bản có thể bổ sung.
- Trả lời theo định dạng JSON chuẩn:
{{
  "best_recipe": null,
  "recipe_id": null,
  "reason": "Không tìm thấy công thức phù hợp trong cơ sở dữ liệu cho các nguyên liệu này.",
  "matched_ingredients": [],
  "missing_ingredients": [],
  "substitutions": [],
  "instructions": [],
  "alternative_recipes": []
}}
"""

    context_parts = []
    for idx, item in enumerate(retrieved_recipes, 1):
        recipe = item["recipe"]
        matched = item["matched_ingredients"]
        missing = item["missing_ingredients"]
        
        # Build ingredients summary
        ing_strs = [f"{i['name']} ({i['quantity']} {i['unit']})".strip() for i in recipe.get("ingredients", [])]
        
        block = f"""--- CÔNG THỨC {idx} ---
- ID: {recipe['id']}
- Tên món: {recipe['name']}
- Mô tả: {recipe['description']}
- Thời gian nấu: {recipe['cook_time_minutes']} phút | Độ khó: {recipe['difficulty']} | Khẩu phần: {recipe['servings']} người
- Nguyên liệu trong công thức: {', '.join(ing_strs)}
- Nguyên liệu khớp hiện có ({len(matched)}): {', '.join(matched) if matched else 'Chưa có'}
- Nguyên liệu còn thiếu ({len(missing)}): {', '.join(missing) if missing else 'Đủ nguyên liệu'}
- Các bước thực hiện:
{recipe['steps_text']}
"""
        context_parts.append(block)

    formatted_context = "\n".join(context_parts)

    prompt = f"""
NGUYÊN LIỆU NGƯỜI DÙNG CÓ:
{', '.join(user_ingredients) if user_ingredients else 'Tất cả'}

CÂU HỎI / YÊU CẦU BỔ SUNG:
{query or 'Hãy chọn món ăn phù hợp nhất với nguyên liệu của tôi và hướng dẫn cách nấu.'}

CÔNG THỨC TRÍCH XUẤT TỪ DATABASE:
{formatted_context}

YÊU CẦU XỬ LÝ:
1. Chọn món ăn phù hợp nhất từ danh sách trên (ưu tiên món khớp nhiều nguyên liệu nhất).
2. Giải thích ngắn gọn lý do tại sao món này được chọn.
3. Liệt kê chính xác nguyên liệu khớp và nguyên liệu còn thiếu.
4. Gợi ý thay thế nguyên liệu (nếu có thể).
5. Trình bày các bước nấu ngắn gọn, rõ ràng theo đúng công thức từ database.
6. Liệt kê các món thay thế khả dĩ khác từ danh sách trích xuất.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Trả về DUY NHẤT một JSON hợp lệ (không kèm markdown code block thừa thãi) theo cấu trúc:
{{
  "best_recipe": "Tên món được chọn",
  "recipe_id": "ID món (ví dụ: recipe-book-0001)",
  "reason": "Giải thích ngắn gọn lý do chọn món",
  "matched_ingredients": ["nguyên liệu 1", "nguyên liệu 2"],
  "missing_ingredients": ["nguyên liệu thiếu 1"],
  "substitutions": ["có thể thay thế A bằng B"],
  "instructions": [
    "Bước 1: ...",
    "Bước 2: ..."
  ],
  "alternative_recipes": [
    {{"id": "...", "name": "...", "matched_count": 2, "missing_count": 1}}
  ]
}}
"""
    return prompt.strip()
