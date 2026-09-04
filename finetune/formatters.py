"""Format công thức data_book.json → markdown assistant (khớp chatbot_service)."""
from __future__ import annotations

from typing import Any


def _amount(ing: dict[str, Any]) -> str:
    q = str(ing.get("quantity", "")).strip()
    u = str(ing.get("unit", "")).strip()
    if q and u:
        return f"{q} {u}"
    return q or u or "vừa đủ"


def format_recipe_markdown(
    recipe: dict[str, Any],
    *,
    owned_ingredients: list[str] | None = None,
) -> str:
    """Sinh câu trả lời assistant chuẩn markdown."""
    name = recipe["name"]
    lines = [
        f"## 🍳 {name}",
        "",
        f"**Mô tả ngắn:** {recipe.get('description', '').strip()}",
        "",
        "### Nguyên liệu",
    ]

    all_names = [i["name"] for i in recipe.get("ingredients", [])]
    for ing in recipe.get("ingredients", []):
        opt = " (tùy chọn)" if ing.get("is_optional") else ""
        lines.append(f"- {ing['name']}: {_amount(ing)}{opt}")

    if owned_ingredients is not None:
        owned_set = {x.strip().lower() for x in owned_ingredients}
        had = [n for n in all_names if n.lower() in owned_set]
        missing = [n for n in all_names if n.lower() not in owned_set]
        lines.extend(
            [
                "",
                "### Nguyên liệu bạn đã có / còn thiếu",
                f"- **Đã có:** {', '.join(had) if had else 'Không có'}",
                f"- **Còn thiếu:** {', '.join(missing) if missing else 'Không thiếu'}",
            ]
        )

    lines.extend(["", "### Cách làm"])
    for step in recipe.get("instructions", []):
        title = step.get("title") or f"Bước {step['step_number']}"
        desc = step.get("description", "").strip()
        lines.append(f"{step['step_number']}. **{title}:** {desc}")

    tips = [s["tip"] for s in recipe.get("instructions", []) if s.get("tip")]
    lines.extend(["", "### Mẹo vặt 👨‍🍳"])
    if tips:
        for tip in tips:
            lines.append(f"- {tip}")
    else:
        lines.append("- Nêm nếm vừa khẩu vị gia đình trước khi dọn món.")

    meta = []
    if recipe.get("cook_time_minutes"):
        meta.append(f"{recipe['cook_time_minutes']} phút")
    if recipe.get("difficulty"):
        meta.append(recipe["difficulty"])
    if recipe.get("servings"):
        meta.append(f"{recipe['servings']} khẩu phần")
    if meta:
        lines.extend(["", f"*Thời gian / độ khó: {' · '.join(meta)}*"])

    return "\n".join(lines)


def format_reject_response(dish_query: str) -> str:
    return (
        f"Xin lỗi, hiện tại mình chưa có công thức **{dish_query.strip()}** trong cơ sở dữ liệu món Việt của dự án.\n\n"
        "Bạn có thể:\n"
        "- Hỏi tên món khác trong sách nấu ăn truyền thống\n"
        "- Liệt kê nguyên liệu đang có để mình gợi ý món phù hợp\n\n"
        "Mình sẽ không tự bịa công thức khi chưa có dữ liệu xác thực."
    )


def format_rag_user_message(recipe: dict[str, Any], question: str) -> str:
    ing_lines = ", ".join(f"{i['name']} {_amount(i)}" for i in recipe.get("ingredients", []))
    steps = " | ".join(
        f"B{ s['step_number']}: {s.get('description', '')[:120]}"
        for s in recipe.get("instructions", [])
    )
    return (
        "[CÔNG THỨC THAM KHẢO]\n"
        f"Tên: {recipe['name']}\n"
        f"Mô tả: {recipe.get('description', '')}\n"
        f"Nguyên liệu: {ing_lines}\n"
        f"Các bước: {steps}\n\n"
        f"Câu hỏi: {question}"
    )
