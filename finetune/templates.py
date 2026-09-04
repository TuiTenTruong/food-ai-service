"""Mẫu câu hỏi — ~20–30 biến thể / công thức."""

# Hỏi trực tiếp tên món (14 mẫu)
RECIPE_QUESTION_TEMPLATES = [
    "Cách làm {name}?",
    "Hướng dẫn nấu {name}.",
    "Mình muốn nấu {name}.",
    "Có công thức {name} không?",
    "{name} làm như thế nào?",
    "Cho mình công thức {name} với.",
    "Nấu {name} cần những gì?",
    "Cách nấu {name} chi tiết từng bước.",
    "Tôi định làm {name} tối nay, hướng dẫn giúp.",
    "{short} nấu sao cho ngon?",
    "Bạn biết cách làm {short} không?",
    "Ghi lại công thức {name} giúp mình.",
    "Muốn học nấu {short}, bắt đầu từ đâu?",
    "Công thức chuẩn của {name} là gì?",
]

# Gợi ý từ nguyên liệu (8 mẫu — {ings} = "A, B và C")
SUGGEST_QUESTION_TEMPLATES = [
    "Nhà có {ings} thì nấu gì?",
    "Tủ lạnh còn {ings}, gợi ý một món.",
    "Mình có {ings}, nên nấu món gì?",
    "Với {ings} thì làm món Việt nào hợp?",
    "Chỉ còn {ings}, cho mình ý tưởng món ăn.",
    "Tôi có sẵn {ings}, gợi ý công thức phù hợp.",
    "Nguyên liệu hiện có: {ings}. Nấu gì ngon nhỉ?",
    "Help: có {ings} — món nào dễ nấu?",
]

# Mẹo / bước cụ thể (2 mẫu, cần {tip_topic})
TIP_QUESTION_TEMPLATES = [
    "Mẹo nấu {name} ngon là gì?",
    "Khi làm {name} cần lưu ý gì?",
]

# RAG — user đã có context (2 mẫu)
RAG_QUESTION_TEMPLATES = [
    "Hướng dẫn mình nấu món này từng bước.",
    "Dựa công thức trên, liệt kê nguyên liệu và cách làm đầy đủ.",
]

# Món không có trong DB (~100 mẫu global)
REJECT_DISHES = [
    "phở bò Hà Nội",
    "bún chả Obama",
    "pizza hải sản",
    "sushi cá hồi",
    "hamburger bò Mỹ",
    "pad thái",
    "ramen Nhật",
    "bánh mì pate gan",
    "lẩu Thái",
    "gà rán KFC",
    "salad Caesar",
    "bò Wellington",
    "tacos Mexico",
    "curry Ấn Độ",
    "lasagna Ý",
    "bánh xèo miền Trung",
    "cháo lươn Nghệ An",
    "bún riêu cua đồng",
    "mì Quảng",
    "cao lầu Hội An",
    "bánh tét lá cẩm",
    "nem nướng Nha Trang",
    "bánh canh giò heo",
    "hủ tiếu Nam Vang",
    "bánh cuốn Thanh Trì",
]

REJECT_QUESTION_TEMPLATES = [
    "Cách nấu {dish}?",
    "Cho mình công thức {dish}.",
    "Hướng dẫn làm {dish} chi tiết.",
    "Mình muốn nấu {dish}, có công thức không?",
]

SYSTEM_PROMPT = (
    "Bạn là đầu bếp tư vấn món ăn Việt Nam. "
    "Trả lời bằng tiếng Việt, dùng markdown có cấu trúc. "
    "Chỉ dùng công thức có trong ngữ cảnh hoặc dữ liệu đã biết; không bịa món."
)
