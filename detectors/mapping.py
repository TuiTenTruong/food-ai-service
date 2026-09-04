"""
Class mapping for Vietnamese Ingredient Detectors.
Contains standard 32 English classes to Vietnamese label translations
matching the NCKH research dataset.
"""

INGREDIENT_CLASSES_EN = [
    "beef", "bellpepper", "bittergourd", "bottlegourd", "broccoli",
    "cabbage", "carrot", "cauliflower", "chayote", "chicken",
    "chickenegg", "chickenleg", "chickenwin", "corn", "cucumber",
    "duckegg", "eggplant", "garlic", "ginger", "jicama",
    "okra", "onion", "pork", "potato", "pumpkin",
    "radish", "scallion", "shrimp", "spongegourd", "sweetpotato",
    "tofu", "tomato"
]

EN_TO_VI_MAPPING = {
    "beef": "Thịt bò",
    "bellpepper": "Ớt chuông",
    "bittergourd": "Khổ qua",
    "bottlegourd": "Bầu",
    "broccoli": "Bông cải xanh",
    "cabbage": "Bắp cải",
    "carrot": "Cà rốt",
    "cauliflower": "Súp lơ trắng",
    "chayote": "Su su",
    "chicken": "Thịt gà",
    "chickenegg": "Trứng gà",
    "chickenleg": "Đùi gà",
    "chickenwin": "Cánh gà",
    "corn": "Bắp ngô",
    "cucumber": "Dưa leo",
    "duckegg": "Trứng vịt",
    "eggplant": "Cà tím",
    "garlic": "Tỏi",
    "ginger": "Gừng",
    "jicama": "Củ đậu",
    "okra": "Đậu bắp",
    "onion": "Hành tây",
    "pork": "Thịt heo",
    "potato": "Khoai tây",
    "pumpkin": "Bí đỏ",
    "radish": "Củ cải trắng",
    "scallion": "Hành lá",
    "shrimp": "Tôm",
    "spongegourd": "Mướp hương",
    "sweetpotato": "Khoai lang",
    "tofu": "Đậu hũ",
    "tomato": "Cà chua",
    # Aliases
    "apple": "Táo",
    "banana": "Chuối",
    "beetroot": "Củ dền",
    "chilli": "Ớt",
    "lettuce": "Xà lách",
    "pineapple": "Dứa",
    "watermelon": "Dưa hấu",
}

def translate_label(label: str) -> str:
    """Translate model prediction label to Vietnamese standard name."""
    clean = label.lower().strip().replace(" ", "").replace("_", "").replace("-", "")
    for en_key, vi_name in EN_TO_VI_MAPPING.items():
        if clean == en_key.lower().replace("_", ""):
            return vi_name
    return EN_TO_VI_MAPPING.get(label.lower().strip(), label)
