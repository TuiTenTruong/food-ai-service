"""
RAG Chatbot Service - Chat với AI về công thức nấu ăn
Sử dụng RAG để tìm kiếm công thức phù hợp và GPT để sinh câu trả lời
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .ingredient_format import format_ingredient_amount
from .llm_client import get_llm_client

load_dotenv()


# ============== SYSTEM PROMPTS ==============

COOKING_ASSISTANT_SYSTEM_PROMPT = """Bạn là một đầu bếp chuyên nghiệp và trợ lý nấu ăn thông minh.

NHIỆM VỤ:
- Tư vấn công thức nấu ăn dựa trên nguyên liệu người dùng có
- Hướng dẫn cách nấu từng bước chi tiết, đầy đủ
- Gợi ý món ăn phù hợp với hoàn cảnh (bữa sáng, bữa tối, tiệc, diet...)
- Giải đáp thắc mắc về nấu ăn, bảo quản thực phẩm, dinh dưỡng

NGUYÊN TẮC:
1. Luôn trả lời bằng tiếng Việt
2. Thân thiện, dễ hiểu, không quá học thuật
3. Nếu có công thức từ database, ưu tiên sử dụng thông tin đó
4. Có thể bổ sung mẹo vặt, biến tấu từ kinh nghiệm
5. Nếu không chắc chắn, thừa nhận và đưa ra gợi ý chung
6. KHÔNG trả lời quá ngắn khi user hỏi công thức — phải viết đủ nguyên liệu và từng bước nấu

ĐỊNH DẠNG TRẢ LỜI:
- Dùng markdown cho format (**, -, 1. 2. 3.)
- Chia nhỏ thành các phần rõ ràng
- Emoji phù hợp để thân thiện hơn 🍳 🥗 👨‍🍳
"""

RECIPE_RESPONSE_FORMAT = """
Khi user hỏi công thức / cách nấu / gợi ý món, BẮT BUỘC trả lời đầy đủ theo mẫu markdown sau (không bỏ sót mục, không trả lời 1-2 câu):

## 🍳 [Tên món]

**Mô tả ngắn:** [1-2 câu giới thiệu món]

### Nguyên liệu
- [nguyên liệu 1]: [lượng]
- [nguyên liệu 2]: [lượng]
- ...

### Nguyên liệu bạn đã có / còn thiếu
- **Đã có:** ...
- **Còn thiếu:** ... (hoặc "Không thiếu")

### Cách làm
1. [Bước 1 chi tiết]
2. [Bước 2 chi tiết]
3. [Bước 3 chi tiết]
4. ...

### Mẹo vặt 👨‍🍳
- [Mẹo 1]
- [Mẹo 2]

Viết tối thiểu 250-400 từ khi hướng dẫn nấu. Không kết thúc sớm khi chưa liệt kê hết bước nấu.
"""

RECIPE_CONTEXT_PROMPT = """
## CÔNG THỨC THAM KHẢO TỪ DATABASE:
{recipe_context}

Dựa vào các công thức trên (nếu có) và kiến thức của bạn, hãy trả lời câu hỏi của người dùng.
Nếu công thức phù hợp, hãy trích dẫn và hướng dẫn chi tiết.
"""


# Precomputed vocabulary for fast rule-based ingredient extraction (<1ms)
_BASE_INGREDIENTS = [
    # Thịt & Gia cầm
    "thịt ba chỉ", "thịt lợn", "thịt heo", "nạc lợn", "nạc heo", "sườn non", "sườn heo", "sườn",
    "thịt bò nạc", "thịt bò", "bắp bò", "bò", "thịt gà", "ức gà", "đùi gà", "cánh gà", "gà ta", "gà",
    "thịt vịt", "vịt", "chim bồ câu",
    # Hải sản
    "tôm sú", "tôm thẻ", "tôm khô", "tôm", "mực ống", "mực trứng", "mực", "bạch tuộc",
    "cá lóc", "cá quả", "cá chép", "cá diêu hồng", "cá hồi", "cá basa", "cá thu", "cá trắm", "cá",
    "cua đồng", "cua biển", "cua", "nghêu", "sò", "ốc", "hến",
    # Trứng & Đậu
    "trứng gà", "trứng vịt", "trứng cút", "trứng",
    "đậu hũ", "đậu phụ", "tàu hũ", "đậu que", "đậu cove", "đậu xanh", "đậu đen", "đậu đỏ", "đậu phộng", "lạc",
    # Rau củ
    "cà chua bi", "cà chua", "dưa leo", "dưa chuột", "cà rốt", "khoai tây", "khoai lang", "khoai môn", "củ cải trắng", "củ cải",
    "bắp cải", "cải thảo", "cải ngọt", "cải thìa", "cải bẹ xanh", "rau muống", "rau mồng tơi", "rau đay", "rau ngót",
    "khổ qua", "mướp đắng", "mướp hương", "mướp", "bầu", "bí xanh", "bí đao", "bí đỏ", "su su", "su hào",
    "bông cải xanh", "súp lơ xanh", "bông cải trắng", "súp lơ trắng", "súp lơ",
    "bắp ngô", "bắp nếp", "bắp ngọt", "ngô", "củ đậu", "củ sắn",
    # Nấm
    "nấm hương", "nấm rơm", "nấm kim châm", "nấm đùi gà", "nấm bào ngư", "nấm mộc nhĩ", "mộc nhĩ", "nấm",
    # Gia vị & Rau thơm
    "hành lá", "hành hoa", "hành tây", "hành tím", "hành khô", "tỏi", "sả", "gừng", "ớt chuông", "ớt", "chanh",
    "ngò gai", "ngò rí", "rau mùi", "húng quế", "thì là", "lá lốt", "tía tô",
    # Tinh bột
    "bún tươi", "bún", "bánh phở", "phở", "miến dong", "miến", "mì tôm", "mì", "cơm nguội", "cơm", "gạo"
]

def _build_precomputed_vocab():
    import unicodedata
    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

    try:
        from rag.retriever import SYNONYMS
        syn_list = SYNONYMS
    except Exception:
        syn_list = []

    vocab = set(_BASE_INGREDIENTS)
    for s in syn_list:
        vocab.update(s)

    sorted_terms = sorted(vocab, key=lambda x: len(x), reverse=True)
    compiled = []
    for term in sorted_terms:
        norm = _strip_accents(term)
        pat_raw = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        pat_norm = re.compile(rf"(?<!\w){re.escape(norm)}(?!\w)", re.IGNORECASE)
        compiled.append((term, pat_raw, pat_norm))
    return compiled

_PRECOMPUTED_VOCAB = _build_precomputed_vocab()


class ChatbotService:
    """
    RAG-based Chatbot Service cho tư vấn nấu ăn
    
    Flow:
    1. Nhận message từ user
    2. Phân tích intent (hỏi công thức, hỏi chung, hỏi về nguyên liệu...)
    3. Nếu cần, retrieve công thức phù hợp từ database
    4. Gọi GPT với context để sinh câu trả lời
    """
    
    def __init__(self):
        self.client = get_llm_client()
        self.model = self.client.model
        self.temperature = float(os.getenv("CHATBOT_TEMPERATURE", "0.5"))
        print(f" [ChatbotService] provider={self.client.provider} model={self.model}")
        self.max_tokens = int(os.getenv("CHATBOT_MAX_TOKENS", "4096"))
        self.min_recipe_chars = int(os.getenv("CHATBOT_MIN_RECIPE_CHARS", "400"))
        self.embedding_service = None
        self.retrieval_service = None
        if os.getenv("USE_CHROMA_RAG", "false").lower() == "true":
            try:
                self.embedding_service = get_embedding_service() if get_embedding_service else None
                self.retrieval_service = get_retrieval_service() if get_retrieval_service else None
            except Exception as e:
                print(f" [ChatbotService] Warning: Chroma retrieval initialization skipped: {e}")
        
        # In-memory conversation storage (for demo)
        # Production nên dùng Redis hoặc Database
        self._conversations: Dict[str, Dict] = {}
    
    def _extract_ingredients_with_llm(self, message: str) -> List[str]:
        """
        [Giải pháp cũ] Trích xuất nguyên liệu được đề cập trong message bằng 1 lượt gọi LLM.
        """
        try:
            print(" [ChatbotService] Extracting ingredients via LLM...")
            response = self.client.chat_completions_create(
                messages=[
                    {
                        "role": "system",
                        "content": "Trích xuất các nguyên liệu thực phẩm được đề cập trong câu. Trả về JSON array, ví dụ: [\"trứng\", \"cà chua\"]. Nếu không có nguyên liệu nào, trả về []"
                    },
                    {
                        "role": "user", 
                        "content": message
                    }
                ],
                temperature=0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            # Clean markdown code blocks
            if "```" in content:
                content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
                content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE).strip()
            # Parse JSON
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception as e:
            print(f" [ChatbotService] Error extracting ingredients via LLM: {e}")
            return []

    def _extract_ingredients_with_rule(self, message: str) -> List[str]:
        """
        [Giải pháp Tối ưu 1] Trích xuất nguyên liệu bằng Rule / Dictionary Matching (<1ms).
        So khớp trực tiếp từ điển 50+ nguyên liệu và từ đồng nghĩa trong CSDL tiếng Việt.
        """
        import unicodedata

        def _remove_accents(s: str) -> str:
            s = unicodedata.normalize("NFD", s)
            return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

        # Các từ đơn tiếng Việt dễ bị trùng lặp âm khi bỏ dấu (ví dụ: 'Tôi' -> 'toi' trùng 'tỏi')
        _AMBIGUOUS_WORDS = {
            "toi", "co", "va", "thi", "lam", "mon", "gi", "an", "o",
            "dau", "nao", "la", "cho", "minh", "nha", "con", "duoc", "khong"
        }

        msg_lower = " " + message.lower() + " "
        msg_norm = " " + _remove_accents(message) + " "
        has_accents = (msg_lower != msg_norm)

        matched_ingredients: List[str] = []
        matched_spans: List[tuple] = []

        for term, pat_raw, pat_norm in _PRECOMPUTED_VOCAB:
            term_norm = _remove_accents(term)

            # 1. So khớp trực tiếp có dấu trước (chính xác tuyệt đối)
            found_match = pat_raw.search(msg_lower)

            # 2. So khớp không dấu an toàn (chỉ cho cụm nhiều từ hoặc từ không bị lưỡng nghĩa)
            if not found_match:
                is_safe = (" " in term) or (term_norm not in _AMBIGUOUS_WORDS and not has_accents)
                if is_safe:
                    found_match = pat_norm.search(msg_norm)

            if found_match:
                span = found_match.span()
                overlap = any(
                    (span[0] >= existing[0] and span[1] <= existing[1]) or
                    (span[0] < existing[1] and span[1] > existing[0])
                    for existing in matched_spans
                )
                if not overlap:
                    matched_ingredients.append(term)
                    matched_spans.append(span)

        if matched_ingredients:
            print(f" [ChatbotService] Rule-based extracted ingredients (<1ms): {matched_ingredients}")
        return matched_ingredients

    def _extract_ingredients(self, message: str) -> List[str]:
        """
        Bộ điều phối trích xuất nguyên liệu dựa trên cấu hình INGREDIENT_EXTRACTION_MODE:
        - 'llm': Chạy theo giải pháp cũ (gọi LLM tách từ).
        - 'rule': Chạy tối ưu bằng từ điển Python (1ms, tiết kiệm 1 lượt LLM).
        - 'hybrid' (mặc định): Thử rule nhanh trước; nếu không có kết quả mới gọi LLM.
        """
        mode = os.getenv("INGREDIENT_EXTRACTION_MODE", "hybrid").lower().strip()

        if mode == "llm":
            return self._extract_ingredients_with_llm(message)
        elif mode == "rule":
            return self._extract_ingredients_with_rule(message)
        else:  # hybrid
            rule_res = self._extract_ingredients_with_rule(message)
            if rule_res:
                return rule_res
            return self._extract_ingredients_with_llm(message)

    def _extract_ingredients_from_message(self, message: str) -> List[str]:
        """Alias tương thích ngược."""
        return self._extract_ingredients(message)
    
    def _normalize_for_match(self, text: str) -> str:
        """Chuẩn hóa text để so khớp keyword (bỏ dấu, lowercase)."""
        import unicodedata

        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        return "".join(c for c in text if unicodedata.category(c) != "Mn")

    def _should_search_recipes(self, message: str) -> bool:
        """
        Kiểm tra xem có cần tìm công thức không.
        Hỗ trợ cả câu tiếng Việt có/không dấu.
        """
        recipe_keywords = [
            "cong thuc", "cach nau", "cach lam", "nau gi", "an gi",
            "recipe", "cook", "mon", "che bien", "lam sao", "huong dan",
            "goi y", "suggest", "nguyen lieu", "ingredient", "nau mon",
            "lam mon", "nau an", "huong dan nau", "chi tiet",
        ]
        normalized = self._normalize_for_match(message)
        return any(kw in normalized for kw in recipe_keywords)
    
    def _format_recipes_for_context(self, retrieved: List[Dict]) -> str:
        """
        Format recipes retrieved thành context cho prompt
        """
        if not retrieved:
            return "Không tìm thấy công thức phù hợp trong database."
        
        parts = []
        for i, item in enumerate(retrieved[:3], 1):  # Top 3
            recipe = item['recipe']
            matched = item.get('matched_ingredients', [])
            missing = item.get('missing_ingredients', [])
            
            text = f"""
### {i}. {recipe.get('name', 'Không rõ')}
- **Mô tả**: {recipe.get('description', 'Không có')}
- **Nguyên liệu khớp**: {', '.join(matched) if matched else 'Không rõ'}
- **Nguyên liệu thiếu**: {', '.join(missing) if missing else 'Không có'}
- **Cách làm**: {recipe.get('steps', 'Không có hướng dẫn')}
"""
            parts.append(text.strip())
        
        return "\n\n".join(parts)

    def _has_recipe_sections(self, content: str) -> bool:
        normalized = self._normalize_for_match(content)
        required = ["nguyen lieu", "cach lam"]
        return all(section in normalized for section in required)

    def _format_steps(self, steps: Any) -> str:
        if isinstance(steps, list):
            return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps) if step)

        if not steps:
            return "1. Chuẩn bị nguyên liệu.\n2. Nấu theo hướng dẫn trong công thức."

        text = str(steps).strip()
        if re.search(r"(?m)^\s*\d+[\.\)]\s+", text):
            return text

        parts = [p.strip() for p in re.split(r"(?:Bước|Buoc)\s*\d+\s*[:.\-]?", text, flags=re.IGNORECASE) if p.strip()]
        if len(parts) > 1:
            return "\n".join(f"{i + 1}. {part}" for i, part in enumerate(parts))

        return f"1. {text}"

    def _build_structured_recipe_response(
        self,
        retrieved: List[Dict],
        user_pantry: Optional[List[str]] = None,
    ) -> str:
        """Sinh công thức markdown đầy đủ từ kết quả RAG (fallback cho model nhỏ)."""
        if not retrieved:
            return ""

        best = retrieved[0]
        recipe = dict(best.get("recipe") or {})
        document = best.get("document") or ""

        if not recipe.get("description") and document:
            desc_match = re.search(r"Mô tả:\s*(.+?)(?:\n\n|\nNguyên liệu:)", document, re.DOTALL)
            if desc_match:
                recipe["description"] = desc_match.group(1).strip()

        if not recipe.get("ingredients") and document:
            ing_match = re.search(r"Nguyên liệu:\s*\n([\s\S]+?)(?:\n\nCách làm:|\Z)", document)
            if ing_match:
                parsed = []
                for line in ing_match.group(1).splitlines():
                    line = line.strip().lstrip("-").strip()
                    if not line:
                        continue
                    amount_match = re.match(r"(.+?)\s*\((.+)\)\s*$", line)
                    if amount_match:
                        parsed.append({"name": amount_match.group(1).strip(), "amount": amount_match.group(2).strip()})
                    else:
                        parsed.append({"name": line, "amount": ""})
                if parsed:
                    recipe["ingredients"] = parsed

        if not recipe.get("steps") and document:
            steps_match = re.search(r"Cách làm:\s*\n([\s\S]+)", document)
            if steps_match:
                recipe["steps"] = steps_match.group(1).strip()

        matched = best.get("matched_ingredients", [])
        missing = best.get("missing_ingredients", [])

        ingredient_lines = []
        for ing in recipe.get("ingredients", []):
            if isinstance(ing, dict):
                name = ing.get("name", "").strip()
                label = format_ingredient_amount(ing.get("quantity"), ing.get("unit"))
                if not label and ing.get("amount"):
                    label = str(ing.get("amount", "")).strip()
                if name:
                    ingredient_lines.append(f"- **{name}**: {label}" if label else f"- **{name}**")
            elif isinstance(ing, str) and ing.strip():
                ingredient_lines.append(f"- **{ing.strip()}**")

        steps_text = self._format_steps(recipe.get("steps", ""))
        pantry_text = ", ".join(user_pantry) if user_pantry else ", ".join(matched)

        return f"""## 🍳 {recipe.get('name', 'Món gợi ý')}

**Mô tả ngắn:** {recipe.get('description', 'Món ăn phù hợp với nguyên liệu bạn đang có.')}

### Nguyên liệu
{chr(10).join(ingredient_lines) if ingredient_lines else '- (Chưa có chi tiết nguyên liệu)'}

### Nguyên liệu bạn đã có / còn thiếu
- **Đã có:** {', '.join(matched) if matched else pantry_text or 'Không rõ'}
- **Còn thiếu:** {', '.join(missing) if missing else 'Không thiếu'}

### Cách làm
{steps_text}

### Mẹo vặt 👨‍🍳
- Nêm nếm từng bước để vừa khẩu vị
- Có thể thay thế nguyên liệu thiếu bằng các loại tương đương nếu cần
"""
    
    def chat(
        self,
        session_id: str,
        message: str,
        recipes: Optional[List[Dict]] = None,
        user_pantry: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main chat method
        
        Args:
            session_id: ID của conversation session
            message: Tin nhắn từ user
            recipes: Danh sách công thức từ database (optional)
            user_pantry: Nguyên liệu user hiện có (optional)
        
        Returns:
            Dict với assistant response và metadata
        """
        print(f"\n{'='*50}")
        print(f" [ChatbotService] Session: {session_id}")
        print(f" [ChatbotService] Message: {message[:100]}...")
        print(f"{'='*50}")
        
        # Get or create conversation
        if session_id not in self._conversations:
            self._conversations[session_id] = {
                'id': session_id,
                'messages': [],
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        
        conversation = self._conversations[session_id]
        
        # Add user message
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        conversation['messages'].append(user_msg)
        
        # Build context
        recipe_context = ""
        retrieved_recipes = []
        
        # Check if we should search recipes
        wants_recipe = self._should_search_recipes(message) or bool(user_pantry)
        if wants_recipe:
            print(" [ChatbotService] Searching recipes...")
            
            # Extract ingredients from message or use pantry
            ingredients = user_pantry or self._extract_ingredients_from_message(message)
            
            if ingredients:
                print(f" [ChatbotService] Ingredients: {ingredients}")
                
                # Try retrieval service first if available
                if self.retrieval_service and recipes:
                    try:
                        retrieved_recipes = self.retrieval_service.retrieve(
                            user_ingredients=ingredients,
                            recipes=recipes,
                            top_k=5
                        )
                        self.retrieval_service.clear()
                    except Exception as e:
                        print(f" [ChatbotService] Retrieval error: {e}")
                        retrieved_recipes = []
                
                # Fallback to in-memory RAG hybrid retriever (works on Modal without chromadb)
                if not retrieved_recipes:
                    try:
                        from rag import get_rag_service
                        rag = get_rag_service()
                        retrieved_recipes = rag.retriever.retrieve(
                            user_ingredients=ingredients,
                            query=message,
                            top_k=5
                        )
                    except Exception as e:
                        print(f" [ChatbotService] RAG fallback retrieval error: {e}")
                
                if retrieved_recipes:
                    recipe_context = self._format_recipes_for_context(retrieved_recipes)

        structured_recipe = ""
        if retrieved_recipes and wants_recipe:
            structured_recipe = self._build_structured_recipe_response(retrieved_recipes, user_pantry)

        # Model local nhỏ dễ hallucinate → ưu tiên template RAG khi đã có công thức khớp
        if structured_recipe:
            print(" [ChatbotService] Using structured RAG recipe template")
            assistant_content = structured_recipe
            used_structured_fallback = True
        else:
            used_structured_fallback = False
            assistant_content = ""
        
        # Build messages for GPT
        wants_recipe = self._should_search_recipes(message) or bool(user_pantry) or bool(recipe_context)
        gpt_messages = [
            {"role": "system", "content": COOKING_ASSISTANT_SYSTEM_PROMPT}
        ]

        if wants_recipe:
            gpt_messages.append({
                "role": "system",
                "content": RECIPE_RESPONSE_FORMAT
            })
        
        # Add recipe context if available
        if recipe_context:
            gpt_messages.append({
                "role": "system",
                "content": RECIPE_CONTEXT_PROMPT.format(recipe_context=recipe_context)
            })
        
        # Add pantry info if available
        if user_pantry:
            gpt_messages.append({
                "role": "system",
                "content": f"Nguyên liệu người dùng hiện có: {', '.join(user_pantry)}"
            })
        
        # Add conversation history (last 10 messages for context window)
        for msg in conversation['messages'][-10:]:
            gpt_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Call GPT (chỉ khi chưa có template RAG)
        if not assistant_content:
            try:
                print(" [ChatbotService] Calling GPT...")
                response = self.client.chat_completions_create(
                    messages=gpt_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                choice = response.choices[0]
                assistant_content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                print(
                    f" [ChatbotService] Response: {len(assistant_content)} chars, "
                    f"finish_reason={finish_reason}, max_tokens={self.max_tokens}"
                )

                if finish_reason == "length" and assistant_content:
                    assistant_content += (
                        "\n\n*(Phản hồi bị cắt do giới hạn độ dài. "
                        "Hãy hỏi tiếp phần còn lại hoặc tăng CHATBOT_MAX_TOKENS.)*"
                    )

                if (
                    wants_recipe
                    and structured_recipe
                    and (
                        len(assistant_content.strip()) < self.min_recipe_chars
                        or not self._has_recipe_sections(assistant_content)
                    )
                ):
                    print(
                        " [ChatbotService] LLM response too short/unstructured, "
                        "using structured RAG recipe template"
                    )
                    assistant_content = structured_recipe
                    used_structured_fallback = True

            except Exception as e:
                print(f" [ChatbotService] GPT Error: {e}")
                assistant_content = structured_recipe or "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
                used_structured_fallback = bool(structured_recipe)
        
        # Extract structured suggested recipes from retrieved_recipes
        suggested_recipes = []
        for r in retrieved_recipes:
            rec = r.get("recipe") or {}
            suggested_recipes.append({
                "id": rec.get("id", ""),
                "name": rec.get("name", ""),
                "image_url": rec.get("image_url", ""),
                "cook_time_minutes": rec.get("cook_time_minutes", 30),
                "difficulty": rec.get("difficulty", "medium"),
                "servings": rec.get("servings", 2),
                "matched_ingredients": r.get("matched_ingredients", []),
                "missing_ingredients": r.get("missing_ingredients", []),
                "combined_score": r.get("combined_score", 0.0),
            })

        # Add assistant message
        assistant_msg = {
            'role': 'assistant',
            'content': assistant_content,
            'suggested_recipes': suggested_recipes,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        conversation['messages'].append(assistant_msg)
        conversation['updated_at'] = assistant_msg['timestamp']
        
        # Build response
        return {
            'session_id': session_id,
            'user_message': user_msg,
            'assistant_message': assistant_msg,
            'suggested_recipes': suggested_recipes,
            'metadata': {
                'recipes_searched': len(retrieved_recipes) > 0,
                'recipes_found': len(retrieved_recipes),
                'model': self.model,
                'provider': self.client.provider,
                'response_chars': len(assistant_content),
                'max_tokens': self.max_tokens,
                'recipe_format_requested': wants_recipe,
                'used_structured_fallback': used_structured_fallback,
            }
        }

    def stream_chat(
        self,
        session_id: str,
        message: str,
        recipes: Optional[List[Dict]] = None,
        user_pantry: Optional[List[str]] = None,
    ):
        """
        [Giải pháp Tối ưu 2] Streaming chat response using Server-Sent Events (SSE).
        Yields text chunks as they are generated by LLM token-by-token.
        Updates conversation history when complete.
        """
        if session_id not in self._conversations:
            self._conversations[session_id] = {
                'id': session_id,
                'messages': [],
                'created_at': datetime.now(timezone.utc).isoformat()
            }

        conversation = self._conversations[session_id]
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        conversation['messages'].append(user_msg)

        recipe_context = ""
        retrieved_recipes = []
        wants_recipe = self._should_search_recipes(message) or bool(user_pantry)

        if wants_recipe:
            ingredients = user_pantry or self._extract_ingredients(message)
            if ingredients:
                try:
                    from rag import get_rag_service
                    rag = get_rag_service()
                    retrieved_recipes = rag.retriever.retrieve(
                        user_ingredients=ingredients,
                        query=message,
                        top_k=5
                    )
                except Exception as e:
                    print(f" [ChatbotService] RAG retrieval error in stream: {e}")

                if retrieved_recipes:
                    recipe_context = self._format_recipes_for_context(retrieved_recipes)

        gpt_messages = [
            {"role": "system", "content": COOKING_ASSISTANT_SYSTEM_PROMPT}
        ]
        if wants_recipe:
            gpt_messages.append({"role": "system", "content": RECIPE_RESPONSE_FORMAT})
        if recipe_context:
            gpt_messages.append({"role": "system", "content": RECIPE_CONTEXT_PROMPT.format(recipe_context=recipe_context)})
        if user_pantry:
            gpt_messages.append({"role": "system", "content": f"Nguyên liệu người dùng hiện có: {', '.join(user_pantry)}"})

        for msg in conversation['messages'][-10:]:
            gpt_messages.append({"role": msg['role'], "content": msg['content']})

        full_chunks = []
        try:
            for chunk in self.client.chat_completions_stream(
                messages=gpt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                full_chunks.append(chunk)
                yield chunk
        except Exception as e:
            err_text = f"\n[Lỗi stream: {e}]"
            full_chunks.append(err_text)
            yield err_text

        # Record assistant message in conversation
        assistant_content = "".join(full_chunks).strip()
        stream_suggested_recipes = []
        for r in retrieved_recipes:
            rec = r.get("recipe") or {}
            stream_suggested_recipes.append({
                "id": rec.get("id", ""),
                "name": rec.get("name", ""),
                "image_url": rec.get("image_url", ""),
                "cook_time_minutes": rec.get("cook_time_minutes", 30),
                "difficulty": rec.get("difficulty", "medium"),
                "servings": rec.get("servings", 2),
                "matched_ingredients": r.get("matched_ingredients", []),
                "missing_ingredients": r.get("missing_ingredients", []),
                "combined_score": r.get("combined_score", 0.0),
            })

        assistant_msg = {
            'role': 'assistant',
            'content': assistant_content,
            'suggested_recipes': stream_suggested_recipes,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        conversation['messages'].append(assistant_msg)
        conversation['updated_at'] = assistant_msg['timestamp']
    
    def get_conversation(self, session_id: str) -> Optional[Dict]:
        """Get conversation by session ID"""
        return self._conversations.get(session_id)
    
    def get_messages(self, session_id: str) -> List[Dict]:
        """Get all messages in a conversation"""
        conv = self._conversations.get(session_id)
        if conv:
            return conv.get('messages', [])
        return []
    
    def create_session(self, title: Optional[str] = None) -> Dict:
        """Create a new chat session"""
        import uuid
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        self._conversations[session_id] = {
            'id': session_id,
            'title': title or 'Cuộc trò chuyện mới',
            'messages': [],
            'created_at': now,
            'updated_at': now
        }
        
        return {
            'id': session_id,
            'title': title or 'Cuộc trò chuyện mới',
            'created_at': now,
            'message_count': 0
        }
    
    def list_sessions(self) -> List[Dict]:
        """List all chat sessions"""
        sessions = []
        for session_id, conv in self._conversations.items():
            sessions.append({
                'id': session_id,
                'title': conv.get('title', 'Cuộc trò chuyện'),
                'created_at': conv.get('created_at'),
                'updated_at': conv.get('updated_at'),
                'message_count': len(conv.get('messages', []))
            })
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False


# Singleton
_chatbot_service = None

def get_chatbot_service() -> ChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
