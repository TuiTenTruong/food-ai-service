"""
RAG Chatbot Service - Chat với AI về công thức nấu ăn
Sử dụng RAG để tìm kiếm công thức phù hợp và GPT để sinh câu trả lời
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

from .embedding_service import get_embedding_service
from .retrieval_service import get_retrieval_service

load_dotenv()


# ============== SYSTEM PROMPTS ==============

COOKING_ASSISTANT_SYSTEM_PROMPT = """Bạn là một đầu bếp chuyên nghiệp và trợ lý nấu ăn thông minh.

NHIỆM VỤ:
- Tư vấn công thức nấu ăn dựa trên nguyên liệu người dùng có
- Hướng dẫn cách nấu từng bước chi tiết
- Gợi ý món ăn phù hợp với hoàn cảnh (bữa sáng, bữa tối, tiệc, diet...)
- Giải đáp thắc mắc về nấu ăn, bảo quản thực phẩm, dinh dưỡng

NGUYÊN TẮC:
1. Luôn trả lời bằng tiếng Việt
2. Thân thiện, dễ hiểu, không quá học thuật
3. Nếu có công thức từ database, ưu tiên sử dụng thông tin đó
4. Có thể bổ sung mẹo vặt, biến tấu từ kinh nghiệm
5. Nếu không chắc chắn, thừa nhận và đưa ra gợi ý chung

ĐỊNH DẠNG TRẢ LỜI:
- Dùng markdown cho format (**, -, 1. 2. 3.)
- Chia nhỏ thành các phần rõ ràng
- Emoji phù hợp để thân thiện hơn 🍳 🥗 👨‍🍳
"""

RECIPE_CONTEXT_PROMPT = """
## CÔNG THỨC THAM KHẢO TỪ DATABASE:
{recipe_context}

Dựa vào các công thức trên (nếu có) và kiến thức của bạn, hãy trả lời câu hỏi của người dùng.
Nếu công thức phù hợp, hãy trích dẫn và hướng dẫn chi tiết.
"""


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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.temperature = 0.7
        self.embedding_service = get_embedding_service()
        self.retrieval_service = get_retrieval_service()
        
        # In-memory conversation storage (for demo)
        # Production nên dùng Redis hoặc Database
        self._conversations: Dict[str, Dict] = {}
        
        print(" [ChatbotService] Initialized with GPT-4o-mini")
    
    def _extract_ingredients_from_message(self, message: str) -> List[str]:
        """
        Trích xuất nguyên liệu được đề cập trong message
        Dùng GPT để phân tích
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            # Parse JSON
            if content.startswith('['):
                return json.loads(content)
            return []
        except Exception as e:
            print(f" [ChatbotService] Error extracting ingredients: {e}")
            return []
    
    def _should_search_recipes(self, message: str) -> bool:
        """
        Kiểm tra xem có cần tìm công thức không
        """
        recipe_keywords = [
            'công thức', 'cách nấu', 'cách làm', 'nấu gì', 'ăn gì',
            'recipe', 'cook', 'món', 'chế biến', 'làm sao', 'hướng dẫn',
            'gợi ý', 'suggest', 'nguyên liệu', 'ingredient'
        ]
        message_lower = message.lower()
        return any(kw in message_lower for kw in recipe_keywords)
    
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
        if recipes and self._should_search_recipes(message):
            print(" [ChatbotService] Searching recipes...")
            
            # Extract ingredients from message or use pantry
            ingredients = user_pantry or self._extract_ingredients_from_message(message)
            
            if ingredients:
                print(f" [ChatbotService] Ingredients: {ingredients}")
                
                # Retrieve relevant recipes
                retrieved_recipes = self.retrieval_service.retrieve(
                    user_ingredients=ingredients,
                    recipes=recipes,
                    top_k=5
                )
                
                recipe_context = self._format_recipes_for_context(retrieved_recipes)
                
                # Clear index after use
                self.retrieval_service.clear()
        
        # Build messages for GPT
        gpt_messages = [
            {"role": "system", "content": COOKING_ASSISTANT_SYSTEM_PROMPT}
        ]
        
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
        
        # Call GPT
        try:
            print(" [ChatbotService] Calling GPT...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=gpt_messages,
                temperature=self.temperature,
                max_tokens=1500
            )
            
            assistant_content = response.choices[0].message.content
            print(f" [ChatbotService] Response: {len(assistant_content)} chars")
            
        except Exception as e:
            print(f" [ChatbotService] GPT Error: {e}")
            assistant_content = "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
        
        # Add assistant message
        assistant_msg = {
            'role': 'assistant',
            'content': assistant_content,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        conversation['messages'].append(assistant_msg)
        conversation['updated_at'] = assistant_msg['timestamp']
        
        # Build response
        return {
            'session_id': session_id,
            'user_message': user_msg,
            'assistant_message': assistant_msg,
            'metadata': {
                'recipes_searched': len(retrieved_recipes) > 0,
                'recipes_found': len(retrieved_recipes),
                'model': self.model
            }
        }
    
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
