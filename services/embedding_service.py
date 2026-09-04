import os
from typing import List, Dict, Any, Tuple, Optional
from .vector_db_service import get_vector_db_service
from .ingredient_format import ingredient_label

class EmbeddingService:
    """
    Service xử lý embedding và vector search qua ChromaDB (persistent)
    - Dùng mô hình sentence-transformers cục bộ (Default: keepitreal/vietnamese-sbert)
    - Lưu trữ lâu dài bằng ChromaDB
    """
    
    def __init__(self):
        self.vector_db = get_vector_db_service()
        self.recipe_data = []
        self.documents = []
        print(" [EmbeddingService] Initialized with local ChromaDB & SentenceTransformers")
    
    def get_embedding(self, text: str) -> List[float]:
        return self.vector_db.get_embedding(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self.vector_db.get_embeddings_batch(texts)
    
    def build_recipe_document(self, recipe: Dict[str, Any]) -> str:
        """
        Chuyển recipe dict thành document text để embedding
        """
        parts = []
        parts.append(f"Tên món: {recipe.get('name', 'Không rõ')}")
        
        if recipe.get('description'):
            parts.append(f"Mô tả: {recipe['description']}")
        
        ingredients = recipe.get('ingredients', [])
        if ingredients:
            ing_lines = []
            for ing in ingredients:
                if isinstance(ing, dict):
                    ing_lines.append(f"- {ingredient_label(ing)}")
                else:
                    ing_lines.append(f"- {ing}")
            parts.append("Nguyên liệu:\n" + "\n".join(ing_lines))
        
        if recipe.get('steps'):
            parts.append(f"Cách làm:\n{recipe['steps']}")
        
        return "\n\n".join(parts)
    
    def build_index(self, recipes: List[Dict[str, Any]]) -> None:
        """
        Đồng bộ danh sách recipes vào ChromaDB
        """
        if not recipes:
            print(" [EmbeddingService] No recipes to index")
            return
        
        self.recipe_data = recipes
        self.documents = [self.build_recipe_document(r) for r in recipes]
        
        print(f" [EmbeddingService] Syncing {len(recipes)} recipes with ChromaDB...")
        self.vector_db.add_recipes(recipes, self.documents)
    
    def search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, Dict[str, Any], str]]:
        """
        Tìm kiếm trong ChromaDB kết hợp lọc Metadata (Pre-filtering)
        
        Returns:
            List of (index, distance, recipe_data, document_text)
        """
        results = self.vector_db.query(query_text=query, top_k=top_k, where=where)
        
        search_results = []
        for res in results:
            res_id = res["id"]
            distance = res["distance"]
            doc_text = res["document"]
            
            # Tìm recipe tương ứng trong recipe_data hiện tại bằng ID
            recipe_info = None
            recipe_idx = -1
            for idx, r in enumerate(self.recipe_data):
                if str(r.get("id")) == res_id:
                    recipe_info = r
                    recipe_idx = idx
                    break
            
            # Fallback nếu không có trong recipe_data cục bộ (ví dụ: truy vấn trực tiếp DB)
            if recipe_info is None:
                # Tạo basic recipe info từ metadata
                meta = res["metadata"]
                recipe_info = {
                    "id": res_id,
                    "name": meta.get("name"),
                    "difficulty": meta.get("difficulty"),
                    "cook_time_minutes": meta.get("cook_time_minutes"),
                    "cuisine_type": meta.get("cuisine_type"),
                    "diet_tags": meta.get("diet_tags", []),
                    "description": "",
                    "steps": "",
                    "ingredients": []
                }
                recipe_idx = 9999  # Placeholder index
                
            search_results.append((
                recipe_idx,
                distance,
                recipe_info,
                doc_text
            ))
            
        return search_results
    
    def clear_index(self) -> None:
        """
        Xóa danh sách recipes in-memory (không xóa ChromaDB để giữ lưu trữ lâu dài)
        """
        self.recipe_data = []
        self.documents = []


# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
