"""
Retrieval Service - Kết hợp Vector Search và Rule-based Scoring
Tìm kiếm recipes phù hợp nhất dựa trên nguyên liệu user có
"""

from typing import List, Dict, Any, Tuple
from .embedding_service import get_embedding_service


class RetrievalService:
    """
    Service retrieval kết hợp:
    1. Semantic search qua FAISS (embedding similarity)
    2. Rule-based scoring theo ingredient overlap
    """
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        print(" [RetrievalService] Initialized")
    
    def _normalize_ingredient(self, name: str) -> str:
        """
        Chuẩn hóa tên nguyên liệu để so sánh
        Ví dụ: "Thịt gà" -> "thit ga"
        """
        import unicodedata
        # Lowercase
        name = name.lower().strip()
        # Remove diacritics (dấu tiếng Việt)
        # Giữ nguyên để matching tốt hơn với tiếng Việt
        return name
    
    def _calculate_ingredient_overlap(
        self, 
        user_ingredients: List[str], 
        recipe_ingredients: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], float]:
        """
        Tính độ trùng khớp nguyên liệu
        
        Returns:
            (matched_ingredients, missing_ingredients, overlap_score)
        """
        user_set = set(self._normalize_ingredient(ing) for ing in user_ingredients)
        
        recipe_names = [ing.get('name', '') for ing in recipe_ingredients]
        recipe_set = set(self._normalize_ingredient(name) for name in recipe_names)
        
        # Tìm nguyên liệu khớp (dùng substring matching cho flexibility)
        matched = []
        missing = []
        
        for recipe_ing in recipe_names:
            recipe_ing_norm = self._normalize_ingredient(recipe_ing)
            found = False
            
            for user_ing in user_ingredients:
                user_ing_norm = self._normalize_ingredient(user_ing)
                
                # Exact match hoặc substring match
                if (user_ing_norm == recipe_ing_norm or 
                    user_ing_norm in recipe_ing_norm or 
                    recipe_ing_norm in user_ing_norm):
                    matched.append(recipe_ing)
                    found = True
                    break
            
            if not found:
                missing.append(recipe_ing)
        
        # Score = matched / total recipe ingredients
        total = len(recipe_names)
        score = len(matched) / total if total > 0 else 0.0
        
        return matched, missing, score
    
    def build_query_text(self, user_ingredients: List[str]) -> str:
        """
        Tạo query text từ danh sách nguyên liệu
        Dùng cho semantic search
        """
        ingredients_str = ", ".join(user_ingredients)
        return f"Tìm công thức món ăn với các nguyên liệu: {ingredients_str}"
    
    def retrieve(
        self, 
        user_ingredients: List[str], 
        recipes: List[Dict[str, Any]], 
        top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top K recipes phù hợp nhất
        
        Flow:
        1. Build FAISS index cho recipes
        2. Semantic search với query từ user ingredients
        3. Rerank theo ingredient overlap score
        4. Trả về top K kết quả
        
        Returns:
            List of dicts với thông tin recipe và scores
        """
        if not recipes:
            return []
        
        # Step 1: Build index
        self.embedding_service.build_index(recipes)
        
        # Step 2: Semantic search
        query = self.build_query_text(user_ingredients)
        
        # Lấy nhiều hơn top_k để rerank
        search_k = min(top_k * 2, len(recipes))
        search_results = self.embedding_service.search(query, top_k=search_k)
        
        if not search_results:
            return []
        
        # Step 3: Calculate ingredient overlap và rerank
        results_with_scores = []
        
        for idx, distance, recipe_data, doc_text in search_results:
            matched, missing, overlap_score = self._calculate_ingredient_overlap(
                user_ingredients, 
                recipe_data.get('ingredients', [])
            )
            
            # Combine scores:
            # - Vector distance (lower is better) -> convert to similarity
            # - Ingredient overlap (higher is better)
            # Normalize distance to 0-1 range (approximate)
            vector_similarity = 1.0 / (1.0 + distance)
            
            # Combined score (weighted average)
            # Ưu tiên ingredient overlap hơn semantic similarity
            if use_rerank:
                combined_score = 0.6 * overlap_score + 0.4 * vector_similarity
            else:
                combined_score = vector_similarity
            
            results_with_scores.append({
                'index': idx,
                'recipe': recipe_data,
                'document': doc_text,
                'vector_distance': distance,
                'vector_similarity': vector_similarity,
                'matched_ingredients': matched,
                'missing_ingredients': missing,
                'overlap_score': overlap_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        results_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top K
        return results_with_scores[:top_k]
    
    def clear(self) -> None:
        """Clear embedding index"""
        self.embedding_service.clear_index()


# Singleton
_retrieval_service = None

def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
