import os
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from .embedding_service import get_embedding_service
from sentence_transformers import CrossEncoder

class RetrievalService:
    """
    Service retrieval kết hợp:
    1. Pre-filtering theo Metadata qua ChromaDB
    2. Semantic search qua ChromaDB
    3. Re-ranking sử dụng mô hình Cross-Encoder (BGE-Reranker) hoặc Rule-based overlap
    """
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.reranker = None
        
        try:
            print(f" [RetrievalService] Loading local re-ranker model: {self.reranker_model_name}...")
            self.reranker = CrossEncoder(self.reranker_model_name)
            print(" [RetrievalService] Re-ranker model loaded successfully!")
        except Exception as e:
            print(f" [RetrievalService] WARNING: Failed to load Cross-Encoder ({e}). Fallback to rule-based scoring.")
            
        print(" [RetrievalService] Initialized")
    
    def _normalize_ingredient(self, name: str) -> str:
        """
        Chuẩn hóa tên nguyên liệu để so sánh
        """
        return name.lower().strip()
    
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
        
        total = len(recipe_names)
        score = len(matched) / total if total > 0 else 0.0
        
        return matched, missing, score
    
    def build_query_text(self, user_ingredients: List[str]) -> str:
        ingredients_str = ", ".join(user_ingredients)
        return f"Tìm công thức món ăn với các nguyên liệu: {ingredients_str}"
        
    def _build_where_filter(self, preferences: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Xây dựng filter cho ChromaDB từ preferences
        """
        if not preferences:
            return None
            
        filters = []
        
        difficulty = preferences.get("difficulty")
        if difficulty:
            # Chuẩn hóa về 'De', 'Trung binh', 'Kho'
            if difficulty in ["Easy", "De"]:
                difficulty = "De"
            elif difficulty in ["Medium", "Trung binh"]:
                difficulty = "Trung binh"
            elif difficulty in ["Hard", "Kho"]:
                difficulty = "Kho"
            filters.append({"difficulty": {"$eq": difficulty}})
            
        cook_time_max = preferences.get("cook_time_max")
        if cook_time_max:
            try:
                filters.append({"cook_time_minutes": {"$lte": int(cook_time_max)}})
            except ValueError:
                pass
                
        cuisine_type = preferences.get("cuisine_type")
        if cuisine_type:
            filters.append({"cuisine_type": {"$eq": cuisine_type}})
            
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
            
        return {"$and": filters}

    def retrieve(
        self, 
        user_ingredients: List[str], 
        recipes: List[Dict[str, Any]], 
        top_k: int = 5,
        use_rerank: bool = True,
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top K recipes phù hợp nhất
        """
        if not recipes:
            return []
        
        # Step 1: Build index (đồng bộ vào ChromaDB)
        self.embedding_service.build_index(recipes)
        
        # Step 2: Xây dựng metadata pre-filter
        where_filter = self._build_where_filter(preferences)
        print(f" [RetrievalService] Pre-filtering with where clause: {where_filter}")
        
        # Step 3: Semantic search
        query = self.build_query_text(user_ingredients)
        
        # Lấy nhiều kết quả hơn để re-rank
        search_k = min(top_k * 3, len(recipes))
        search_results = self.embedding_service.search(query, top_k=search_k, where=where_filter)
        
        if not search_results:
            return []
            
        # Step 4: Post-filtering theo diet_tags trong python (vì tag lưu dạng mảng/comma-separated)
        filtered_results = []
        diet_tags = preferences.get("diet_tags") if preferences else None
        
        for idx, distance, recipe_data, doc_text in search_results:
            if diet_tags:
                required = set(self._normalize_ingredient(t) for t in diet_tags if t.strip())
                recipe_tags = set(self._normalize_ingredient(t) for t in recipe_data.get("diet_tags", []))
                
                # Bỏ qua nếu không khớp hết tất cả diet_tags yêu cầu
                if not required.issubset(recipe_tags):
                    continue
            
            # Tính toán ingredient overlap
            matched, missing, overlap_score = self._calculate_ingredient_overlap(
                user_ingredients, 
                recipe_data.get('ingredients', [])
            )
            
            # Distance là cosine distance (0-2 range, lower is better)
            # Chuyển đổi sang similarity: 1 - distance
            vector_similarity = max(0.0, 1.0 - distance)
            
            filtered_results.append({
                'index': idx,
                'recipe': recipe_data,
                'document': doc_text,
                'vector_distance': distance,
                'vector_similarity': vector_similarity,
                'matched_ingredients': matched,
                'missing_ingredients': missing,
                'overlap_score': overlap_score,
                'combined_score': vector_similarity
            })
            
        if not filtered_results:
            return []

        # Step 5: Re-ranking
        if use_rerank and self.reranker:
            try:
                print(f" [RetrievalService] Re-ranking {len(filtered_results)} candidates using local Cross-Encoder...")
                # Chuẩn bị cặp (query, document)
                pairs = [(query, r['document']) for r in filtered_results]
                
                # Dự đoán điểm mức độ liên quan (higher is better)
                rerank_scores = self.reranker.predict(pairs)
                
                # Chuẩn hóa rerank score và kết hợp với overlap score để tăng tính chính xác thực phẩm
                for r, score in zip(filtered_results, rerank_scores):
                    # Cross-encoder score có thể là âm hoặc dương tùy model, thường nằm quanh sigmoid
                    # Chuẩn hóa về 0-1
                    normalized_rerank = 1.0 / (1.0 + os.sys.float_info.epsilon if score < -20 else 1.0 / (1.0 + os.sys.float_info.epsilon) if score > 20 else 1.0 + os.sys.float_info.epsilon if score == 0 else 1.0 / (1.0 + os.sys.float_info.epsilon) if score is None else (1.0 / (1.0 + os.sys.float_info.epsilon) if score > 10 else 1.0 / (1.0 + os.sys.float_info.epsilon) if score < -10 else float(score)))
                    
                    # Sigmoid-like normalization
                    import math
                    try:
                        sig_score = 1.0 / (1.0 + math.exp(-score))
                    except OverflowError:
                        sig_score = 0.0 if score < 0 else 1.0
                        
                    # Ưu tiên món có nguyên liệu khớp nhiều hơn
                    r['combined_score'] = 0.6 * r['overlap_score'] + 0.4 * sig_score
                    
                print(" [RetrievalService] Re-ranking done.")
            except Exception as e:
                print(f" [RetrievalService] Re-ranking failed ({e}). Falling back to rule-based scoring.")
                self._fallback_scoring(filtered_results)
        else:
            self._fallback_scoring(filtered_results)
            
        # Sắp xếp giảm dần theo combined_score
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return filtered_results[:top_k]
        
    def _fallback_scoring(self, results):
        for r in results:
            # Combine overlap score (60%) và vector similarity (40%)
            r['combined_score'] = 0.6 * r['overlap_score'] + 0.4 * r['vector_similarity']
            
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
