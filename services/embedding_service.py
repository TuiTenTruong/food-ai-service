"""
Embedding Service - Xử lý OpenAI Embeddings và FAISS Vector Store
Tạo embeddings cho recipes và thực hiện vector search
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """
    Service xử lý embedding và vector search
    - Dùng OpenAI text-embedding-3-small
    - Index bằng FAISS (in-memory)
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        
        # FAISS index sẽ được build động mỗi request
        self.index = None
        self.embedding_matrix = None
        self.search_backend = None
        self.documents = []
        self.recipe_data = []
        
        print(" [EmbeddingService] Initialized with OpenAI embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Lấy embedding vector cho một đoạn text
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Lấy embeddings cho nhiều texts cùng lúc (hiệu quả hơn)
        OpenAI cho phép batch tối đa ~8000 tokens
        """
        if not texts:
            return []
        
        # Chia batch nếu quá nhiều
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def build_recipe_document(self, recipe: Dict[str, Any]) -> str:
        """
        Chuyển recipe dict thành document text để embedding
        
        Input recipe format:
        {
            "id": "1",
            "name": "Trứng chiên cà chua",
            "description": "Món ăn đơn giản...",
            "steps": "1. Đánh trứng...",
            "ingredients": [{"name": "trứng", "amount": "2 quả"}, ...]
        }
        """
        parts = []
        
        # Tên món
        parts.append(f"Tên món: {recipe.get('name', 'Không rõ')}")
        
        # Mô tả
        if recipe.get('description'):
            parts.append(f"Mô tả: {recipe['description']}")
        
        # Nguyên liệu
        ingredients = recipe.get('ingredients', [])
        if ingredients:
            ing_lines = []
            for ing in ingredients:
                name = ing.get('name', '')
                amount = ing.get('amount', '')
                if amount:
                    ing_lines.append(f"- {name} ({amount})")
                else:
                    ing_lines.append(f"- {name}")
            parts.append("Nguyên liệu:\n" + "\n".join(ing_lines))
        
        # Cách làm
        if recipe.get('steps'):
            parts.append(f"Cách làm:\n{recipe['steps']}")
        
        return "\n\n".join(parts)
    
    def build_index(self, recipes: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index từ danh sách recipes
        
        Flow:
        1. Convert mỗi recipe thành document text
        2. Tạo embeddings cho tất cả documents
        3. Build FAISS index
        """
        if not recipes:
            print(" [EmbeddingService] No recipes to index")
            return
        
        print(f" [EmbeddingService] Building index for {len(recipes)} recipes...")
        
        # Lưu recipe data để truy xuất sau
        self.recipe_data = recipes
        
        # Build documents
        self.documents = [self.build_recipe_document(r) for r in recipes]
        
        # Lấy embeddings
        print(" [EmbeddingService] Getting embeddings from OpenAI...")
        embeddings = self.get_embeddings_batch(self.documents)
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype='float32')

        # Ưu tiên FAISS, fallback sang NumPy nếu faiss import/runtime bị lỗi.
        try:
            import faiss

            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.index.add(embeddings_np)
            self.embedding_matrix = None
            self.search_backend = "faiss"
            print(f" [EmbeddingService] FAISS index built with {self.index.ntotal} vectors")
        except Exception as e:
            self.index = None
            self.embedding_matrix = embeddings_np
            self.search_backend = "numpy"
            print(f" [EmbeddingService] FAISS unavailable ({e}). Fallback to NumPy search")
            print(f" [EmbeddingService] NumPy index built with {len(self.embedding_matrix)} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any], str]]:
        """
        Tìm kiếm top K recipes phù hợp với query
        
        Returns:
            List of (index, distance, recipe_data, document_text)
        """
        if self.search_backend == "faiss" and (self.index is None or self.index.ntotal == 0):
            return []

        if self.search_backend == "numpy" and (self.embedding_matrix is None or len(self.embedding_matrix) == 0):
            return []
        
        # Embedding query
        query_embedding = self.get_embedding(query)
        query_np = np.array([query_embedding], dtype='float32')
        
        # Search
        if self.search_backend == "faiss":
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_np, k)
        else:
            all_distances = np.sum((self.embedding_matrix - query_np) ** 2, axis=1)
            sorted_indices = np.argsort(all_distances)
            k = min(top_k, len(sorted_indices))
            selected_indices = sorted_indices[:k]
            distances = np.array([all_distances[selected_indices]], dtype='float32')
            indices = np.array([selected_indices], dtype='int64')
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.recipe_data):
                results.append((
                    int(idx),
                    float(dist),
                    self.recipe_data[idx],
                    self.documents[idx]
                ))
        
        return results
    
    def clear_index(self) -> None:
        """Xóa index hiện tại"""
        self.index = None
        self.embedding_matrix = None
        self.search_backend = None
        self.documents = []
        self.recipe_data = []


# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Lấy singleton instance của EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
