import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

class VectorDBService:
    def __init__(self):
        self.db_dir = os.getenv("VECTOR_DB_DIR", "data/chroma_db")
        self.collection_name = "recipes"
        
        # Create DB directory if not exists
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize chromadb
        self.client = chromadb.PersistentClient(path=self.db_dir)
        
        # Load local embedding model
        self.model_name = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
        print(f" [VectorDBService] Loading local embedding model: {self.model_name}...")
        self.embedding_model = SentenceTransformer(self.model_name)
        print(" [VectorDBService] Embedding model loaded successfully!")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # cosine similarity
        )
        print(f" [VectorDBService] ChromaDB collection '{self.collection_name}' initialized.")

    def get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, convert_to_numpy=True).tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def add_recipes(self, recipes: List[Dict[str, Any]], documents: List[str]):
        """
        Add list of recipes to ChromaDB.
        Each recipe should have metadata:
          - id
          - name
          - difficulty
          - cook_time_minutes
          - cuisine_type
          - diet_tags (stored as comma-separated string because ChromaDB only supports simple types in metadata)
        """
        if not recipes or not documents:
            return
        
        ids = [str(r["id"]) for r in recipes]
        metadatas = []
        for r in recipes:
            tags = r.get("diet_tags")
            tags_str = ""
            if isinstance(tags, list):
                tags_str = ",".join(tags)
            elif isinstance(tags, str):
                tags_str = tags
                
            metadatas.append({
                "id": str(r["id"]),
                "name": str(r.get("name", "")),
                "difficulty": str(r.get("difficulty", "De")),
                "cook_time_minutes": int(r.get("cook_time_minutes") or 0),
                "cuisine_type": str(r.get("cuisine_type") or ""),
                "diet_tags": tags_str
            })
            
        embeddings = self.get_embeddings_batch(documents)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f" [VectorDBService] Added {len(recipes)} recipes to ChromaDB.")

    def query(self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for relevant recipes with optional metadata filter.
        """
        query_embedding = self.get_embedding(query_text)
        
        # Ensure top_k is valid and bounded
        k = max(1, top_k)
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where
        )
        
        # Format results to match search expectations
        formatted = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []
            
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        
        for i in range(len(ids)):
            # diet_tags back to list
            meta = dict(metadatas[i])
            if "diet_tags" in meta and meta["diet_tags"]:
                meta["diet_tags"] = meta["diet_tags"].split(",")
            else:
                meta["diet_tags"] = []
                
            formatted.append({
                "id": ids[i],
                "distance": distances[i],
                "metadata": meta,
                "document": documents[i]
            })
            
        return formatted

    def clear(self):
        """Reset collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(" [VectorDBService] ChromaDB collection cleared.")

_vector_db_service = None

def get_vector_db_service() -> VectorDBService:
    global _vector_db_service
    if _vector_db_service is None:
        _vector_db_service = VectorDBService()
    return _vector_db_service
