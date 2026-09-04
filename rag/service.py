"""
High-Level RAG Service.
Coordinates Food Database, Hybrid Retriever, Prompt Builder, and LLM Client.
Provides deterministic fallback to top retrieved recipe if LLM is unavailable.
"""

import logging
from typing import List, Dict, Any, Optional

from .food_db import get_food_database
from .retriever import HybridRetriever
from .prompt_builder import build_recipe_suggestion_prompt, SYSTEM_PROMPT
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)


class RAGService:
    """End-to-End RAG Service for Vietnamese Recipe Suggestion."""

    def __init__(self):
        self.food_db = get_food_database()
        self.retriever = HybridRetriever(self.food_db)
        self.llm_client = get_llm_client()

    def suggest_recipe(
        self,
        user_ingredients: List[str],
        preferences: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main RAG pipeline:
        1. Query database & retrieve Top-K relevant recipes
        2. If no recipes match, return graceful no-result response
        3. Build strict zero-hallucination prompt
        4. Call LLM to summarize and format instructions
        5. If LLM fails, fallback gracefully to retrieved recipe metadata
        """
        logger.info(f"[RAGService] Processing request with {len(user_ingredients)} ingredients")

        # Step 1: Retrieval
        retrieved = self.retriever.retrieve(
            user_ingredients=user_ingredients,
            query=query,
            preferences=preferences,
            top_k=top_k
        )

        # Step 2: Handle No-Result
        if not retrieved:
            logger.info("[RAGService] No matching recipes found in database.")
            return {
                "best_recipe": None,
                "recipe_id": None,
                "reason": "Cơ sở dữ liệu 96 món ăn hiện chưa có công thức phù hợp với nguyên liệu của bạn.",
                "matched_ingredients": [],
                "missing_ingredients": [],
                "substitutions": [],
                "instructions": [],
                "alternative_recipes": []
            }

        # Step 3: Build Prompt
        prompt = build_recipe_suggestion_prompt(user_ingredients, retrieved, query=query)

        # Step 4: Call LLM
        try:
            llm_response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2
            )
            if llm_response and llm_response.get("best_recipe"):
                # Enrich best recipe with metadata from retrieved item
                best_id = str(llm_response.get("recipe_id", ""))
                matched_recipe = next((r["recipe"] for r in retrieved if r["recipe"]["id"] == best_id), retrieved[0]["recipe"])

                llm_response.setdefault("image_url", matched_recipe.get("image_url"))
                llm_response.setdefault("cook_time_minutes", matched_recipe.get("cook_time_minutes"))
                llm_response.setdefault("difficulty", matched_recipe.get("difficulty"))
                llm_response.setdefault("servings", matched_recipe.get("servings"))
                return llm_response
        except Exception as e:
            logger.warning(f"[RAGService] LLM generation failed ({e}), using rule-based fallback.")

        # Step 5: Deterministic Fallback directly from retrieved DB records
        top_match = retrieved[0]
        recipe = top_match["recipe"]

        instructions = [
            f"Bước {step.get('step_number')}: {step.get('title', '')} - {step.get('description')}"
            for step in recipe.get("instructions", [])
        ]

        alternatives = [
            {
                "id": alt["recipe"]["id"],
                "name": alt["recipe"]["name"],
                "matched_count": len(alt["matched_ingredients"]),
                "missing_count": len(alt["missing_ingredients"])
            }
            for alt in retrieved[1:4]
        ]

        return {
            "best_recipe": recipe["name"],
            "recipe_id": recipe["id"],
            "reason": f"Khớp {len(top_match['matched_ingredients'])}/{len(recipe.get('ingredients', []))} nguyên liệu từ cơ sở dữ liệu.",
            "image_url": recipe.get("image_url"),
            "cook_time_minutes": recipe.get("cook_time_minutes"),
            "difficulty": recipe.get("difficulty"),
            "servings": recipe.get("servings"),
            "matched_ingredients": top_match["matched_ingredients"],
            "missing_ingredients": top_match["missing_ingredients"],
            "substitutions": [],
            "instructions": instructions,
            "alternative_recipes": alternatives
        }


_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """Singleton getter for RAGService."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
