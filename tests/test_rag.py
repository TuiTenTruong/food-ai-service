"""
Unit and Integration Tests for RAG Architecture.
Tests criteria:
- Food database integrity (96 recipes)
- Hybrid retrieval with exact match & semantic ranking
- Top-K configuration and context bounding
- Zero-hallucination on no-result queries
- Deterministic fallback on LLM failure
- Agent 3 Task 4 specific culinary queries
"""

import pytest
from rag import (
    FoodDatabase,
    get_food_database,
    HybridRetriever,
    build_recipe_suggestion_prompt,
    RAGService,
    get_rag_service,
)


class TestFoodDatabase:
    """Tests for the curated 96 Vietnamese recipes database."""

    def test_database_loading_and_count(self):
        db = get_food_database()
        recipes = db.get_all_recipes()
        assert len(recipes) == 96, f"Expected 96 recipes, got {len(recipes)}"

    def test_recipe_schema_integrity(self):
        db = get_food_database()
        for recipe in db.get_all_recipes():
            assert "id" in recipe
            assert "name" in recipe and len(recipe["name"]) > 0
            assert "ingredients" in recipe and len(recipe["ingredients"]) > 0
            assert "instructions" in recipe and len(recipe["instructions"]) > 0

    def test_search_by_name(self):
        db = get_food_database()
        results = db.search_by_name("thịt bò")
        assert len(results) >= 1
        assert any("thịt bò" in r["name"].lower() or "bò" in r["name"].lower() for r in results)


class TestRAGRetriever:
    """Tests for hybrid retrieval logic and Top-K constraints."""

    def test_retrieval_egg_and_tomato(self):
        db = get_food_database()
        retriever = HybridRetriever(db)

        retrieved = retriever.retrieve(
            user_ingredients=["trứng", "cà chua"],
            top_k=5
        )
        assert len(retrieved) > 0
        assert len(retrieved) <= 5

        # Best match should contain both or at least one
        top_recipe = retrieved[0]["recipe"]
        matched = retrieved[0]["matched_ingredients"]
        assert len(matched) >= 1
        assert "Cà chua nấu trứng" in [r["recipe"]["name"] for r in retrieved]

    def test_top_k_constraint(self):
        db = get_food_database()
        retriever = HybridRetriever(db)

        for k in [1, 3, 5]:
            results = retriever.retrieve(user_ingredients=["thịt lợn", "hành tây"], top_k=k)
            assert len(results) <= k


class TestZeroHallucinationAndNoResult:
    """Tests that non-existent ingredients or dishes gracefully return no-result."""

    def test_foreign_ingredients_no_result(self):
        """Ingredients completely absent from Vietnamese database must return no recipes."""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=["socola đen nguyên chất", "kem tươi whipping cream", "phô mai parmesan"]
        )
        assert result["best_recipe"] is None
        assert result["recipe_id"] is None
        assert "chưa có công thức phù hợp" in result["reason"]
        assert len(result["matched_ingredients"]) == 0

    def test_nonexistent_recipe_query(self):
        """Query for a non-existent recipe like Pizza or Sushi returns no-result."""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=[],
            query="Cách làm sushi cá hồi Nhật Bản truyền thống?"
        )
        assert result["best_recipe"] is None
        assert result["recipe_id"] is None


class TestPromptBuilderConstraints:
    """Ensure context bounds and strict instructions."""

    def test_prompt_context_length(self):
        db = get_food_database()
        retriever = HybridRetriever(db)
        retrieved = retriever.retrieve(user_ingredients=["trứng", "cà chua"], top_k=3)

        prompt = build_recipe_suggestion_prompt(["trứng", "cà chua"], retrieved)
        assert "CÔNG THỨC TRÍCH XUẤT TỪ DATABASE:" in prompt
        assert "Cà chua nấu trứng" in prompt
        # Must not contain all 96 recipes
        assert len(prompt) < 15000, "Prompt must be strictly bounded in length"


class TestAgent3RequiredQueries:
    """Test standard evaluation queries specified by Agent 3 Task 4."""

    def test_query_pho_bo(self):
        """Query: 'Cách làm phở bò?'"""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=["thịt bò", "bánh phở"],
            query="Cách làm phở bò?"
        )
        assert result["best_recipe"] is not None
        assert "phở" in result["best_recipe"].lower() or "bò" in result["best_recipe"].lower()

    def test_query_chicken_low_calorie(self):
        """Query: 'Món nào có thịt gà và ít calo?'"""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=["thịt gà"],
            preferences={"difficulty": "Dễ"},
            query="Món nào có thịt gà và ít calo?"
        )
        assert result["best_recipe"] is not None
        assert "gà" in result["best_recipe"].lower()

    def test_query_egg_tomato(self):
        """Query: 'Tôi có trứng và cà chua, có thể làm món gì?'"""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=["trứng", "cà chua"],
            query="Tôi có trứng và cà chua, có thể làm món gì?"
        )
        assert result["best_recipe"] is not None
        assert len(result["matched_ingredients"]) >= 1

    def test_query_non_existent_dish(self):
        """Query: 'Cách làm một món không tồn tại trong database?'"""
        service = get_rag_service()
        result = service.suggest_recipe(
            user_ingredients=["phô mai mozzarella", "lá oregano", "bột bánh pizza"],
            query="Cách làm bánh pizza hải sản phong cách Ý?"
        )
        # Should gracefully decline without hallucinating
        assert result["best_recipe"] is None
        assert result["recipe_id"] is None
