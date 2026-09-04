"""
Food Database Access Layer.
Manages the 96 traditional Vietnamese recipes from NCKH research dataset.
Supports dual-source:
1. Direct connection to MySQL / MariaDB via DATABASE_URL if available.
2. Local fallback to data_book.json (fully self-contained, perfect for Modal serverless).
"""

import os
import json
import logging
import unicodedata
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def remove_accents(text: str) -> str:
    """Remove Vietnamese diacritics for robust fuzzy matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", text).lower().strip()


class FoodDatabase:
    """
    Central repository for food recipe data.
    Ensures zero hallucination by serving as the single source of truth.
    """

    def __init__(self, json_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        candidates = [
            json_path,
            os.getenv("FOOD_DB_PATH"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_book.json"),
            os.path.join(base_dir, "rag", "data_book.json"),
            os.path.join(base_dir, "ingredient_task", "data_book.json"),
            os.path.join(os.path.dirname(base_dir), "be_nckh", "app", "schemas", "ingredient_task", "data_book.json"),
            os.path.join(os.path.dirname(base_dir), "be_nckh", "data", "data_book.json"),
        ]
        
        self.json_path = next((p for p in candidates if p and os.path.exists(p)), None)
        self._recipes: List[Dict[str, Any]] = []
        self._recipes_by_id: Dict[str, Dict[str, Any]] = {}
        self.load_data()

    def load_from_mysql(self) -> bool:
        """
        Load recipes and ingredients directly from MySQL database if DATABASE_URL is available
        or reachable on default localhost:3306.
        Returns True if successfully loaded from MySQL, False otherwise.
        """
        db_url = os.getenv("DATABASE_URL") or "mysql+pymysql://root:@localhost:3306/nckh"

        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
            with engine.connect() as conn:
                # Query recipes
                recipes_res = conn.execute(text("""
                    SELECT id, name, description, image_url, cook_time_minutes, difficulty, servings, 
                           cuisine_type, diet_tags, is_featured, total_favorites, total_views, source
                    FROM recipes
                """)).mappings().all()

                if not recipes_res:
                    return False

                # Query ingredients with LEFT JOIN to safely handle all relations
                ri_res = conn.execute(text("""
                    SELECT ri.recipe_id, COALESCE(i.name, ri.ingredient_id) AS name, 
                           ri.quantity, ri.unit, ri.is_optional, i.category_id
                    FROM recipe_ingredients ri
                    LEFT JOIN ingredients i ON ri.ingredient_id = i.id
                    ORDER BY ri.recipe_id, ri.sort_order
                """)).mappings().all()

                # Query steps
                steps_res = conn.execute(text("""
                    SELECT recipe_id, step_number, title, description, tip
                    FROM recipe_steps
                    ORDER BY recipe_id, step_number
                """)).mappings().all()

                # Group by recipe_id
                ingredients_by_recipe = {}
                for row in ri_res:
                    rid = str(row["recipe_id"])
                    raw_name = row["name"] or ""
                    ingredients_by_recipe.setdefault(rid, []).append({
                        "name": raw_name,
                        "normalized_name": remove_accents(raw_name),
                        "quantity": row["quantity"] or "",
                        "unit": row["unit"] or "",
                        "is_optional": bool(row["is_optional"]),
                        "category_id": row["category_id"] or ""
                    })

                steps_by_recipe = {}
                for row in steps_res:
                    rid = str(row["recipe_id"])
                    step_num = row["step_number"]
                    title = row["title"]
                    desc = row["description"] or ""
                    tip = row["tip"]
                    step_str = f"Bước {step_num}: "
                    if title:
                        step_str += f"**{title}** - "
                    step_str += desc
                    if tip:
                        step_str += f" (Mẹo: {tip})"
                    steps_by_recipe.setdefault(rid, []).append({
                        "step_number": step_num,
                        "title": title,
                        "description": desc,
                        "tip": tip,
                        "text": step_str
                    })

                normalized_recipes = []
                for r in recipes_res:
                    rid = str(r["id"])
                    r_steps = steps_by_recipe.get(rid, [])
                    recipe_obj = {
                        "id": rid,
                        "name": r["name"],
                        "normalized_name": remove_accents(r["name"]),
                        "description": r["description"] or "",
                        "image_url": r["image_url"] or "",
                        "cook_time_minutes": r["cook_time_minutes"] or 30,
                        "difficulty": r["difficulty"] or "easy",
                        "servings": r["servings"] or 2,
                        "cuisine_type": r["cuisine_type"] or "Vietnamese",
                        "diet_tags": json.loads(r["diet_tags"]) if isinstance(r["diet_tags"], str) else (r["diet_tags"] or []),
                        "ingredients": ingredients_by_recipe.get(rid, []),
                        "instructions": r_steps,
                        "steps_text": "\n".join(s["text"] for s in r_steps),
                        "is_featured": bool(r.get("is_featured", 0)),
                        "total_favorites": r.get("total_favorites") or 0,
                        "total_views": r.get("total_views") or 0,
                        "source": "mysql"
                    }
                    normalized_recipes.append(recipe_obj)

                self._recipes = normalized_recipes
                self._recipes_by_id = {r["id"]: r for r in self._recipes}
                logger.info(f"Loaded {len(self._recipes)} recipes directly from MySQL database.")
                return True
        except Exception as e:
            logger.info(f"MySQL not available or offline ({e}), fallback to JSON dataset.")
            return False

    def load_data(self) -> None:
        """Load and normalize recipe records from MySQL with local JSON fallback."""
        # 1. Try loading directly from MySQL database first
        if self.load_from_mysql():
            return

        # 2. Fallback to local JSON (self-contained, optimal for Modal serverless)
        if self.json_path and os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                normalized_recipes = []
                for idx, item in enumerate(raw_data, 1):
                    recipe_id = item.get("id") or f"recipe-book-{idx:04d}"
                    
                    # Flatten steps to text for easy retrieval context
                    steps_list = []
                    instructions = item.get("instructions", [])
                    for step in instructions:
                        step_num = step.get("step_number", len(steps_list) + 1)
                        title = step.get("title")
                        desc = step.get("description", "")
                        tip = step.get("tip")
                        step_str = f"Bước {step_num}: "
                        if title:
                            step_str += f"**{title}** - "
                        step_str += desc
                        if tip:
                            step_str += f" (Mẹo: {tip})"
                        steps_list.append(step_str)

                    recipe_obj = {
                        "id": str(recipe_id),
                        "name": item.get("name", ""),
                        "normalized_name": remove_accents(item.get("name", "")),
                        "description": item.get("description", ""),
                        "image_url": item.get("image_url", ""),
                        "cook_time_minutes": item.get("cook_time_minutes", 30),
                        "difficulty": item.get("difficulty", "easy"),
                        "servings": item.get("servings", 2),
                        "cuisine_type": item.get("cuisine_type", "Vietnamese"),
                        "diet_tags": item.get("diet_tags", []),
                        "ingredients": [
                            {
                                "name": ing.get("name", ""),
                                "normalized_name": remove_accents(ing.get("name", "")),
                                "quantity": ing.get("quantity", ""),
                                "unit": ing.get("unit", ""),
                                "is_optional": ing.get("is_optional", False),
                                "category_id": ing.get("category_id", "")
                            }
                            for ing in item.get("ingredients", [])
                        ],
                        "instructions": instructions,
                        "steps_text": "\n".join(steps_list),
                        "source": item.get("source", "")
                    }
                    normalized_recipes.append(recipe_obj)

                self._recipes = normalized_recipes
                self._recipes_by_id = {r["id"]: r for r in self._recipes}
                logger.info(f"Loaded {len(self._recipes)} recipes from {self.json_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load recipes from JSON: {e}")

        logger.warning("No recipes loaded into FoodDatabase!")

    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Return all available recipes in database."""
        return list(self._recipes)

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Find recipe by ID."""
        return self._recipes_by_id.get(str(recipe_id))

    def search_by_name(self, query: str) -> List[Dict[str, Any]]:
        """Search recipes by name with diacritic-insensitive matching."""
        if not query:
            return []
        norm_q = remove_accents(query)
        return [r for r in self._recipes if norm_q in r["normalized_name"]]

    def filter_recipes(
        self,
        difficulty: Optional[str] = None,
        max_cook_time: Optional[int] = None,
        cuisine_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter recipes based on user constraints."""
        results = self._recipes

        if difficulty:
            diff_clean = remove_accents(difficulty.lower().strip())
            # Map Vietnamese terms
            diff_map = {
                "de": "easy",
                "easy": "easy",
                "trung binh": "medium",
                "medium": "medium",
                "kho": "hard",
                "hard": "hard"
            }
            mapped_diff = diff_map.get(diff_clean, diff_clean)
            results = [r for r in results if r.get("difficulty", "").lower() == mapped_diff]

        if max_cook_time is not None:
            results = [r for r in results if r.get("cook_time_minutes", 0) <= max_cook_time]

        if cuisine_type:
            c_clean = cuisine_type.lower().strip()
            results = [r for r in results if c_clean in r.get("cuisine_type", "").lower()]

        return results


# Global singleton instance
_db_instance: Optional[FoodDatabase] = None

def get_food_database() -> FoodDatabase:
    """Retrieve singleton FoodDatabase instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FoodDatabase()
    return _db_instance
