"""
Hybrid Recipe Retriever.
Combines exact & synonym ingredient overlap with semantic text search.
Filters by user preferences (difficulty, cook time, cuisine).
Limits context strictly to Top-K items to avoid prompt stuffing.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from .food_db import get_food_database, remove_accents, FoodDatabase

logger = logging.getLogger(__name__)

# Common Vietnamese food synonyms (with diacritics)
SYNONYMS = [
    {"thịt lợn", "thịt heo", "nạc lợn", "nạc heo", "thịt ba chỉ", "ba chỉ"},
    {"khổ qua", "mướp đắng"},
    {"dưa leo", "dưa chuột"},
    {"củ đậu", "củ sắn"},
    {"bắp ngô", "bắp", "ngô"},
    {"thịt bò", "bò", "thịt bò nạc", "bắp bò"},
    {"thịt gà", "gà", "thịt ức gà", "ức gà", "đùi gà", "gà ta"},
    {"đậu hũ", "đậu phụ", "tàu hũ"},
    {"súp lơ xanh", "bông cải xanh"},
    {"súp lơ trắng", "bông cải trắng"},
    {"mướp hương", "mướp"},
    {"trứng gà", "trứng vịt", "trứng"},
    {"cà chua", "cà chua bi"},
    {"hành lá", "hành hoa"},
]


def are_synonyms(a: str, b: str) -> bool:
    """Check if two ingredient strings are culinary synonyms."""
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return True

    # 1. Check accented synonyms
    for syn_set in SYNONYMS:
        if a_clean in syn_set and b_clean in syn_set:
            return True

    # 2. Multi-word phrase matching with accents preserved (e.g. "thịt bò" in "thịt bò phi lê")
    words_a = a_clean.split()
    words_b = b_clean.split()
    if len(words_a) >= 2 and len(words_b) >= 2:
        set_a, set_b = set(words_a), set(words_b)
        if set_a == set_b or set_a.issubset(set_b) or set_b.issubset(set_a):
            return True

    # 3. Normalized comparison (without accents)
    norm_a = remove_accents(a_clean)
    norm_b = remove_accents(b_clean)
    if norm_a == norm_b:
        # Prevent false equality for distinct accented short words (e.g. "bơ" vs "bò", "cá" vs "cà")
        if (a_clean != norm_a or b_clean != norm_b) and len(norm_a) <= 3 and a_clean != b_clean:
            return False
        return True

    # 4. Synonyms with normalized checking
    for syn_set in SYNONYMS:
        norm_set = {remove_accents(s) for s in syn_set}
        if norm_a in norm_set and norm_b in norm_set:
            return True

    return False


class HybridRetriever:
    """
    Retriever coordinating exact ingredient overlap and semantic scoring.
    """

    def __init__(self, food_db: Optional[FoodDatabase] = None):
        self.food_db = food_db or get_food_database()
        self.default_top_k = int(os.getenv("RAG_TOP_K", "5"))

    def _match_ingredients(
        self, user_ingredients: List[str], recipe_ingredients: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], float]:
        """
        Calculate matched and missing ingredients for a recipe.
        Returns: (matched_ingredients, missing_ingredients, overlap_ratio)
        """
        matched = []
        missing = []

        recipe_names = [ing["name"] for ing in recipe_ingredients]

        for r_name in recipe_names:
            is_matched = False
            for u_name in user_ingredients:
                if are_synonyms(u_name, r_name):
                    is_matched = True
                    matched.append(r_name)
                    break
            if not is_matched:
                missing.append(r_name)

        total_ingredients = len(recipe_names)
        overlap_score = len(matched) / total_ingredients if total_ingredients > 0 else 0.0
        return matched, missing, overlap_score

    def _calculate_text_relevance(self, query: str, recipe: Dict[str, Any]) -> Tuple[float, int]:
        """
        Calculate word-level semantic relevance between query and recipe.
        Uses accented word matching to avoid false positives (e.g. cá vs cà).
        Returns (relevance_score, title_matches_count).
        """
        if not query:
            return 0.0, 0

        # Accented tokenization to preserve semantic distinction (cá != cà, bò != bơ)
        query_terms = set(query.lower().split())
        stop_words = {
            "và", "có", "thể", "làm", "món", "gì", "nấu", "như", "nào", "cách",
            "cho", "tôi", "mình", "một", "những", "các", "ít", "nhiều", "trong",
            "truyền", "thống", "hương", "vị", "bữa", "cơm", "gia", "đình",
            "ngon", "quen", "thuộc", "phù", "hợp", "đơn", "giản", "đậm", "đà"
        }
        filtered_terms = query_terms - stop_words
        if not filtered_terms:
            return 0.0, 0

        title_terms = set(recipe["name"].lower().split()) - stop_words
        target_text = f"{recipe['description']} {' '.join(recipe.get('diet_tags', []))}"
        desc_terms = set(target_text.lower().split()) - stop_words

        title_matches = filtered_terms.intersection(title_terms)
        desc_matches = filtered_terms.intersection(desc_terms) - title_matches

        # Score calculation
        score = (len(title_matches) * 2.5 + len(desc_matches) * 0.5) / (len(filtered_terms) * 1.5)
        return min(round(score, 4), 1.0), len(title_matches)

    def retrieve(
        self,
        user_ingredients: Optional[List[str]] = None,
        query: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top K recipes strictly based on relevance and overlap.
        When ingredients are given, at least 1 ingredient must match to be considered relevant.
        """
        k = top_k or self.default_top_k
        user_ings = [i.strip() for i in (user_ingredients or []) if i.strip()]

        # 1. Apply metadata pre-filters
        prefs = preferences or {}
        candidate_recipes = self.food_db.filter_recipes(
            difficulty=prefs.get("difficulty"),
            max_cook_time=prefs.get("cook_time_max"),
            cuisine_type=prefs.get("cuisine_type")
        )

        logger.info(
            f"[RAG Retriever] Initial candidate recipes: {len(candidate_recipes)} "
            f"for user_ingredients={user_ings}, query='{query}'"
        )

        scored_candidates = []
        for recipe in candidate_recipes:
            matched, missing, overlap_ratio = self._match_ingredients(
                user_ings, recipe.get("ingredients", [])
            )

            text_score, title_matches = self._calculate_text_relevance(query, recipe) if query else (0.0, 0)

            # Filtering rules:
            if user_ings:
                # When user ingredients are provided, require at least 1 true matched ingredient
                if len(matched) == 0:
                    continue
                combined_score = (overlap_ratio * 0.75) + (text_score * 0.25)
            else:
                # When only query is provided (e.g. "Cách làm món X?"):
                # Require title match to ensure the requested dish actually exists in the database
                if title_matches == 0 or text_score < 0.25:
                    continue
                combined_score = text_score

            scored_candidates.append({
                "recipe": recipe,
                "matched_ingredients": matched,
                "missing_ingredients": missing,
                "overlap_score": round(overlap_ratio, 4),
                "text_score": round(text_score, 4),
                "combined_score": round(combined_score, 4)
            })

        # Sort descending by combined_score, tie-break by matched count
        scored_candidates.sort(
            key=lambda x: (x["combined_score"], len(x["matched_ingredients"])),
            reverse=True
        )

        top_results = scored_candidates[:k]

        logger.info(f"[RAG Retriever] Retrieved {len(top_results)} relevant recipes (Top {k}):")
        for idx, r in enumerate(top_results, 1):
            logger.info(
                f"  {idx}. {r['recipe']['name']} - Score: {r['combined_score']:.2f} "
                f"(Matched: {len(r['matched_ingredients'])}, Missing: {len(r['missing_ingredients'])})"
            )

        return top_results
