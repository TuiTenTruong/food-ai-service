"""
RAG Controller - Điều phối toàn bộ flow RAG
Kết nối Embedding → Retrieval → LLM
"""

from typing import List, Dict, Any
from .retrieval_service import get_retrieval_service
from .llm_service import get_llm_service
from .rag_schemas import RecipeSuggestRequest, RecipeSuggestResponse, AlternativeRecipe


class RAGController:
    """
    Controller chính cho RAG Recipe Suggestion
    
    Flow:
    1. Nhận request với user_ingredients và recipes
    2. Retrieval: tìm top K recipes phù hợp
    3. LLM: sinh gợi ý cuối cùng
    4. Trả response
    """
    
    def __init__(self):
        self.retrieval_service = get_retrieval_service()
        self.llm_service = get_llm_service()
        print(" [RAGController] Initialized")
    
    def suggest_recipe(self, request: RecipeSuggestRequest) -> RecipeSuggestResponse:
        """
        Main method: nhận request, trả response
        """
        print(f"\n{'='*50}")
        print(f" [RAGController] Processing request")
        print(f" - User ingredients: {request.user_ingredients}")
        print(f" - Recipes count: {len(request.recipes)}")
        print(f" - Top K: {request.top_k}")
        print(f"{'='*50}\n")
        
        # Convert Pydantic models to dicts - bao gồm thêm thông tin image, cook_time, etc.
        recipes_dict = [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "steps": r.steps,
                "ingredients": [
                    {"name": ing.name, "amount": ing.amount}
                    for ing in r.ingredients
                ],
                "image_url": r.image_url,
                "cook_time_minutes": r.cook_time_minutes,
                "difficulty": r.difficulty,
                "servings": r.servings
            }
            for r in request.recipes
        ]
        
        # Step 1: Retrieval
        print(" [RAGController] Step 1: Retrieving relevant recipes...")
        retrieved = self.retrieval_service.retrieve(
            user_ingredients=request.user_ingredients,
            recipes=recipes_dict,
            top_k=request.top_k
        )
        
        if not retrieved:
            print(" [RAGController] No recipes retrieved")
            return RecipeSuggestResponse(
                best_recipe=None,
                recipe_id=None,
                reason="Không tìm thấy công thức phù hợp",
                matched_ingredients=[],
                missing_ingredients=[],
                substitutions=[],
                instructions=[],
                alternative_recipes=[]
            )
        
        print(f" [RAGController] Retrieved {len(retrieved)} recipes")
        for i, r in enumerate(retrieved):
            print(f"   {i+1}. {r['recipe']['name']} (score: {r['combined_score']:.2f})")
        
        # Step 2: LLM Generation
        print("\n [RAGController] Step 2: Generating suggestion with LLM...")
        llm_result = self.llm_service.generate_suggestion(
            user_ingredients=request.user_ingredients,
            retrieved_recipes=retrieved
        )
        
        # Step 3: Build response
        print("\n [RAGController] Step 3: Building response...")
        
        # Tìm best recipe trong retrieved để lấy thông tin chi tiết
        best_recipe_info = None
        best_recipe_id = llm_result.get('recipe_id')
        if best_recipe_id:
            for r in retrieved:
                if r['recipe'].get('id') == best_recipe_id:
                    best_recipe_info = r['recipe']
                    break
        
        # Parse alternative recipes với thông tin chi tiết
        alternatives = []
        for alt in llm_result.get('alternative_recipes', []):
            if isinstance(alt, dict):
                # Tìm recipe info trong retrieved
                alt_id = str(alt.get('id', ''))
                alt_info = None
                for r in retrieved:
                    if r['recipe'].get('id') == alt_id:
                        alt_info = r['recipe']
                        break
                
                alternatives.append(AlternativeRecipe(
                    id=alt_id,
                    name=alt.get('name', ''),
                    matched_count=alt.get('matched_count', 0),
                    missing_count=alt.get('missing_count', 0),
                    image_url=alt_info.get('image_url') if alt_info else None,
                    cook_time_minutes=alt_info.get('cook_time_minutes') if alt_info else None,
                    difficulty=alt_info.get('difficulty') if alt_info else None,
                    description=alt_info.get('description') if alt_info else None
                ))
        
        response = RecipeSuggestResponse(
            best_recipe=llm_result.get('best_recipe'),
            recipe_id=llm_result.get('recipe_id'),
            reason=llm_result.get('reason'),
            image_url=best_recipe_info.get('image_url') if best_recipe_info else None,
            cook_time_minutes=best_recipe_info.get('cook_time_minutes') if best_recipe_info else None,
            difficulty=best_recipe_info.get('difficulty') if best_recipe_info else None,
            servings=best_recipe_info.get('servings') if best_recipe_info else None,
            matched_ingredients=llm_result.get('matched_ingredients', []),
            missing_ingredients=llm_result.get('missing_ingredients', []),
            substitutions=llm_result.get('substitutions', []),
            instructions=llm_result.get('instructions', []),
            alternative_recipes=alternatives
        )
        
        print(f" [RAGController] Done! Best recipe: {response.best_recipe}")
        
        # Clear index sau mỗi request (vì index được build động)
        self.retrieval_service.clear()
        
        return response
    
    def suggest_recipe_simple(
        self, 
        user_ingredients: List[str], 
        recipes: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Version đơn giản hơn, nhận dict thay vì Pydantic model
        Tiện cho testing
        """
        # Step 1: Retrieval
        retrieved = self.retrieval_service.retrieve(
            user_ingredients=user_ingredients,
            recipes=recipes,
            top_k=top_k
        )
        
        if not retrieved:
            return {
                "best_recipe": None,
                "reason": "Không tìm thấy công thức phù hợp"
            }
        
        # Step 2: LLM
        result = self.llm_service.generate_suggestion(
            user_ingredients=user_ingredients,
            retrieved_recipes=retrieved
        )
        
        # Clear
        self.retrieval_service.clear()
        
        return result


# Singleton
_rag_controller = None

def get_rag_controller() -> RAGController:
    global _rag_controller
    if _rag_controller is None:
        _rag_controller = RAGController()
    return _rag_controller
