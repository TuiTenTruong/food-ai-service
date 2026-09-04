"""
RAG Package for Food AI Service.
Provides Food Database access, Hybrid Retriever, Strict Prompt Builder,
and RAG Service grounded in 96 Vietnamese dishes.
"""

from .food_db import FoodDatabase, get_food_database
from .retriever import HybridRetriever
from .prompt_builder import build_recipe_suggestion_prompt, SYSTEM_PROMPT
from .llm_client import LLMClient, get_llm_client
from .service import RAGService, get_rag_service

__all__ = [
    "FoodDatabase",
    "get_food_database",
    "HybridRetriever",
    "build_recipe_suggestion_prompt",
    "SYSTEM_PROMPT",
    "LLMClient",
    "get_llm_client",
    "RAGService",
    "get_rag_service",
]
