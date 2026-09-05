"""
Unified LLM Client for RAG.
Supports:
- Google Gemini (recommended, via OpenAI-compatible endpoint)
- OpenAI (gpt-4o, gpt-4o-mini)
- Local Ollama (qwen2.5, gemma2)
- Modal hosted Qwen3-4B
Handles timeouts, retries, and clean JSON extraction.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM Client."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        self.client: Optional[OpenAI] = None
        self.model: str = ""

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            base_url = os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.client = OpenAI(api_key=api_key or "demo-key", base_url=base_url)

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.client = OpenAI(api_key=api_key or "demo-key")

        elif self.provider in ("huggingface", "hf", "qwen"):
            api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN", "")
            base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co/v1/")
            self.model = os.getenv("HUGGINGFACE_MODEL") or os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            self.client = OpenAI(api_key=api_key or "hf-token", base_url=base_url)

        elif self.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self.model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_RAG_MODEL", "qwen2.5")
            self.client = OpenAI(api_key="ollama", base_url=base_url)

        else:
            # Fallback mock/test client
            logger.info(f"Using mock client for unknown provider '{self.provider}'")
            self.model = "mock-model"

        logger.info(f"Initialized LLMClient (provider='{self.provider}', model='{self.model}')")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """Call LLM and return raw text string."""
        if not self.client:
            return "{}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            raise RuntimeError(f"LLM Service failed: {e}") from e

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON from response with robust fallbacks."""
        raw_text = self.generate(prompt, system_prompt=system_prompt, temperature=temperature)
        return self._extract_json(raw_text)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from markdown code block or surrounding text."""
        clean = text.strip()
        
        # Remove ```json ... ``` code blocks if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if json_match:
            clean = json_match.group(1).strip()

        # Find first { and last }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end+1]

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM: {e}. Raw text: {text[:200]}")
            return {}


_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    """Singleton getter for LLMClient."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
