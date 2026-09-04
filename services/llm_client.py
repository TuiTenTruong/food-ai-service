"""
Client LLM thống nhất — OpenAI, Gemini (OpenAI-compatible), Ollama, Qwen (Modal HTTP).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from .llm_config import LLMSettings, get_llm_settings, llm_settings_summary


def _wrap_chat_content(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def flatten_messages_for_qwen(messages: List[Dict[str, str]]) -> str:
    """Gộp lịch sử hội thoại thành một prompt cho endpoint Modal /chat."""
    blocks: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            blocks.append(f"### Hướng dẫn hệ thống\n{content}")
        elif role == "user":
            blocks.append(f"### Người dùng\n{content}")
        elif role == "assistant":
            blocks.append(f"### Trợ lý\n{content}")
        else:
            blocks.append(content)

    if not blocks:
        return ""

    # Modal endpoint chỉ nhận một chuỗi user — giữ nguyên toàn bộ ngữ cảnh.
    return "\n\n".join(blocks)


class LLMClient:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_llm_settings()
        self._openai_client: OpenAI | None = None

        if self.settings.provider in ("openai", "ollama", "gemini"):
            kwargs: dict[str, Any] = {"api_key": self.settings.api_key or "ollama"}
            if self.settings.base_url:
                kwargs["base_url"] = self.settings.base_url
            self._openai_client = OpenAI(**kwargs)

        if self.settings.provider == "qwen" and not self.settings.qwen_api_url:
            raise ValueError(
                "LLM_PROVIDER=qwen cần QWEN_API_URL hoặc MODAL_CHAT_URL "
                "(URL endpoint Modal /chat sau khi deploy finetune/modal/serve.py)"
            )

        if self.settings.provider == "openai" and not self.settings.api_key:
            raise ValueError("LLM_PROVIDER=openai cần OPENAI_API_KEY")

        if self.settings.provider == "gemini" and not self.settings.api_key:
            raise ValueError("LLM_PROVIDER=gemini cần GEMINI_API_KEY")

        print(
            f" [LLMClient] {self.settings.provider} | model={self.settings.model} | "
            f"{llm_settings_summary(self.settings)}"
        )

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def provider(self) -> str:
        return self.settings.provider

    def chat_completions_create(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        **_kwargs: Any,
    ) -> Any:
        if self.settings.provider == "qwen":
            return self._qwen_chat(messages, temperature, max_tokens)

        # Gemini 2.5 thinking models may return empty content if max_tokens quá thấp
        effective_max_tokens = max_tokens
        if self.settings.provider == "gemini" and effective_max_tokens < 256:
            effective_max_tokens = 256

        assert self._openai_client is not None
        return self._openai_client.chat.completions.create(
            model=model or self.settings.model,
            messages=messages,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )

    def _qwen_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        url = self.settings.qwen_api_url
        payload = {
            "message": flatten_messages_for_qwen(messages),
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
        timeout = int(__import__("os").getenv("QWEN_REQUEST_TIMEOUT", "180"))

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise RuntimeError(str(data["error"]))

        reply = (data.get("reply") or data.get("content") or "").strip()
        if not reply:
            raise RuntimeError(f"Qwen API trả về rỗng: {data}")

        return _wrap_chat_content(reply)


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
