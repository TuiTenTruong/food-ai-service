"""Cấu hình LLM — chọn provider qua biến môi trường."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

VALID_PROVIDERS = ("openai", "gemini", "qwen", "ollama")


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    # OpenAI / Ollama (OpenAI-compatible)
    api_key: str = ""
    base_url: str | None = None
    # Qwen fine-tune (Modal HTTP)
    qwen_api_url: str = ""


def resolve_llm_provider() -> str:
    """
    Ưu tiên LLM_PROVIDER; tương thích USE_OLLAMA và MODAL_CHAT_URL cũ.

    LLM_PROVIDER=openai | gemini | qwen | ollama
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in VALID_PROVIDERS:
        return explicit

    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        return "ollama"

    if os.getenv("QWEN_API_URL") or os.getenv("MODAL_CHAT_URL"):
        return "qwen"

    return "openai"


def get_llm_settings() -> LLMSettings:
    provider = resolve_llm_provider()

    if provider == "gemini":
        base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip()
        return LLMSettings(
            provider="gemini",
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url=base_url.rstrip("/") + "/",
        )

    if provider == "ollama":
        return LLMSettings(
            provider="ollama",
            model=os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_RAG_MODEL", "qwen2.5")),
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )

    if provider == "qwen":
        url = (
            os.getenv("QWEN_API_URL", "").strip()
            or os.getenv("MODAL_CHAT_URL", "").strip()
        )
        return LLMSettings(
            provider="qwen",
            model=os.getenv("QWEN_MODEL", "nckh-qwen3-4b-lora"),
            qwen_api_url=url.rstrip("/"),
        )

    return LLMSettings(
        provider="openai",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


def llm_settings_summary(settings: LLMSettings | None = None) -> dict:
    cfg = settings or get_llm_settings()
    summary = {"provider": cfg.provider, "model": cfg.model}
    if cfg.provider == "gemini":
        summary["base_url"] = cfg.base_url
    elif cfg.provider == "qwen":
        summary["qwen_api_url"] = cfg.qwen_api_url or "(chưa cấu hình)"
    elif cfg.provider == "ollama":
        summary["base_url"] = cfg.base_url
    return summary
