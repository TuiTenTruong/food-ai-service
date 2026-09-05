"""
Local Qwen 2.5 Inference Service on GPU using Hugging Face Transformers.
Supports Qwen/Qwen2.5-3B-Instruct (or custom model specified in env).
"""

import os
import sys
import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("QwenService")

_qwen_instance = None


class QwenService:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

    def load(self):
        """Load tokenizer and model onto GPU (cuda) with float16."""
        if self.is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading tokenizer for '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        logger.info(f"Loading weights for '{self.model_id}' onto {device} ({dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        self.is_loaded = True
        logger.info(f"'{self.model_id}' loaded successfully into GPU memory!")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate chat response using official chat template of Qwen 2.5.
        """
        if not self.is_loaded:
            self.load()

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        device = next(self.model.parameters()).device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        do_sample = temperature > 0.05
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ):
        """
        Stream chat tokens in real-time using TextIteratorStreamer.
        Yields text chunks as they are generated on GPU.
        """
        if not self.is_loaded:
            self.load()

        from transformers import TextIteratorStreamer
        from threading import Thread

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        device = next(self.model.parameters()).device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        do_sample = temperature > 0.05
        gen_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "streamer": streamer,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()


def get_qwen_service() -> QwenService:
    global _qwen_instance
    if _qwen_instance is None:
        _qwen_instance = QwenService()
    return _qwen_instance
