"""
FastAPI Server to run the fine-tuned Qwen model (Base + LoRA) locally.
Compatible with the `food-ai-service` endpoint.

To run:
    python serve_docker.py
"""
import os
import torch
import time
import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serve_docker")

# Environment Variables
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "Maikhang/nckh-qwen3-4b-lora")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8001"))

# Global model & tokenizer
model = None
tokenizer = None

def parse_flattened_prompt(prompt: str) -> List[Dict[str, str]]:
    """
    Parses a flattened prompt string back into a structured messages list.
    Handles '### Hướng dẫn hệ thống', '### Người dùng', and '### Trợ lý' markers.
    """
    prompt = prompt.strip()
    if not prompt:
        return []
    
    markers = ["### Hướng dẫn hệ thống", "### Người dùng", "### Trợ lý", "### System", "### User", "### Assistant"]
    has_markers = any(m in prompt for m in markers)
    
    if not has_markers:
        return [{"role": "user", "content": prompt}]
    
    parts = prompt.split("### ")
    messages = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith("Hướng dẫn hệ thống") or part.startswith("System"):
            content = part.replace("Hướng dẫn hệ thống", "", 1).replace("System", "", 1).strip()
            messages.append({"role": "system", "content": content})
        elif part.startswith("Người dùng") or part.startswith("User"):
            content = part.replace("Người dùng", "", 1).replace("User", "", 1).strip()
            messages.append({"role": "user", "content": content})
        elif part.startswith("Trợ lý") or part.startswith("Assistant"):
            content = part.replace("Trợ lý", "", 1).replace("Assistant", "", 1).strip()
            messages.append({"role": "assistant", "content": content})
        else:
            if messages:
                messages[-1]["content"] += "\n\n" + part
            else:
                messages.append({"role": "user", "content": part})
                
    return messages

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    logger.info("==================================================")
    logger.info("STARTING LOCAL LLM SERVER (FASTAPI + TRANSFORMERS)")
    logger.info("==================================================")
    logger.info(f"Base Model: {BASE_MODEL}")
    logger.info(f"LoRA Adapter: {LORA_ADAPTER}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device Capability: {torch.cuda.get_device_capability(0)}")

    # Load tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    if LOAD_IN_4BIT and torch.cuda.is_available():
        logger.info("Loading Base Model in 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map=device_map,
            token=HF_TOKEN,
            trust_remote_code=True
        )
    else:
        logger.info("Loading Base Model in FP16...")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=HF_TOKEN,
            trust_remote_code=True
        )

    # Load adapter
    if LORA_ADAPTER and LORA_ADAPTER.strip():
        logger.info(f"Loading LoRA Adapter from: {LORA_ADAPTER} ...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, token=HF_TOKEN)
        logger.info("Successfully merged LoRA adapter dynamically.")
    else:
        logger.info("No LoRA Adapter specified. Running base model directly.")
        model = base_model

    model.eval()
    logger.info("✓ Model is fully loaded and ready!")
    yield
    # Cleanup
    logger.info("Shutting down local LLM server.")

app = FastAPI(
    title="Local LLM Inference Service",
    description="Serves the fine-tuned Qwen model (Base + LoRA) inside Docker container.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.3

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "base_model": BASE_MODEL,
        "lora_adapter": LORA_ADAPTER,
        "cuda": torch.cuda.is_available()
    }

@app.post("/chat")
@app.post("/api/chat")  # fallback compatible path
async def chat(request: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        t0 = time.time()
        
        # 1. Parse prompt back to messages
        messages = parse_flattened_prompt(request.message)
        logger.info(f"Received request. Parsed into {len(messages)} chat turns.")
        
        # 2. Format with original Qwen Chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable internal COT if the model is Qwen3
        )
        
        # 3. Tokenize input
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        # 4. Generate
        logger.info(f"Generating reply (max_new_tokens={request.max_new_tokens}, temp={request.temperature})...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else 0.001,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # 5. Decode output
        generated_tokens = outputs[0][input_len:]
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Generation completed in {latency_ms}ms. Response length: {len(reply)} chars.")
        
        return {
            "reply": reply,
            "latency_ms": latency_ms,
            "tokens_generated": len(generated_tokens)
        }
        
    except Exception as e:
        logger.exception("Error during text generation")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
