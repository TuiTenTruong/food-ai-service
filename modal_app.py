"""
Modal Production Deployment for Food AI Service.
Provides serverless GPU/CPU deployment on Modal with FastAPI ASGI interface.
"""

import os
import modal

# 1. App definition
app = modal.App("food-ai-service")

# 2. Container Image definition
# Uses Debian slim with CUDA drivers, system graphics libs, and ML packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "ultralytics>=8.3.0",
        "rfdetr>=1.10.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pillow>=10.0.0",
        "sentence-transformers>=3.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "httpx>=0.27.0"
    )
    .add_local_dir("models_weights", remote_path="/root/food-ai-service/models_weights")
    .add_local_dir("ingredient_task", remote_path="/root/food-ai-service/ingredient_task")
    .add_local_dir("detectors", remote_path="/root/food-ai-service/detectors")
    .add_local_dir("rag", remote_path="/root/food-ai-service/rag")
    .add_local_dir("api", remote_path="/root/food-ai-service/api")
    .add_local_dir("services", remote_path="/root/food-ai-service/services")
    .add_local_file("run_ai.py", remote_path="/root/food-ai-service/run_ai.py")
)

# 3. Secrets configuration
# Modal Secret 'food-ai-secrets' injects GEMINI_API_KEY, INGREDIENT_MODEL, etc.
# Fallback to empty list if running locally without Modal secrets
secrets = []
try:
    secrets.append(modal.Secret.from_name("food-ai-secrets"))
except Exception:
    pass

# 4. GPU Selection: T4 GPU provides balanced inference performance & cost
gpu_spec = os.environ.get("MODAL_GPU", "T4")
selected_gpu = gpu_spec if gpu_spec and gpu_spec != "None" else None


# 5. Serverless Web Endpoint
@app.function(
    image=image,
    gpu=selected_gpu,
    secrets=secrets,
    timeout=600,
    scaledown_window=300,  # Keep idle container warm for 5 minutes
)
@modal.asgi_app()
def fastapi_app():
    """
    Exposes the FastAPI application as a serverless ASGI web endpoint on Modal.
    The FastAPI lifespan handler loads the configured detector (YOLO26, RT-DETR, or RF-DETR)
    and Food Database once on container boot.
    """
    import sys
    sys.path.insert(0, "/root/food-ai-service")
    os.chdir("/root/food-ai-service")

    from run_ai import app as web_app
    return web_app
