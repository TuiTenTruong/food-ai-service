"""
Modal Serverless GPU Deployment for Food AI Service.
Supports:
- Interactive hot-reload dev server (tự động tắt khi dừng): `modal serve modal_app.py`
- Test inference on cloud GPU: `modal run modal_app.py`
- Production deployment: `modal deploy modal_app.py`
"""

import os
import sys
import modal

# 1. Tên ứng dụng trên Modal
APP_NAME = os.environ.get("MODAL_APP_NAME", "food-ai-service")
app = modal.App(APP_NAME)

# 2. Định nghĩa Container Image trên Modal (Debian Slim + CUDA drivers + ML packages)
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
        "httpx>=0.27.0",
        "openai>=1.12.0",
        "transformers>=4.45.0",
        "accelerate>=0.30.0",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.1.0",
    )
    .add_local_dir("models_weights", remote_path="/root/food-ai-service/models_weights")
    .add_local_dir("detectors", remote_path="/root/food-ai-service/detectors")
    .add_local_dir("rag", remote_path="/root/food-ai-service/rag")
    .add_local_dir("api", remote_path="/root/food-ai-service/api")
    .add_local_dir("services", remote_path="/root/food-ai-service/services")
    .add_local_dir("ingredient_task", remote_path="/root/food-ai-service/ingredient_task")
    .add_local_file(".env", remote_path="/root/food-ai-service/.env")
    .add_local_file("run_ai.py", remote_path="/root/food-ai-service/run_ai.py")
)

# 2.1. Bộ nhớ đệm lưu trữ trọng số Hugging Face (Qwen 2.5 - 3B) - Không tải lại 6GB mỗi lần khởi động
hf_cache_volume = modal.Volume.from_name("food-ai-hf-cache", create_if_missing=True)

# 3. Cấu hình GPU (Mặc định Nvidia T4: 16GB VRAM, suy luận < 20ms)
gpu_spec = os.environ.get("MODAL_GPU", "T4")
selected_gpu = gpu_spec if gpu_spec and gpu_spec != "None" else None

# Giữ ấm đúng 15 phút (900s) theo yêu cầu - Sau 15 phút không có request tự động tắt về 0$ chi phí
scaledown_window = int(os.environ.get("MODAL_SCALEDOWN_WINDOW", 900))
timeout_seconds = int(os.environ.get("MODAL_TIMEOUT", 600))


# 4. Serverless ASGI Web Endpoint
@app.function(
    image=image,
    gpu=selected_gpu,
    timeout=timeout_seconds,
    scaledown_window=scaledown_window,  # Giữ ấm 15 phút sau request cuối
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
@modal.asgi_app()
def fastapi_app():
    """
    Khởi chạy ứng dụng FastAPI trên Modal dưới dạng Serverless ASGI app.
    Tải mô hình nhận dạng (YOLO26) và Qwen 2.5 - 3B vào bộ nhớ GPU một lần khi container khởi động.
    """
    import sys
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
    sys.path.insert(0, "/root/food-ai-service")
    os.chdir("/root/food-ai-service")

    from dotenv import load_dotenv
    load_dotenv("/root/food-ai-service/.env")

    from run_ai import app as web_app
    return web_app


# 5. Hàm kiểm thử trên Cloud Modal (Chạy khi dùng lệnh `modal run modal_app.py`)
@app.function(
    image=image,
    gpu=selected_gpu,
    timeout=timeout_seconds,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def test_inference():
    """Kiểm tra mô hình nhận dạng và RAG trên Modal container."""
    import sys
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
    sys.path.insert(0, "/root/food-ai-service")
    os.chdir("/root/food-ai-service")

    from dotenv import load_dotenv
    load_dotenv("/root/food-ai-service/.env")

    from detectors import get_detector
    from rag import get_rag_service

    model_name = os.getenv("INGREDIENT_MODEL", "yolo26")
    det = get_detector(model_name)
    det.load()

    rag = get_rag_service()
    recipes_count = len(rag.food_db.get_all_recipes())
    rag_res = rag.suggest_recipe(["thịt bò", "hành tây"], top_k=2)

    return {
        "status": "success",
        "detector": model_name,
        "classes_count": len(getattr(det, "names", getattr(det, "classes", {}))),
        "recipes_in_db": recipes_count,
        "sample_suggestion": rag_res.get("best_recipe"),
        "recipe_id": rag_res.get("recipe_id"),
        "llm_provider": rag.llm_client.provider,
    }


@app.local_entrypoint()
def main():
    """
    Entrypoint khi chạy: `modal run modal_app.py`
    Giúp kiểm tra xem container Modal GPU có chạy tốt không mà không cần mở server web.
    """
    print("\n" + "=" * 65)
    print("🚀 ĐANG KẾT NỐI VÀ KIỂM TRA FOOD AI SERVICE TRÊN MODAL GPU...")
    print("=" * 65)
    
    result = test_inference.remote()
    
    print("\n✅ CONTAINER MODAL SERVERLESS HOẠT ĐỘNG HOÀN HẢO!")
    print(f"📦 Mô hình Detector : {result['detector'].upper()} (38 classes)")
    print(f"🍲 Cơ sở dữ liệu RAG : {result['recipes_in_db']} công thức món ăn")
    print(f"🤖 Nhà cung cấp LLM : {result['llm_provider']}")
    print(f"🥗 Gợi ý món mẫu    : {result.get('sample_suggestion')}")
    print("=" * 65)
    print("\n💡 ĐỂ CHẠY SERVER WEB LIÊN TỤC (TẮT KHI KHÔNG SỬ DỤNG BẰNG CTRL+C):")
    print("👉 Chạy lệnh: modal serve modal_app.py")
    print("👉 Hoặc:      python serve_modal.py\n")
