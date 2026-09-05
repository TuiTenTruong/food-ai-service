#!/usr/bin/env python
"""
Runner Script for Modal Serverless Food AI Service.
Usage:
    python serve_modal.py          # Khởi chạy interactive dev server (tắt bằng Ctrl+C)
    python serve_modal.py --test   # Kiểm tra nhanh container trên Modal GPU rồi thoát
"""

import sys
import os
import subprocess

# Đảm bảo mã hóa UTF-8 an toàn trên mọi terminal Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Nạp biến môi trường từ .env
load_dotenv()

def print_header():
    detector = os.getenv("INGREDIENT_MODEL", "yolo26")
    llm = os.getenv("LLM_PROVIDER", "gemini")
    gpu = os.getenv("MODAL_GPU", "T4")
    db_path = os.getenv("FOOD_DB_PATH", "rag/data_book.json")

    print("\n" + "=" * 65)
    print("        [FOOD AI SERVICE] - MODAL SERVERLESS RUNNER         ")
    print("=" * 65)
    print(f"[*] Mô hình Detector : {detector.upper()} (Trọng số: models_weights/yolo_best.pt)")
    print(f"[*] Nhà cung cấp LLM : {llm.upper()}")
    print(f"[*] Cấu hình GPU    : Nvidia {gpu}")
    print(f"[*] CSDL Món ăn      : {db_path} (96 công thức Việt Nam)")
    print("=" * 65)

def check_modal_auth():
    """Kiểm tra xem thư viện modal đã cài đặt chưa."""
    try:
        import modal
        return True
    except ImportError:
        print("\n[!] Thư viện 'modal' chưa được cài đặt!")
        print("[>] Vui lòng chạy: pip install modal")
        return False

def main():
    print_header()

    if not check_modal_auth():
        sys.exit(1)

    # Chế độ chạy
    is_test_mode = "--test" in sys.argv or "--run" in sys.argv or "run" in sys.argv
    
    if is_test_mode:
        print("\n[+] Đang chạy kiểm thử cloud container (modal run modal_app.py)...")
        cmd = [sys.executable, "-m", "modal", "run", "modal_app.py"]
    else:
        print("\n[+] Đang khởi động Modal Serverless Dev Server...")
        print("[i] Khi server sẵn sàng, Modal sẽ cấp một đường dẫn URL công khai (HTTPS).")
        print("[i] Khi không sử dụng nữa, chỉ cần nhấn CTRL + C để TẮT SERVER (Chi phí = 0$).\n")
        cmd = [sys.executable, "-m", "modal", "serve", "modal_app.py"]

    try:
        # Chạy lệnh modal trực tiếp với mã hóa UTF-8 bắt buộc
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n" + "=" * 65)
        print("[OK] ĐÃ DỪNG SERVER MODAL THÀNH CÔNG!")
        print("[*] Container trên Modal Cloud đã tự động tắt.")
        print("[*] Không còn tài nguyên nào chạy ngầm và KHÔNG phát sinh chi phí.")
        print("=" * 65 + "\n")
    except subprocess.CalledProcessError as e:
        if e.returncode != 0:
            print(f"\n[!] Lệnh kết thúc với mã lỗi: {e.returncode}")
            print("[>] Nếu chưa xác thực tài khoản Modal, hãy chạy: modal setup")

if __name__ == "__main__":
    main()
