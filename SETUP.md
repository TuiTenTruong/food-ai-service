# HƯỚNG DẪN THIẾT LẬP VÀ KHỞI CHẠY FOOD-AI-SERVICE

Tài liệu này hướng dẫn chi tiết từ A đến Z cách cài đặt, cấu hình cơ sở dữ liệu MySQL, chọn mô hình LLM (Hugging Face Qwen 2.5 / Ollama / Gemini), nạp trọng số YOLO26, chạy kiểm thử và triển khai AI Service lên Cloud Modal GPU.

---

## 1. Kiến trúc hệ thống

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          FOOD-AI-SERVICE (FastAPI)                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Multi-Model Detector:                                               │
│    - YOLO26: models_weights/yolo_best.pt (38 classes, 6.2MB)          │
│    - RT-DETR: models_weights/rtdetr_best.pt                            │
│    - RF-DETR: models_weights/rfdetr_best.pth                           │
│                                                                        │
│ 2. Zero-Hallucination RAG Engine:                                      │
│    - Primary Data: MySQL `nckh` (96 recipes trực tiếp từ database)     │
│    - Offline Fallback: rag/data_book.json (tự động khi MySQL offline)  │
│                                                                        │
│ 3. LLM Reasoning Providers:                                           │
│    - Hugging Face: Qwen/Qwen2.5-7B-Instruct (Khuyến nghị, Serverless)  │
│    - Local: Ollama (qwen2.5 / gemma2)                                  │
│    - Cloud: Google Gemini / OpenAI                                     │
│                                                                        │
│ 4. Deployment:                                                         │
│    - Local: uvicorn run_ai:app --port 8000                             │
│    - Cloud Serverless GPU: Modal.com (Nvidia T4 / A10G)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10 / 11, Linux hoặc macOS.
- **Python**: 3.10 – 3.13 (khuyến nghị tạo môi trường ảo `.venv`).
- **MySQL / MariaDB**: XAMPP, Laragon hoặc Docker MySQL (port mặc định `3306`).
- **Git**: Đã cài đặt git.
- **RAM**: Tối thiểu 8GB RAM (16GB nếu load nhiều mô hình cùng lúc).

---

## 3. Bước 1: Thiết lập cơ sở dữ liệu MySQL

Hệ thống RAG truy vấn trực tiếp từ bảng `recipes`, `recipe_ingredients`, `ingredients` và `recipe_steps` đã lưu trong MySQL.

### 1.1. Tạo CSDL `nckh` (nếu chưa có)
Mở phpMyAdmin / DBeaver hoặc chạy lệnh SQL:
```sql
CREATE DATABASE IF NOT EXISTS `nckh` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

### 1.2. Nạp dữ liệu từ bản sao lưu `nckh (2).sql`
Bản sao lưu chuẩn nhất của bạn nằm tại: `be_nckh/backups/nckh (2).sql`.

**Cách A: Dùng phpMyAdmin / Laragon**
1. Mở phpMyAdmin (`http://localhost/phpmyadmin`).
2. Chọn CSDL `nckh`.
3. Nhấp tab **Nhập (Import)** -> Chọn file `be_nckh/backups/nckh (2).sql` -> Bấm **Nhập (Go)**.

**Cách B: Dùng lệnh Command Prompt / PowerShell**
```powershell
# Chạy từ thư mục gốc dự án
mysql -u root -p nckh < be_nckh\backups\"nckh (2).sql"
```
*(Nếu dùng XAMPP mặc định không có mật khẩu root, bỏ qua `-p`)*.

> **Lưu ý về tính độc lập:** Nếu MySQL không hoạt động hoặc chưa khởi động, AI Service **vẫn chạy bình thường** nhờ cơ chế tự động chuyển sang tập dữ liệu dự phòng [rag/data_book.json](file:///d:/NCKH/Code/food-ai-service/rag/data_book.json) có sẵn đầy đủ 96 món ăn.

---

## 4. Bước 2: Cài đặt môi trường Python

Mở PowerShell tại thư mục `food-ai-service`:

```powershell
cd d:\NCKH\Code\food-ai-service

# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 3. Nâng cấp pip
python -m pip install -U pip

# 4. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 5. Bước 3: Cấu hình file môi trường `.env`

Tạo file `.env` từ file mẫu `.env.example`:

```powershell
copy .env.example .env
```

Nội dung file `.env` chuẩn như sau:

```ini
# ==============================================================================
# CẤU HÌNH DỊCH VỤ FOOD AI SERVICE
# ==============================================================================
PORT=8000
APP_ENV=development

# ==============================================================================
# MÔ HÌNH NHẬN DIỆN NGUYÊN LIỆU (DETECTOR)
# ==============================================================================
# Chọn 1 trong 3 mô hình: yolo26 | rtdetr | rfdetr
INGREDIENT_MODEL=yolo26

# Đường dẫn trọng số mô hình
YOLO26_MODEL_PATH=models_weights/yolo_best.pt
RTDETR_MODEL_PATH=models_weights/rtdetr_best.pt
RFDETR_MODEL_PATH=models_weights/rfdetr_best.pth
DETECTION_CONFIDENCE=0.25

# ==============================================================================
# KẾT NỐI CƠ SỞ DỮ LIỆU MYSQL (RAG)
# ==============================================================================
DATABASE_URL=mysql+pymysql://root:@localhost:3306/nckh
RAG_TOP_K=5

# ==============================================================================
# CẤU HÌNH LLM (HUGGING FACE / OLLAMA / GEMINI / OPENAI)
# ==============================================================================
# Lựa chọn 1: Dùng mô hình Hugging Face mã nguồn mở (Khuyến nghị Qwen 2.5)
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_your_token_here
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct

# Lựa chọn 2: Dùng Ollama Local (miễn phí, offline)
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_CHAT_MODEL=qwen2.5

# Lựa chọn 3: Dùng Google Gemini
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_gemini_api_key
# GEMINI_MODEL=gemini-2.5-flash
```

### Cách lấy Hugging Face API Key (Miễn phí):
1. Đăng ký tài khoản tại [huggingface.co](https://huggingface.co).
2. Vào **Settings** -> **Access Tokens** -> Tạo token mới với quyền **Read**.
3. Dán token vào `HUGGINGFACE_API_KEY=hf_xxxx`.

---

## 6. Bước 4: Kiểm tra trọng số mô hình nhận diện

Mô hình YOLO26 do bạn huấn luyện đã được tích hợp sẵn trong thư mục:
- **`models_weights/yolo_best.pt`**: Trọng số mô hình YOLO26 (38 classes, 6.2MB).

Kiểm tra sự tồn tại của file:
```powershell
Test-Path models_weights\yolo_best.pt   # Phải trả về True
```

---

## 7. Bước 5: Khởi chạy AI Service

Sau khi cấu hình xong `.env`, chạy lệnh:

```powershell
.venv\Scripts\activate
$env:PYTHONUTF8=1; python run_ai.py
```

Khi chạy thành công, console sẽ hiển thị:
```text
INFO:     Loaded 96 recipes directly from MySQL database.
INFO:     Detector loaded successfully: yolo26
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Các đường dẫn kiểm tra:
- **Health check & trạng thái mô hình**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Tài liệu Swagger UI tương tác**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Danh sách detector khả dụng**: [http://127.0.0.1:8000/api/v1/models](http://127.0.0.1:8000/api/v1/models)

---

## 8. Bước 6: Kiểm tra API bằng cURL / Postman

### 8.1. Nhận diện nguyên liệu từ ảnh (POST `/api/v1/ingredients/detect`)
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/ingredients/detect" `
  -F "file=@duong/dan/toi/anh_nguyen_lieu.jpg" `
  -F "threshold=0.25"
```
*(Hệ thống cũng tương thích 100% với endpoint cũ: `/api/ai/analyze-image` của `be_nckh`)*.

### 8.2. Gợi ý món ăn RAG từ danh sách nguyên liệu (POST `/api/v1/recipes/suggest`)
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/recipes/suggest" `
  -H "Content-Type: application/json" `
  -d '{"ingredients": ["thịt bò", "hành tây", "cà chua"], "top_k": 3}'
```

---

## 9. Bước 7: Chạy kiểm thử tự động (Unit Tests)

Dự án bao gồm bộ kiểm thử tự động toàn diện với 32 bài test kiểm tra API, Detector, RAG và Modal:

```powershell
.venv\Scripts\activate
python -m pytest tests/ -v
```

Kết quả mong đợi:
```text
================= 32 passed in ~80s ==================
```

---

## 10. Bước 8: Triển khai lên Cloud Modal GPU (Tùy chọn)

Để triển khai serverless microservice lên GPU đám mây của Modal.com (chạy tự động co giãn, hỗ trợ GPU Nvidia T4 / A10G):

```powershell
# 1. Cài đặt modal client
pip install modal

# 2. Xác thực tài khoản Modal
modal setup

# 3. Chạy thử nghiệm trên cloud
modal run modal_app.py

# 4. Triển khai chính thức nhận URL Web Endpoint
modal deploy modal_app.py
```

Sau khi deploy, Modal sẽ cung cấp URL công khai (HTTPS) để cấu hình vào ứng dụng Flutter hoặc Backend.

---

## 11. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `MySQL not available or offline` | XAMPP/Laragon chưa bật MySQL hoặc sai mật khẩu trong `.env` | Kiểm tra bật MySQL ở port 3306. Dịch vụ AI vẫn chạy bình thường với file fallback `rag/data_book.json`. |
| `LLM error: 401 Unauthorized` | Sai Hugging Face / Gemini API Key | Kiểm tra lại giá trị `HUGGINGFACE_API_KEY` hoặc `GEMINI_API_KEY` trong file `.env`. |
| `UnicodeEncodeError: 'charmap'` | Terminal Windows mặc định dùng mã hóa CP1252 | Luôn chạy với biến môi trường UTF-8: `$env:PYTHONUTF8=1; python run_ai.py`. |
| `FileNotFoundError: models_weights/yolo_best.pt` | Chưa có file trọng số | File đã được lưu sẵn trong git. Hãy đảm bảo bạn đang ở thư mục `food-ai-service`. |
