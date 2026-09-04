# Modal Deployment Guide for Food AI Service

Tài liệu hướng dẫn chi tiết triển khai hệ thống **Food AI Service** lên nền tảng **Modal Serverless GPU Cloud**.

---

## 1. Architecture

Hệ thống AI Service hoạt động độc lập trên Modal theo mô hình Serverless Container:

```
+-------------------------------------------------------------+
|                     Client (Mobile / Web)                   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Main Backend (be_nckh)                     |
+-------------------------------------------------------------+
                               |
                               v (HTTPS REST API)
+-------------------------------------------------------------+
|              Food AI Service on Modal Cloud                 |
|                                                             |
|  +-------------------------------------------------------+  |
|  |             Modal App (FastAPI ASGI)                  |  |
|  +-------------------------------------------------------+  |
|         |                                   |               |
|         v                                   v               |
|  +---------------------+            +--------------------+  |
|  | Modular Detectors   |            | Strict RAG Engine  |  |
|  | - YOLO26 (YOLO11)   |            | - Food Database    |  |
|  | - RT-DETR           |            |   (96 Recipes)     |  |
|  | - RF-DETR           |            | - Hybrid Retriever |  |
|  | - Future Detectors  |            | - Zero-Hallucin.   |  |
|  | (ENV-Selected)      |            | - Gemini / OpenAI  |  |
|  +---------------------+            +--------------------+  |
|         |                                   |               |
|  +-------------------------------------------------------+  |
|  |              NVIDIA T4 / A10G GPU VRAM                |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

### Nguyên tắc thiết kế Modal:
- **Tách biệt Modal runtime khỏi Business Logic**: Toàn bộ logic nhận diện (`detectors/`), RAG (`rag/`), và router (`api/`) được tổ chức thành các module chuẩn Python/FastAPI. `modal_app.py` chỉ đóng vai trò khai báo môi trường container, GPU và ASGI gateway.
- **Không phụ thuộc Kaggle hay GPU Cloud cũ**: Toàn bộ quy trình inference chạy trên hạ tầng Modal chính thức với chi phí theo giây sử dụng.
- **Model Loading Lifecycle**: Trọng số mô hình và cơ sở dữ liệu được nạp đúng **1 lần** vào GPU/RAM khi container khởi động thông qua FastAPI `lifespan`. Các request tiếp theo chia sẻ instance detector trong bộ nhớ (`app.state.detector`).

---

## 2. Requirements

- Python 3.10, 3.11 hoặc 3.12 (khuyến nghị Python 3.11)
- Tài khoản Modal (đăng ký tại [modal.com](https://modal.com))
- Trọng số mô hình đã tải về thư mục `food-ai-service/models_weights/`:
  - `yolo26_best.pt` (~5.6 MB)
  - `rtdetr_best.pt` (~66 MB)
  - `rfdetr_best.pth` (~134 MB)
- Cơ sở dữ liệu công thức món ăn: `ingredient_task/data_book.json`
- API Key (Gemini API hoặc OpenAI API) cho module RAG.

---

## 3. Modal Installation

Cài đặt Modal CLI trên máy phát triển:

```bash
pip install modal
```

Xác minh phiên bản:

```bash
modal --version
```

---

## 4. Authentication

Đăng nhập tài khoản Modal từ terminal:

```bash
modal setup
```

Lệnh trên sẽ mở trình duyệt web để cấp quyền truy cập API Token vào máy tính của bạn (lưu cấu hình tại `~/.modal.toml`).

---

## 5. Secrets Management

Tất cả các API key và cấu hình nhạy cảm được quản lý qua **Modal Secrets**, tuyệt đối không commit vào git.

### Tạo Secret trên Modal Dashboard hoặc CLI:

```bash
modal secret create food-ai-secrets \
  GEMINI_API_KEY="your-gemini-api-key-here" \
  INGREDIENT_MODEL="rtdetr" \
  DETECTION_CONFIDENCE="0.25" \
  RAG_TOP_K="5" \
  LLM_PROVIDER="gemini" \
  LLM_MODEL="gemini-2.5-flash"
```

Secret `food-ai-secrets` sẽ được tự động inject vào container khi ứng dụng khởi chạy.

---

## 6. Environment Variables

Bảng tổng hợp các biến môi trường được hỗ trợ:

| Tên biến | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `INGREDIENT_MODEL` | `rtdetr` | Model nhận diện nguyên liệu được nạp vào GPU (`yolo26`, `rtdetr`, `rfdetr`). |
| `MODAL_GPU` | `T4` | Loại GPU trên Modal (`T4`, `A10G`, hoặc `None` để chạy CPU). |
| `DETECTION_CONFIDENCE` | `0.25` | Ngưỡng tin cậy tối thiểu của bounding box. |
| `RAG_TOP_K` | `5` | Số lượng công thức món ăn tối đa lấy từ cơ sở dữ liệu. |
| `LLM_PROVIDER` | `gemini` | Nhà cung cấp LLM (`gemini`, `openai`, `ollama`, `qwen`). |
| `GEMINI_API_KEY` | *(Secret)* | Khóa API Google Gemini. |
| `OPENAI_API_KEY` | *(Secret)* | Khóa API OpenAI (nếu dùng provider `openai`). |
| `PORT` | `8000` | Cổng HTTP khi chạy local development. |

---

## 7. Local Development

Chạy trực tiếp trên máy phát triển không cần Modal:

```bash
cd food-ai-service

# Kích hoạt virtualenv
.\.venv\Scripts\activate

# Khởi chạy server FastAPI
python run_ai.py
```

Server sẽ lắng nghe tại `http://localhost:8000`.

---

## 8. Local Testing với Modal

Modal cung cấp tính năng **Hot Reload Dev Server** chạy trên cloud trong khi đồng bộ mã nguồn cục bộ:

```bash
cd food-ai-service
modal serve modal_app.py
```

Modal sẽ in ra một đường dẫn HTTPS tạm thời (ví dụ `https://<username>--food-ai-service-fastapi-app-dev.modal.run`). Bạn có thể test API trực tiếp qua link này. Khi lưu file trên máy local, container sẽ tự động reload.

---

## 9. Deployment Command

Để triển khai production lâu dài lên hạ tầng Modal:

```bash
cd food-ai-service
modal deploy modal_app.py
```

Quá trình deploy:
1. Modal build container image với PyTorch, CUDA, Ultralytics, RF-DETR.
2. Upload weights mô hình và dữ liệu công thức món ăn vào image snapshot.
3. Tạo endpoint HTTPS cố định production.

---

## 10. Production Endpoint

Sau khi deploy thành công, Modal sẽ cung cấp URL cố định:

```
https://<workspace>--food-ai-service-fastapi-app.modal.run
```

### Các Endpoint chính:
- **`GET /health`**: Kiểm tra trạng thái hệ thống, active detector, trạng thái nạp mô hình, và số lượng công thức trong RAG.
- **`GET /api/v1/models`**: Danh sách các detector có sẵn và detector đang kích hoạt.
- **`POST /api/v1/ingredients/detect`**: API nhận diện nguyên liệu chuẩn production (nhận file ảnh, trả về bounding box `[x1, y1, x2, y2]` và confidence `[0.0, 1.0]`).
- **`POST /api/v1/models/{model_name}/detect`**: Thử nghiệm nhận diện với detector cụ thể.
- **`POST /api/v1/recipes/suggest`**: API gợi ý công thức RAG từ cơ sở dữ liệu món ăn Việt Nam.
- **`POST /api/ai/analyze-image`**: Endpoint tương thích ngược cho backend `be_nckh`.
- **`POST /api/ai/recipe-suggest`**: Endpoint tương thích ngược RAG cho `be_nckh`.

---

## 11. Backend Integration (be_nckh)

Để kết nối backend `be_nckh` (Node.js/Flask/Spring) sang Modal, chỉ cần cập nhật biến môi trường trong backend:

```env
# Trong be_nckh/.env
AI_SERVICE_URL=https://<workspace>--food-ai-service-fastapi-app.modal.run
```

Tất cả các API call từ backend tới AI Service sẽ thông qua giao thức HTTPS tiêu chuẩn. Vì Modal hỗ trợ backward compatibility (`POST /api/ai/analyze-image` và `POST /api/ai/recipe-suggest`), `be_nckh` không cần thay đổi bất kỳ dòng code gọi API nào.

---

## 12. Model Loading Lifecycle

- **Tránh tải mô hình lặp lại**: Trong `run_ai.py`, hàm `lifespan(app)` đọc `INGREDIENT_MODEL` và gọi `detector.load()` đúng 1 lần khi container khởi động.
- Trọng số mô hình được cache trong VRAM/RAM của container.
- Mỗi HTTP request đến endpoint `/api/v1/ingredients/detect` sẽ tái sử dụng `request.app.state.detector`, đảm bảo độ trễ inference chỉ từ **15ms - 80ms** trên GPU T4.

---

## 13. GPU Configuration

Cấu hình GPU trong `modal_app.py`:

```python
@app.function(
    image=image,
    gpu="T4",               # Lựa chọn: "T4" (tiết kiệm), "A10G" (hiệu năng cao)
    scaledown_window=300,   # Giữ ấm container trong 5 phút sau request cuối
    timeout=600             # Timeout tối đa cho request (giây)
)
```

- **T4 GPU**: 16GB VRAM, chi phí thấp, tối ưu cho bài toán object detection thời gian thực.
- **CPU Fallback**: Nếu muốn chạy hoàn toàn bằng CPU để tiết kiệm chi phí, cấu hình `MODAL_GPU="None"`.

---

## 14. Health Check

Kiểm tra trạng thái triển khai:

```bash
curl -X GET "https://<workspace>--food-ai-service-fastapi-app.modal.run/health"
```

Response mẫu:
```json
{
  "status": "healthy",
  "service": "Food AI Service",
  "version": "3.0.0",
  "active_detector": "rtdetr",
  "detector_loaded": true,
  "available_detectors": ["yolo26", "rtdetr", "rfdetr"],
  "rag_recipes_count": 96,
  "llm": {
    "provider": "gemini",
    "model": "gemini-2.5-flash"
  }
}
```

---

## 15. Logging & Monitoring

- Xem log thời gian thực:
  ```bash
  modal app logs food-ai-service
  ```
- Xem dashboard trực quan trên trình duyệt: Truy cập [modal.com/apps](https://modal.com/apps) để theo dõi:
  - Tần suất gọi API (Requests per second)
  - Thời gian phản hồi (Latency)
  - Số lượng container đang chạy (Active / Idle Containers)
  - Lượng GPU VRAM tiêu thụ.

---

## 16. Error Handling

Hệ thống xử lý lỗi đồng bộ ở tất cả các tầng:
- **File ảnh không hợp lệ / rỗng**: Trả về `400 Bad Request` kèm thông báo lỗi chi tiết.
- **Model không hỗ trợ**: Trả về `400 Bad Request` kèm danh sách model hợp lệ.
- **Lỗi inference GPU**: Bắt exception, log traceback và trả về `500 Internal Server Error`.
- **RAG No-Result**: Trả về `200 OK` với `best_recipe: null` và giải thích lý do rõ ràng, tuyệt đối không bịa đặt công thức.

---

## 17. Cold Start Optimization

- Thiết lập `scaledown_window=300` (giữ container sống thêm 5 phút khi không có request) giúp giảm thiểu số lần cold start.
- Weights mô hình được nướng sẵn vào Image layer (`.add_local_dir("models_weights", ...)`), không cần tải qua mạng khi container khởi động.
- Cold start container Modal với GPU T4 mất trung bình **12 - 18 giây**, sau đó các request tiếp theo chỉ mất **30 - 80 mili-giây**.

---

## 18. Troubleshooting

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `Unsupported detector: 'xyz'` | Biến `INGREDIENT_MODEL` không hợp lệ | Đặt `INGREDIENT_MODEL` thành một trong: `yolo26`, `rtdetr`, `rfdetr`. |
| `Model weight not found` | Thiếu file `.pt` / `.pth` trong `models_weights/` | Đảm bảo file trọng số tồn tại trước khi chạy `modal deploy`. |
| `Modal Secret not found` | Chưa tạo secret `food-ai-secrets` | Chạy lệnh `modal secret create food-ai-secrets ...` như mục 5. |
| `CUDA out of memory` | Ảnh đầu vào quá lớn hoặc nhiều worker | Hệ thống đã mặc định cấu hình `workers=1` và resize ảnh khi tiền xử lý. |

---

## 19. Adding a New Detector

Để thêm một model mới (ví dụ `DINO`):
1. Tạo file `food-ai-service/detectors/dino_detector.py` kế thừa `BaseDetector`.
2. Đăng ký với decorator `@register_detector("dino")`.
3. Đặt trọng số vào `models_weights/dino_best.pt`.
4. Cập nhật `INGREDIENT_MODEL=dino` trong Modal Secret.
5. Deploy lại với `modal deploy modal_app.py`.
**Không cần sửa bất kỳ file router hay core pipeline nào.**

---

## 20. Updating RAG Database

Để cập nhật cơ sở dữ liệu công thức món ăn:
1. Thêm hoặc cập nhật công thức trong `food-ai-service/ingredient_task/data_book.json`.
2. Chạy test `pytest tests/test_rag.py` để kiểm tra độ chính xác tra cứu.
3. Chạy `modal deploy modal_app.py` để đồng bộ dữ liệu mới lên Modal Cloud.
