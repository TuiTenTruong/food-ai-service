# Review Report (Agent 3 - Independent QA Engineer)

Báo cáo kiểm tra và theo dõi lỗi độc lập của Agent 3 trong toàn bộ quá trình tái cấu trúc, dọn dẹp, phát triển bộ nhận dạng nguyên liệu (Modular Detectors), RAG món ăn và triển khai Modal Cloud.

---

## ISSUE-001: Corrupted YOLO Checkpoint Header
- **Severity**: CRITICAL
- **Owner**: Agent 1
- **Affected Files**: `models_weights/yolo_best.pt`
- **Problem**: File `models_weights/yolo_best.pt` bị hỏng cấu trúc zip header (`BadZipFile: Bad CRC-32 for file data/944`), khiến `YOLO("models_weights/yolo_best.pt")` ném lỗi khi nạp trọng số.
- **Evidence**:
  ```
  zipfile.BadZipFile: Bad CRC-32 for file 'data/944'
  ```
- **Expected Fix**: Khôi phục file checkpoint YOLO hợp lệ tương thích với bài toán 38/32 lớp nguyên liệu.
- **Resolution**: Agent 1 đã trích xuất trọng số nguyên gốc còn nguyên vẹn từ `D:\Download\V1 YoloV8-Resnet-20260603T061249Z-3-001\V1 YoloV8-Resnet\best.pt` và lưu thành `food-ai-service/models_weights/yolo26_best.pt`. Đã kiểm tra inference độc lập và nhận diện chính xác.
- **Status**: VERIFIED (FIXED)

---

## ISSUE-002: PyTorch 2.6+ `weights_only=True` Default Security Restriction
- **Severity**: HIGH
- **Owner**: Agent 1
- **Affected Files**: `detectors/rtdetr_detector.py`, `detectors/rfdetr_detector.py`
- **Problem**: Trên PyTorch 2.6+, hàm `torch.load()` mặc định đặt `weights_only=True`. Các mô hình tùy biến như RT-DETR và RF-DETR chứa các class tùy biến và cấu trúc module phức tạp sẽ bị từ chối unpickle với lỗi `UnsupportedGlobal: ...`.
- **Evidence**:
  ```
  torch.serialization.WeightsOnlyError: Unsupported global: GLOBAL rfdetr.models.backbones...
  ```
- **Expected Fix**: Đảm bảo nạp trọng số nội bộ an toàn bằng cách cấu hình `weights_only=False` cho các checkpoint cục bộ đã được xác minh.
- **Resolution**: Agent 1 đã triển khai helper an toàn `_configure_trusted_torch()` trong `rtdetr_detector.py` và `rfdetr_detector.py`, nạp checkpoint bằng `torch.load(..., weights_only=False)`.
- **Status**: VERIFIED (FIXED)

---

## ISSUE-003: Substring False-Positive in Ingredient Retrieval
- **Severity**: MEDIUM
- **Owner**: Agent 2
- **Affected Files**: `rag/retriever.py`
- **Problem**: Khi chuẩn hóa chuỗi bỏ dấu tiếng Việt để tìm kiếm, nguyên liệu ngắn như `"kem"` (cream) bị so khớp nhầm với từ `"kèm"` trong cụm từ hướng dẫn `"ăn kèm rau sống"`, dẫn đến kết quả gợi ý sai lệch khi tìm nguyên liệu tráng miệng.
- **Evidence**: Người dùng tìm nguyên liệu kem nhưng hệ thống lại trả về món mặn có cụm từ "ăn kèm".
- **Expected Fix**: Bổ sung ranh giới từ (word boundary `\b`) trong biểu thức chính quy và yêu cầu tỷ lệ trùng khớp thực tế giữa danh sách nguyên liệu của người dùng với nguyên liệu món ăn trong cơ sở dữ liệu.
- **Resolution**: Agent 2 đã bổ sung `\b{kw}\b` vào hàm so khớp và đưa ra ràng buộc: khi người dùng truyền nguyên liệu cụ thể, món ăn được chọn bắt buộc phải có ít nhất 1 nguyên liệu thực sự khớp (`len(matched) > 0`).
- **Status**: VERIFIED (FIXED)

---

## ISSUE-004: Unbounded Context in Legacy LLM Prompts
- **Severity**: HIGH
- **Owner**: Agent 2
- **Affected Files**: `services/chat_service.py`, `services/chatbot_service.py`
- **Problem**: Kiến trúc cũ đưa toàn bộ lịch sử trò chuyện và danh sách recipe lớn vào prompt mà không có giới hạn Top-K hoặc giới hạn độ dài context, gây lãng phí token và nguy cơ tràn context window.
- **Evidence**: Code cũ format toàn bộ chunks công thức vào string prompt.
- **Expected Fix**: Triển khai `Top-K` retrieval nghiêm ngặt (mặc định Top 5, giới hạn context dưới 15.000 ký tự) và xây dựng prompt yêu cầu LLM không được bịa đặt dữ liệu ngoài context.
- **Resolution**: Agent 2 đã xây dựng package `rag/` với `HybridRetriever`, `build_recipe_suggestion_prompt()` với hệ thống lọc Top-K và prompt ràng buộc nghiêm ngặt.
- **Status**: VERIFIED (FIXED)

---

## ISSUE-005: Heavy Site-Packages Import Overhead Blocking Application Import
- **Severity**: MEDIUM
- **Owner**: Agent 1 & Agent 2
- **Affected Files**: `run_ai.py`
- **Problem**: Việc import eager các module nặng như `chromadb` và `HuggingFaceEmbeddings` ngay tại top-level của `run_ai.py` làm chậm đáng kể thời gian khởi động server và gây timeout khi chạy các tác vụ CLI nhẹ.
- **Evidence**: Quá trình `import run_ai` mất nhiều thời gian do khởi tạo Rust bindings của ChromaDB.
- **Expected Fix**: Chuyển các dịch vụ legacy và embedding phụ trợ sang cơ chế lazy import bên trong hàm `lifespan` hoặc endpoint tương ứng.
- **Resolution**: Đã tái cấu trúc `run_ai.py`, lazy-import `chatbot_service` và `llm_config` chỉ khi ứng dụng thực sự cần sử dụng.
- **Status**: VERIFIED (FIXED)
