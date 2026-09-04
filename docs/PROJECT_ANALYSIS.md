# Project Analysis

## Project Overview
Dự án Nghiên cứu Khoa học (NCKH) tập trung vào bài toán tự động nhận diện nguyên liệu thực phẩm từ hình ảnh thực tế và gợi ý công thức chế biến các món ăn truyền thống Việt Nam phù hợp. Hệ thống hướng đến người dùng gia đình và người nấu ăn hàng ngày, giúp tận dụng tối đa nguyên liệu sẵn có trong tủ lạnh / bếp để đề xuất món ăn ngon, tiết kiệm và giảm thiểu lãng phí thực phẩm.

Hệ thống bao gồm 3 thành phần chính:
1. **Frontend Mobile App (`fe_nckh`)**: Ứng dụng di động xây dựng bằng Flutter, cho phép người dùng chụp ảnh nguyên liệu, quản lý tủ đồ ăn (pantry), xem danh mục 96 món ăn truyền thống và nhận tư vấn nấu nướng từ trợ lý AI.
2. **Backend Web Service (`be_nckh`)**: Dịch vụ backend xây dựng bằng Python Flask, kết nối cơ sở dữ liệu quan hệ MariaDB/MySQL (`nckh`), quản lý thông tin người dùng, danh mục món ăn, công thức nấu ăn, các bước chế biến và lịch sử quét nguyên liệu.
3. **AI Microservice (`food-ai-service`)**: Dịch vụ AI xây dựng bằng Python FastAPI, chịu trách nhiệm xử lý thị giác máy tính (Computer Vision) để nhận diện nguyên liệu thực phẩm và ứng dụng mô hình ngôn ngữ lớn kết hợp RAG (Retrieval-Augmented Generation) để đưa ra hướng dẫn nấu nướng.

---

## Current Architecture
Kiến trúc luồng dữ liệu hiện tại:

```text
[Người dùng / Flutter App (fe_nckh)]
              |
              | (HTTP REST API, port 5000)
              v
     [Backend Flask (be_nckh)]
              |
              +---> [MariaDB / MySQL (database nckh, port 3306)]
              |     (Lưu trữ 96 recipes, 47 ingredients, pantry, scan_sessions)
              |
              | (HTTP REST API, port 8000)
              v
   [AI Service FastAPI (food-ai-service)]
              |
              +---> [Computer Vision Pipeline]
              |     (Hiện tại: YOLOv8 + ResNet50 phân loại 38 nhãn)
              |     (Trọng số sẵn có: YOLO26, RT-DETR, RF-DETR 32 nhãn)
              |
              +---> [RAG & Chatbot Service]
                    (ChromaDB vector store + sentence-transformers / BGE-Reranker)
                    (LLM: Gemini / OpenAI / Ollama / Qwen3-4B LoRA)
```

### Các hạn chế cốt lõi của kiến trúc hiện tại:
1. **Phụ thuộc triển khai thủ công**: AI Service đang chạy như một tiến trình FastAPI cục bộ (`run_ai.py`), chưa tận dụng được hạ tầng serverless linh hoạt như Modal.
2. **Chưa chuẩn hóa Detector**: Code `vision_service.py` hiện tại cố định quy trình YOLOv8 phát hiện bounding box rồi crop ảnh đưa qua ResNet-50 phân loại, trong khi thư mục `models_weights/` đã có 3 mô hình Object Detection hoàn chỉnh (`yolo_best.pt`, `rtdetr_best.pt`, `rfdetr_best.pth`) có thể phát hiện và phân loại trực tiếp 32 nguyên liệu Việt Nam mà không cần mạng phụ ResNet.
3. **Chưa có Registry / Strategy cho Model**: Không có cơ chế đổi model nhận diện qua biến môi trường (`INGREDIENT_MODEL`). Muốn đổi model phải sửa code cứng.
4. **RAG chưa tối ưu với đặc thù dữ liệu**: Mỗi request RAG đang đẩy danh sách recipe từ backend sang để AI service build in-memory index vào ChromaDB rồi xóa index (`self.retrieval_service.clear()`), gây lãng phí tài nguyên tính toán và có nguy cơ hallucination nếu không kiểm soát chặt chẽ.
5. **Code training và Kaggle còn lẫn trong repository**: Tồn tại thư mục `finetune/kaggle/` với notebook và script huấn luyện LoRA cùng các tệp rác copy từ dự án khác.

---

## Directory Structure
Cấu trúc cây thư mục cấp cao của repository:

```text
d:\NCKH\Code
├── .agent/                           # Persistent project memory (Multi-Agent system)
│   ├── PROJECT_STATE.md
│   ├── TASK_BOARD.md
│   ├── CHANGELOG.md
│   ├── DECISIONS.md
│   ├── HANDOFF.md
│   └── REVIEW_STATE.md
├── be_nckh/                          # Backend Flask Application
│   ├── app/
│   │   ├── api/                      # Blueprints (chat, ingredient, pantry, recipe, scan)
│   │   ├── clients/                  # HTTP clients (ai_service_client.py)
│   │   ├── models/                   # SQLAlchemy models (recipe, ingredient, scan, etc.)
│   │   ├── repositories/             # Data access layer
│   │   ├── services/                 # Business logic layer (vision, recipe suggestion, chat)
│   │   └── utils/                    # Media URL, response helpers
│   ├── backups/                      # SQL backups (nckh (2).sql)
│   ├── database.sql                  # MySQL database dump (96 recipes)
│   └── run.py                        # Backend entrypoint (port 5000)
├── fe_nckh/                          # Frontend Flutter Application
│   ├── lib/                          # Dart source code (screens, widgets, providers)
│   └── pubspec.yaml
├── food-ai-service/                  # AI Microservice (FastAPI)
│   ├── data/                         # Local debug images & vector storage
│   ├── finetune/                     # [CẦN DỌN DẸP] Code fine-tune Qwen3-4B, Kaggle, Docker
│   ├── ingredient_task/              # 96 công thức món ăn chuẩn (data_book.json) & ảnh món ăn
│   ├── models_weights/               # Trọng số pre-trained (yolo_best.pt, rtdetr_best.pt, rfdetr_best.pth)
│   ├── services/                     # Current AI service implementations
│   │   ├── vision_service.py         # Current CV pipeline (YOLO + ResNet)
│   │   ├── rag_controller.py         # RAG coordinator
│   │   ├── retrieval_service.py      # ChromaDB retrieval + CrossEncoder
│   │   ├── embedding_service.py      # Vietnamese SBERT embedding
│   │   ├── llm_service.py            # LLM prompt & caller
│   │   └── chatbot_service.py        # Conversational assistant
│   ├── run_ai.py                     # FastAPI entrypoint (port 8000)
│   └── requirements.txt
├── docs/                             # Tài liệu kỹ thuật
│   ├── DOCKER_RUN_GUIDE.md
│   └── QWEN3_FINETUNE.md
├── setup.md                          # Hướng dẫn cài đặt dự án
└── README.md
```

---

## Backend
- **Framework**: Flask với Flask-SQLAlchemy, Flask-CORS, Flask-Migrate.
- **Cấu hình (`be_nckh/app/config.py`)**:
  - `DATABASE_URL`: `mysql+pymysql://root:@localhost:3306/nckh`
  - `AI_SERVICE_BASE_URL`: `http://localhost:8000`
  - `AI_SERVICE_TIMEOUT`: 30 giây (mặc định)
  - `VISION_API_ENDPOINT`: `http://127.0.0.1:8000/api/ai/analyze-image`
- **Các thành phần chính**:
  - `be_nckh/app/clients/ai_service_client.py`: Class `AIServiceClient` gọi endpoint `POST /api/ai/recipe-suggest` và `GET /health` sang `food-ai-service`.
  - `be_nckh/app/services/vision_service.py`: Class `VisionService` nhận file ảnh từ upload của mobile app, gửi multipart/form-data sang `food-ai-service` tại `POST /api/ai/analyze-image`.
  - `be_nckh/app/services/scan_service.py`: Điều phối luồng scan ảnh, map nhãn AI nhận diện với bảng `ingredients`, sau đó gọi gợi ý món ăn.
  - `be_nckh/app/services/recipe_suggestion_service.py`: Gợi ý công thức dựa trên nguyên liệu người dùng có, query công thức từ database và gọi `AIServiceClient` để sinh câu trả lời RAG.

---

## AI Service
- **Framework**: FastAPI (asynchronous), Uvicorn server (port 8000).
- **Điểm vào chính (`run_ai.py`)**:
  - Quản lý lifespan khởi tạo các dịch vụ vào biến toàn cục `models`:
    - `models["vision"] = VisionService()`
    - `models["ai_assistant"] = CookingLangChainService()`
    - `models["chatbot"] = get_chatbot_service()`
    - `models["rag_controller"] = get_rag_controller()`
- **Tình trạng hiện tại của các service con**:
  - `services/vision_service.py`: Đang khởi tạo YOLOv8 + ResNet-50. Tuy nhiên trong thư mục `models_weights/` thực tế không có tệp `resnet_best.pth`, mà có 3 mô hình phát hiện trực tiếp 32 nguyên liệu: `yolo_best.pt`, `rtdetr_best.pt`, `rfdetr_best.pth`.
  - `services/retrieval_service.py`: Sử dụng `sentence_transformers.CrossEncoder` (mặc định `BAAI/bge-reranker-base`) và `embedding_service.py` (`keepitreal/vietnamese-sbert`). Điểm bất hợp lý là phương thức `retrieve()` lại nhận tham số `recipes` được gửi từ Backend mỗi lần và gọi `build_index(recipes)` để nạp vào ChromaDB rồi `clear()`.
  - `services/llm_client.py`: Cung cấp client gọi thống nhất qua `LLM_PROVIDER`: `gemini` (OpenAI-compatible endpoint), `openai`, `ollama`, hoặc `qwen` (Modal URL).

---

## Database
Cơ sở dữ liệu trung tâm là MariaDB/MySQL `nckh` (được dump trong `be_nckh/database.sql` và `be_nckh/backups/nckh (2).sql`):
1. `ingredient_categories`: 5 danh mục nguyên liệu (`c1`: Thịt cá, `c2`: Trứng sữa, `c3`: Rau củ, `c4`: Tinh bột, `c5`: Gia vị).
2. `ingredients`: 47 nguyên liệu phổ biến trong ẩm thực Việt Nam (id `ing-book-0001` đến `ing-book-0047`), gồm tên, icon emoji, category_id, image_url.
3. `recipes`: 96 công thức món ăn độc bản (id `recipe-book-0001` đến `recipe-book-0096`), gồm tên món, mô tả, image_url, `cook_time_minutes`, `difficulty` (`easy`, `medium`, `hard`), `servings`, `cuisine_type` (`Vietnamese`), `diet_tags` (JSON), và `source` (nguồn sách ẩm thực uy tín).
4. `recipe_ingredients`: Liên kết món ăn và nguyên liệu, chứa định lượng `quantity` và đơn vị `unit` (`gram`, `ml`, `quả`, `muỗng xúp`...).
5. `recipe_steps`: Các bước nấu ăn chi tiết từng món (`step_number`, `title`, `description`, `tip`).
6. `pantry_items`: Nguyên liệu trong tủ bếp của người dùng.
7. `scan_sessions`: Lịch sử các phiên quét ảnh nhận diện nguyên liệu và kết quả gợi ý món.

Ngoài ra, toàn bộ dữ liệu 96 món ăn này được lưu trữ dưới dạng JSON có cấu trúc đầy đủ tại `food-ai-service/ingredient_task/data_book.json` và `be_nckh/data/data_book.json`.

---

## Current APIs
### AI Service APIs:
- `GET /health`: Kiểm tra trạng thái AI Service và cấu hình LLM đang chọn.
- `POST /api/ai/analyze-image` (deprecated): Nhận diện nguyên liệu từ ảnh bằng Computer Vision và trả về danh sách nguyên liệu kèm gợi ý món ban đầu.
- `POST /api/ai/recipe-suggest` & `POST /internal/rag/recipe-suggest`: Nhận `user_ingredients`, danh sách `recipes` từ backend, thực hiện semantic search và gọi LLM gợi ý món.
- `POST /api/ai/chat/sessions`: Tạo phiên trò chuyện mới.
- `POST /api/ai/chat/sessions/{session_id}/messages`: Gửi tin nhắn trao đổi về ẩm thực, dinh dưỡng với trợ lý AI.

### Backend APIs liên quan AI:
- `POST /api/scan`: Tiếp nhận ảnh tải lên từ Flutter app, gọi `VisionService.detect_ingredients()`, lưu phiên vào `scan_sessions` và trả về kết quả.
- `POST /api/recipes/suggest`: Tiếp nhận danh sách nguyên liệu từ mobile app, gọi `AIServiceClient.suggest_recipe()`.
- `POST /api/chat/sessions/{session_id}/messages`: Tiếp nhận tin nhắn người dùng và chuyển tiếp sang AI Service.

---

## Current Models
Thư mục `food-ai-service/models_weights/` chứa 3 file trọng số thực tế:
1. **`yolo_best.pt`** (118,462,302 bytes ~ 118 MB):
   - Kiến trúc: Ultralytics Detection Model thế hệ mới (chứa block `C3k2` của YOLO11 / YOLO26).
   - Huấn luyện nhận diện trực tiếp 32 nguyên liệu Việt Nam.
2. **`rtdetr_best.pt`** (66,345,021 bytes ~ 66 MB):
   - Kiến trúc: Ultralytics RT-DETR (Real-Time DEtection TRansformer).
   - Nhận diện 32 class nguyên liệu Việt Nam: `['beef', 'bellpepper', 'bittergourd', 'bottlegourd', 'broccoli', 'cabbage', 'carrot', 'cauliflower', 'chayote', 'chicken', 'chickenegg', 'chickenleg', 'chickenwin', 'corn', 'cucumber', 'duckegg', 'eggplant', 'garlic', 'ginger', 'jicama', 'okra', 'onion', 'pork', 'potato', 'pumpkin', 'radish', 'scallion', 'shrimp', 'spongegourd', 'sweetpotato', 'tofu', 'tomato']`.
3. **`rfdetr_best.pth`** (134,223,909 bytes ~ 134 MB):
   - Kiến trúc: Roboflow `RFDETRMedium` (rfdetr version 1.9.4).
   - Backbone: `dinov2_windowed_small`, 4 decoder layers, 300 queries, resolution 576.
   - Nhận diện cùng 32 class nguyên liệu trên.

---

## Detection Flow
### Luồng hiện tại trong code cũ (`services/vision_service.py`):
```text
Ảnh bytes -> cv2.imdecode
          -> YOLOv8 dự đoán bounding box (conf > 0.25)
          -> Crop vùng ảnh bbox
          -> ResNet-50 phân loại class (conf > 0.40)
          -> Gán nhãn tiếng Việt từ dictionary class_names
          -> Lưu ảnh debug (annotated)
          -> Trả về danh sách [{"name": ..., "confidence": ...}]
```
*Nhược điểm*: Quy trình 2 giai đoạn (detect rồi crop qua ResNet) đã lạc hậu so với 3 model End-to-End Object Detection có sẵn trong `models_weights/`. Cần chuyển đổi sang pipeline trực tiếp từ một trong ba detector (YOLO26, RT-DETR, RF-DETR).

---

## Current LLM Flow
```text
User Question / Ingredients
           |
           v
Backend `be_nckh` query toàn bộ / một phần recipes từ MySQL
           |
           v
Gửi payload recipes sang `food-ai-service` qua POST `/api/ai/recipe-suggest`
           |
           v
`food-ai-service` build index embedding động vào ChromaDB
           |
           v
Semantic search + Overlap score + CrossEncoder Rerank
           |
           v
LLM Prompt (Gemini / OpenAI / Ollama / Qwen) với Top K công thức
           |
           v
JSON Parse -> Trả về best_recipe, reasons, missing_ingredients, instructions
           |
           v
Clear ChromaDB in-memory index
```

---

## Deployment
- **Hiện tại**: AI Service chạy local qua lệnh `python run_ai.py` trên Windows.
- **Tài liệu cũ (`docs/QWEN3_FINETUNE.md`)**: Mô tả pipeline train LoRA trên Kaggle, sau đó push adapter lên HuggingFace và deploy riêng model LLM lên Modal bằng `modal deploy serve.py`.
- **Mục tiêu mới**: Chuyển toàn bộ AI Service (bao gồm cả Detector và RAG Service) lên nền tảng **Modal** (`modal_app.py`), cho phép chạy serverless có thể scale, đồng thời vẫn giữ khả năng chạy local thuận tiện.

---

## Environment Variables
Bảng các biến môi trường quan trọng:
| Tên biến | Chức năng | Giá trị mẫu |
|---|---|---|
| `PORT` | Cổng dịch vụ AI Service khi chạy local | `8000` |
| `DATABASE_URL` | Kết nối cơ sở dữ liệu MySQL | `mysql+pymysql://root:@localhost:3306/nckh` |
| `INGREDIENT_MODEL` | Mô hình nhận diện được chọn khởi động | `rtdetr` (hoặc `yolo26`, `rfdetr`) |
| `DETECTION_CONFIDENCE` | Ngưỡng độ tin cậy tối thiểu cho detection | `0.25` |
| `RAG_TOP_K` | Số công thức tối đa đưa vào ngữ cảnh LLM | `5` |
| `LLM_PROVIDER` | Nhà cung cấp LLM phục vụ RAG và Chatbot | `gemini` (hoặc `openai`, `ollama`, `qwen`) |
| `GEMINI_API_KEY` | Khóa API của Google Gemini | `AIzaSy...` |
| `GEMINI_MODEL` | Model Gemini sử dụng | `gemini-2.5-flash` |
| `OPENAI_API_KEY` | Khóa API của OpenAI | `sk-...` |
| `OPENAI_MODEL` | Model OpenAI sử dụng | `gpt-4o-mini` |
| `MODAL_GPU` | Cấu hình GPU trên Modal | `T4` (hoặc `A10G`) |

---

## Kaggle Related Code
- `food-ai-service/finetune/kaggle/fine-tune-qwen3-4b.ipynb`: Notebook Jupyter để chạy fine-tune trên Kaggle GPU.
- `food-ai-service/finetune/kaggle/train_lora.py`: Script huấn luyện LoRA cho Qwen3-4B trên Kaggle.
- `docs/QWEN3_FINETUNE.md`: Tài liệu hướng dẫn quy trình Kaggle -> HuggingFace -> Modal.
*Đánh giá*: Không phục vụ cho production runtime của AI Service. Cần loại bỏ khỏi production package để tránh phụ thuộc và nhầm lẫn.

---

## LoRA Related Code
- `food-ai-service/finetune/kaggle/train_lora.py`: Code huấn luyện adapter LoRA.
- `food-ai-service/finetune/requirements-train.txt`: Gói phụ thuộc huấn luyện LoRA (`unsloth`, `trl`, `peft`, `xformers`).
- `food-ai-service/finetune/generate_dataset.py`, `formatters.py`, `templates.py`: Code sinh tập dữ liệu SFT từ công thức nấu ăn.
*Đánh giá*: Phần training LoRA không được chạy trong production AI Service. Nếu người dùng chọn dùng model fine-tuned thì chỉ gọi qua endpoint inference hoặc sử dụng Gemini/OpenAI API có sẵn. Toàn bộ code train LoRA sẽ được dọn dẹp khỏi production.

---

## Training Related Code
- Toàn bộ script trong `food-ai-service/finetune/kaggle/` và các file cấu hình huấn luyện đi kèm.

---

## Unused Code
1. `food-ai-service/finetune/implementation_plan.md`: Đây là file kế hoạch của một ứng dụng quản lý chi tiêu (`Spending Diary App` / `Luan-Van/Project`) bị copy nhầm vào thư mục `finetune/`. Hoàn toàn không liên quan đến đề tài NCKH món ăn.
2. `food-ai-service/finetune/interface-api.ipynb`: Notebook thử nghiệm API cũ không sử dụng.
3. `food-ai-service/pip-install.log`: File log cài đặt pip UTF-16LE dung lượng lớn (~100 KB) bị sót trong source code.
4. `food-ai-service/test_api.py`, `food-ai-service/test_rag.py`: Đã bị xóa ở commit trước nhưng cần bổ sung bộ test chính thức vào thư mục `tests/`.

---

## Duplicate Files
- `food-ai-service/ingredient_task/data_book.json` và `be_nckh/data/data_book.json`: Hai file JSON chứa danh sách 96 món ăn giống hệt nhau. Đây là nguồn dữ liệu chuẩn mực của đề tài.

---

## Potential Problems
1. **Lỗi Không Load Được YOLO26 Do Thiếu Block `C3k2`**:
   - Thư viện `ultralytics==8.0.200` cài đặt cũ không có định nghĩa `C3k2` (khối backbone mới của YOLO11/YOLO26). Khi gọi `torch.load` hoặc `YOLO('models_weights/yolo_best.pt')` sẽ bắn lỗi `AttributeError: Can't get attribute 'C3k2'`.
   - *Giải pháp*: Cập nhật `ultralytics>=8.3.0`.
2. **Cơ Chế PyTorch 2.6+ `weights_only=True`**:
   - PyTorch 2.6+ mặc định bật kiểm tra `weights_only=True`, khiến các checkpoint Ultralytics và DETR bị chặn unpickle lớp DetectionModel.
   - *Giải pháp*: Cấu hình an toàn `weights_only=False` cho các checkpoint cục bộ tin cậy trong thư mục `models_weights/`.
3. **Phụ Thuộc MySQL Khi Deploy Lên Modal Serverless**:
   - Khi chạy trên Modal Cloud, container không thể kết nối tới `localhost:3306` của máy dev. Nếu service cố kết nối MySQL sẽ bị crash.
   - *Giải pháp*: Xây dựng cơ chế fallback thông minh: nếu `DATABASE_URL` kết nối được thì dùng, nếu không thì tự động load từ `data_book.json` sẵn có trong container.
4. **Không Thống Nhất Schema Bounding Box & Confidence**:
   - Giữa các model Object Detection khác nhau, format bbox có thể là `[x, y, w, h]` hoặc `[x1, y1, x2, y2]`, confidence có thể từ 0-100 hoặc 0.0-1.0.
   - *Giải pháp*: Chuẩn hóa 100% về bbox `[x1, y1, x2, y2]` pixel nguyên và confidence `0.0 - 1.0`.

---

## Recommended Refactoring
1. **Module hóa Detector (`detectors/`)**:
   - Tạo interface trừu tượng `BaseDetector`.
   - Tạo Registry và Decorator `@register_detector`.
   - Implement `YOLO26Detector`, `RTDETRDetector`, `RFDETRDetector`.
   - Chọn model bằng ENV `INGREDIENT_MODEL`.
   - Hỗ trợ thêm detector mới (ví dụ `DINODetector`, `GroundingDINODetector`) mà không cần sửa core pipeline.
2. **Chuẩn hóa API Detection**:
   - `POST /api/v1/ingredients/detect` (endpoint chính thức).
   - `GET /api/v1/models` (trạng thái model).
   - Giữ `POST /api/ai/analyze-image` để tương thích ngược với `be_nckh`.
3. **Tái cấu trúc RAG (`rag/`)**:
   - Tách biệt Data Access (`food_db.py`), Hybrid Retriever (`retriever.py`), Prompt Builder (`prompt_builder.py`), LLM Client (`llm_client.py`), và Service (`service.py`).
   - Cung cấp Top-K có cấu hình (`RAG_TOP_K`), xử lý tình huống không có kết quả (`no-result`) không hallucinate.
4. **Triển khai Modal App (`modal_app.py`)**:
   - Xây dựng file cấu hình Modal độc lập, mount model weights, cấu hình GPU `T4` và Secret cho API keys.
5. **Thiết lập Bộ Test Suite Toàn Diện (`tests/`)**:
   - Kiểm thử tự động cả 18 tiêu chí yêu cầu trước khi cấp phép PASS.
