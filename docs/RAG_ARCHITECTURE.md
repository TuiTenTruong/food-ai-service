# RAG Architecture

## Previous LLM Architecture
Trước khi tái cấu trúc, dịch vụ AI (`food-ai-service`) sử dụng mô hình kết hợp giữa LangChain (`CookingLangChainService`), Chatbot Service cục bộ, và một pipeline RAG tạm thời:
1. Client hoặc Backend `be_nckh` gửi request kèm danh sách công thức (`recipes`) được truy vấn từ MySQL.
2. `retrieval_service.py` tiếp nhận payload và gọi `embedding_service.build_index(recipes)` để nạp động vào ChromaDB vector store.
3. Sử dụng mô hình `keepitreal/vietnamese-sbert` để tạo vector embedding và `BAAI/bge-reranker-base` để rerank.
4. Sau khi truy xuất, hàm `clear()` được gọi để xóa index ChromaDB tạm thời.
5. Gọi LLM qua `llm_client.py` (Gemini / OpenAI / Ollama / Qwen3-4B LoRA).

---

## Problems With Previous Architecture
1. **Lãng phí tài nguyên và độ trễ cao**: Việc build embedding và nạp vào vector store động trên từng HTTP request, sau đó xóa trắng index là phản mẫu (anti-pattern). Điều này gây trễ nghiêm trọng (2-5 giây cho mỗi lượt gợi ý món).
2. **Nguy cơ Hallucination (Ảo giác)**: Khi không có dữ liệu khớp hoặc khi prompt chưa siết chặt, LLM có xu hướng tự sáng tạo ra công thức không có thực hoặc bịa đặt số liệu dinh dưỡng không nằm trong sách tham khảo.
3. **Phụ thuộc kết nối MySQL trực tiếp**: Khi triển khai lên nền tảng đám mây Serverless như Modal, container không thể kết nối tới `localhost:3306` của máy phát triển. Nếu không có cơ chế dữ liệu độc lập, toàn bộ luồng RAG sẽ bị sập.
4. **Không tối ưu cho truy vấn nguyên liệu thực phẩm**: Tìm kiếm vector thuần túy thường nhầm lẫn giữa các nguyên liệu có ngữ cảnh từ tương đồng nhưng bản chất sinh học khác nhau (ví dụ: "thịt heo" và "thịt bò").

---

## Food Database
Cơ sở dữ liệu cốt lõi gồm **96 công thức món ăn truyền thống độc bản Việt Nam** được chuẩn hóa trong đề tài NCKH, trích dẫn từ 4 cuốn sách ẩm thực kinh điển:
- *Hướng dẫn nấu ăn 200 món truyền thống*
- *Kỹ thuật chế biến 300 món ăn ngon*
- *555 món ăn Việt Nam*
- *Nấu ăn gia đình miền Nam*

### Schema thực tế của từng món ăn:
- `id`: Mã định danh (ví dụ `recipe-book-0001` đến `recipe-book-0096`).
- `name`: Tên món ăn tiếng Việt chuẩn (`Thịt bò nấu sốt vang`, `Canh mây cà chua`...).
- `description`: Mô tả hương vị, màu sắc và cảm quan của món.
- `cook_time_minutes`: Thời gian chế biến (phút).
- `difficulty`: Độ khó (`easy`, `medium`, `hard`).
- `servings`: Khẩu phần phục vụ (số người ăn).
- `cuisine_type`: Ẩm thực (`Vietnamese`).
- `diet_tags`: Thẻ phân loại món (`Món mặn`, `Canh`, `Món xào`, `Thanh nhiệt`...).
- `ingredients`: Danh sách nguyên liệu chi tiết:
  - `name`: Tên nguyên liệu.
  - `quantity`: Định lượng (ví dụ: 500, 2, 15).
  - `unit`: Đơn vị đo (`gram`, `ml`, `quả`, `muỗng xúp`...).
  - `is_optional`: Nguyên liệu bắt buộc hay tùy chọn.
  - `category_id`: Nhóm thực phẩm (`c1`: Thịt cá, `c2`: Trứng sữa, `c3`: Rau củ, `c4`: Tinh bột, `c5`: Gia vị).
- `instructions`: Các bước thực hiện tuần tự:
  - `step_number`: Số thứ tự bước.
  - `title`: Tên bước (`Sơ chế`, `Tẩm ướp`, `Xào nấu`, `Hoàn thiện`).
  - `description`: Nội dung hướng dẫn kỹ thuật chi tiết.
  - `tip`: Mẹo vặt từ nghệ nhân nấu ăn.
- `source`: Tên sách và số trang dẫn chứng nguồn gốc.

---

## Selected RAG Strategy
**Chiến lược được chọn: HYBRID RAG (Ingredient Exact/Synonym Overlap + Semantic Relevance + Metadata Filtering)**.

```text
[Yêu cầu Người Dùng] (Nguyên liệu: "trứng", "cà chua", Yêu cầu: "nấu canh nhanh")
          |
          v
[Pre-Filtering theo Metadata]
- Lọc theo độ khó (difficulty)
- Lọc theo thời gian tối đa (cook_time_max)
- Lọc theo loại hình món (cuisine_type / diet_tags)
          |
          v
[Ingredient Matcher & Synonym Expansion]
- So khớp từ đồng nghĩa ẩm thực (thịt heo = thịt lợn, mướp đắng = khổ qua...)
- Tính tỷ lệ phủ nguyên liệu (overlap_ratio = matched / total_recipe_ingredients)
- Loại bỏ các món không khớp bất kỳ nguyên liệu nào (Hard Constraint chống dương tính giả)
          |
          v
[Text Relevance & Rerank]
- Tính điểm tương đồng ngữ nghĩa từ khóa và mô tả
- Điểm tổng hợp: Combined = 0.80 * Overlap_Score + 0.20 * Text_Score
          |
          v
[Top-K Selector (Mặc định Top 5)]
          |
          v
[Context Builder & Zero-Hallucination Prompt]
          |
          v
[LLM (Gemini / OpenAI / Ollama)] ---> [Fallback Tự Động sang DB Metadata nếu LLM lỗi]
          |
          v
[Structured Response (JSON)]
```

---

## Why This Strategy Was Selected
1. **Chính xác tuyệt đối về nguyên liệu**: Nấu ăn đòi hỏi tính thực tế cao. Nếu người dùng chỉ có trứng và cà chua, hệ thống không được phép gợi ý món đòi hỏi thịt bò đắt tiền chỉ vì embedding vector có độ tương đồng ngữ nghĩa mơ hồ.
2. **Không phụ thuộc Vector Database cồng kềnh**: Tập dữ liệu 96 món ăn có kích thước tối ưu (~135 KB). Việc lưu trữ và đánh chỉ mục trong bộ nhớ (In-memory Indexing) giúp tốc độ truy xuất đạt **dưới 5 mili-giây**, nhanh hơn 100 lần so với việc gọi qua Vector DB ngoài.
3. **Độc lập và tự vận hành trên Modal Cloud**: Không cần cài đặt cluster ChromaDB/Qdrant riêng, giúp giảm thiểu tối đa chi phí duy trì hạ tầng serverless.

---

## Retrieval Flow
1. Nhận danh sách nguyên liệu `user_ingredients` và tùy chọn lọc `preferences`.
2. Kiểm tra `FoodDatabase` (Dual Source: kết nối MySQL hoặc load từ `data_book.json`).
3. Lọc sơ bộ danh sách ứng viên qua các tiêu chí thời gian, độ khó.
4. Quét từng ứng viên qua bộ đối sánh từ đồng nghĩa `are_synonyms()`.
5. Tính `overlap_ratio` và `text_score`.
6. Sắp xếp giảm dần theo điểm tổng hợp và trích xuất `TOP_K` (mặc định 5 món).

---

## Context Building
Context được xây dựng ngắn gọn, chỉ bao gồm Top-K món ăn đã lọc. Mỗi món được format gồm:
- Mã định danh ID và tên món ăn.
- Thời gian nấu, độ khó, khẩu phần.
- Danh sách nguyên liệu cần thiết kèm định lượng và đơn vị.
- Danh sách nguyên liệu hiện có (đã khớp) và còn thiếu.
- Các bước nấu chi tiết kèm mẹo nấu ăn.

Context này tuyệt đối **không đưa toàn bộ 96 món ăn vào prompt**, giúp tiết kiệm token (~800 - 1500 tokens thay vì 40.000 tokens) và ngăn ngừa LLM bị quá tải ngữ cảnh (lost in the middle).

---

## Prompt Strategy
System Prompt đặt ra các quy tắc bất khả xâm phạm:
- Chỉ được phép gợi ý các món ăn nằm trong danh sách công thức được cung cấp.
- Tuyệt đối không tự bịa đặt món ăn mới hoặc suy đoán công thức không có trong database.
- Bắt buộc trả về đúng định dạng JSON chuẩn hóa để API parse an toàn.
- Nếu không có món ăn nào phù hợp, phải trả về `best_recipe: null` và lý do rõ ràng.

---

## LLM Integration
- **Mặc định / Khuyến nghị**: **Google Gemini** (`gemini-2.5-flash` qua endpoint OpenAI-compatible). Tốc độ cực nhanh, chi phí thấp, hỗ trợ tiếng Việt xuất sắc.
- **Tùy chọn mở rộng**: OpenAI (`gpt-4o-mini`), Ollama cục bộ (`qwen2.5`), hoặc Qwen3-4B fine-tuned.
- **Cơ chế Fallback Tự Động (Self-Healing)**: Nếu API Key của LLM chưa được cung cấp hoặc dịch vụ bên ngoài bị gián đoạn / timeout, `RAGService` sẽ tự động kích hoạt bộ format dữ liệu xác định (deterministic formatter) để trích xuất thẳng thông tin từ công thức có điểm cao nhất trong database và trả về cho người dùng mà không làm gián đoạn trải nghiệm.

---

## API
Endpoint chính: `POST /api/ai/recipe-suggest` (hoặc `POST /internal/rag/recipe-suggest`)
Payload:
```json
{
  "user_ingredients": ["trứng", "cà chua"],
  "preferences": {
    "difficulty": "easy",
    "cook_time_max": 30
  },
  "top_k": 5
}
```
Response:
```json
{
  "best_recipe": "Canh mây cà chua",
  "recipe_id": "recipe-book-0004",
  "reason": "Khớp 2/4 nguyên liệu chính từ cơ sở dữ liệu.",
  "cook_time_minutes": 15,
  "difficulty": "easy",
  "servings": 4,
  "matched_ingredients": ["Cà chua", "Trứng gà"],
  "missing_ingredients": ["Hành lá", "Dầu ăn"],
  "substitutions": [],
  "instructions": [
    "Bước 1: Nấu canh - Phi thơm hành, xào cà chua mềm với muối tiêu đường rồi thêm nước đun sôi..."
  ],
  "alternative_recipes": [
    {"id": "recipe-book-0015", "name": "Trứng vịt nhồi thịt", "matched_count": 2, "missing_count": 5}
  ]
}
```

---

## Error Handling
- **Database Offline**: Tự động fallback sang file JSON chuẩn `ingredient_task/data_book.json`.
- **LLM Error / Timeout**: Tự động trả về dữ liệu chuẩn từ món ăn có điểm overlap cao nhất, đi kèm thông báo lý do rõ ràng.
- **Invalid JSON Output**: Sử dụng bộ bóc tách Regex `_extract_json()` xử lý các trường hợp LLM bao bọc codeblock ````json ... ```` hoặc thêm lời bình thừa thãi.

---

## No Result Handling
Khi người dùng nhập các nguyên liệu hoàn toàn không thể kết hợp trong 96 món Việt Nam (ví dụ: `['socola', 'kem tươi', 'phô mai mozarella']`):
- Hệ thống trả về `best_recipe: null`.
- Lý do: *"Cơ sở dữ liệu 96 món ăn hiện chưa có công thức phù hợp với nguyên liệu của bạn."*
- Không bịa đặt món ăn giả.

---

## Reindex Strategy
**Not required.**
Dữ liệu 96 món ăn được load và chuẩn hóa 1 lần duy nhất khi service khởi động (`load_data()`). Không cần phải reindex hay embed lại trên mỗi request.

---

## Testing
Module RAG được kiểm thử tự động trên các kịch bản:
1. `test_rag_retrieval`: Kiểm tra tìm kiếm các nguyên liệu phổ biến (trứng, thịt gà, cà chua).
2. `test_rag_no_result`: Kiểm tra với nguyên liệu ngoại lai không có trong văn hóa ẩm thực 96 món.
3. `test_rag_prompt_context`: Đảm bảo kích thước prompt không vượt quá ngưỡng token cho phép.
4. `test_db_fallback`: Kiểm tra hoạt động khi chạy độc lập không có MySQL.

---

## Limitations
- Số lượng công thức hiện tại cố định ở 96 món của đề tài nghiên cứu.
- Các gia vị cơ bản (muối, nước mắm, tiêu, dầu ăn) được xếp vào nhóm gia vị sẵn có trong gia đình.

---

## Future Improvements
- Mở rộng cơ sở dữ liệu lên 300+ món ăn truyền thống.
- Bổ sung ước lượng calo và giá trị vi chất dinh dưỡng cho từng khẩu phần.
- Tích hợp multimodal (gửi kèm ảnh đĩa thức ăn thành phẩm để AI nhận xét độ chín).
