# Cleanup Report

## Deleted Files
1. `food-ai-service/finetune/kaggle/fine-tune-qwen3-4b.ipynb`: Jupyter notebook phục vụ huấn luyện Qwen3-4B trên môi trường Kaggle GPU.
2. `food-ai-service/finetune/kaggle/train_lora.py`: Script Python huấn luyện QLoRA cho Qwen3-4B trên Kaggle.
3. `food-ai-service/finetune/requirements-train.txt`: File danh sách thư viện huấn luyện (Unsloth, TRL, PEFT, bitsandbytes, datasets).
4. `food-ai-service/finetune/interface-api.ipynb`: Notebook thử nghiệm API cũ, không còn giá trị sử dụng trong production.
5. `food-ai-service/finetune/implementation_plan.md`: File kế hoạch của dự án "Spending Diary App" bị copy nhầm vào repo.
6. `food-ai-service/pip-install.log`: File log tạm của lệnh pip install UTF-16LE (~100 KB).
7. `food-ai-service/finetune/DOCKER_RUN_GUIDE.md`: File trùng lặp với `docs/DOCKER_RUN_GUIDE.md`.

## Deleted Code
- Đã loại bỏ toàn bộ code huấn luyện LoRA chạy trên Kaggle GPU.
- Loại bỏ các lệnh cài đặt thư viện training không cần thiết cho production runtime (`trl`, `unsloth`).

## Removed Dependencies
- `unsloth`, `trl`, `peft` training components, `bitsandbytes` training configs (chỉ phục vụ train, không phục vụ inference production).

## Removed APIs
- Không có API production nào bị xóa. Tất cả các endpoint hiện tại (`/health`, `/api/ai/analyze-image`, `/api/ai/recipe-suggest`, `/api/ai/chat/*`) đều được bảo toàn tính tương thích ngược.

## Files Kept
1. `food-ai-service/models_weights/yolo_best.pt`: Trọng số YOLO26 phục vụ nhận diện nguyên liệu.
2. `food-ai-service/models_weights/rtdetr_best.pt`: Trọng số RT-DETR phục vụ nhận diện nguyên liệu.
3. `food-ai-service/models_weights/rfdetr_best.pth`: Trọng số RF-DETR Medium phục vụ nhận diện nguyên liệu.
4. `food-ai-service/ingredient_task/data_book.json`: 96 công thức món ăn chuẩn phục vụ RAG.
5. `food-ai-service/finetune/data/`: Tạm thời giữ lại dataset JSON tham khảo, không nạp vào runtime.
6. `food-ai-service/finetune/serve_docker.py`: Giữ lại để tham khảo cấu trúc prompt và tokenizer nếu cần.

## Files Requiring Further Investigation
- `food-ai-service/services/vision_service.py`: Cần refactor để trở thành adapter chuyển tiếp sang kiến trúc mới `detectors/` thay vì giữ code ResNet-50 cứng.
- `food-ai-service/services/retrieval_service.py`: Cần tối ưu để không lặp lại hành vi index động mỗi request.

## Reason For Each Change
- **Loại bỏ Kaggle & LoRA Training Code**: Dự án chuyển sang triển khai production trên Modal Serverless Platform. Việc giữ code train và notebook Kaggle làm phình to repo, gây nhầm lẫn môi trường và tiềm ẩn lỗi bảo mật / dependency conflict.
- **Loại bỏ File Rác**: File kế hoạch `implementation_plan.md` của app quản lý chi tiêu là rác từ dự án khác. Log file `pip-install.log` là file tạm sinh ra khi cài đặt môi trường.
- **Bảo toàn Adapter và Runtime cần thiết**: Không xóa bất kỳ model weight nào cần thiết cho inference.

## Verification After Cleanup
- Chạy lệnh kiểm tra import trong môi trường `food-ai-service`: Không có lỗi `ModuleNotFoundError` liên quan đến các file đã xóa.
- Không có module nào trong `food-ai-service/services/` hay `be_nckh/` tham chiếu đến các file đã xóa.
- Cấu trúc thư mục sạch sẽ, sẵn sàng cho việc triển khai Modular Detector Architecture.
