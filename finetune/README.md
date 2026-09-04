# Fine-tune Qwen3-4B — Pipeline NCKH

Sinh **~2.600 mẫu chất lượng cao** từ **96 công thức** (không inflate dataset).

## 1. Sinh dataset

```powershell
cd finetune
python generate_dataset.py --stats --also-qwen
```

| File | Mô tả |
|------|-------|
| `data/nckh_sft_train.json` | Format `messages` (TRL / HuggingFace) |
| `data/nckh_sft_train_qwen.json` | Format Qwen `conversations` |

**Cấu trúc / công thức:** 26 mẫu (14 hỏi trực tiếp + 8 gợi ý NL + 2 mẹo + 2 RAG)  
**Từ chối:** 100 mẫu món không có trong DB  
**Tổng:** 2.596 mẫu

## 2. Fine-tune trên Kaggle

1. Upload `data/nckh_sft_train.json` vào Kaggle Dataset hoặc Notebook
2. Copy `kaggle/train_lora.py` vào notebook
3. Settings → **GPU T4 x2**
4. Secrets → `HF_TOKEN` (quyền write)
5. Sửa `HF_REPO = "username/nckh-qwen3-4b-lora"`
6. Chạy train (~1–2 giờ)

```python
!pip install -q trl peft bitsandbytes accelerate transformers datasets
import os
os.environ["HF_REPO"] = "YOUR_USERNAME/nckh-qwen3-4b-lora"
os.environ["DATA_PATH"] = "/kaggle/input/nckh-sft/nckh_sft_train.json"
!python train_lora.py
```

## 3. Push Hugging Face

Script tự push khi có `HF_TOKEN`. Kiểm tra tại `https://huggingface.co/<username>/nckh-qwen3-4b-lora`.

## 4. Deploy Modal.com

```powershell
pip install modal
modal token new
modal secret create huggingface HF_TOKEN=hf_xxx
cd finetune/modal
$env:HF_MODEL_ID="YOUR_USERNAME/nckh-qwen3-4b-lora"
modal deploy serve.py
```

Endpoint: `https://<workspace>--nckh-cooking-chat-chat.modal.run`

```bash
curl -X POST .../chat -H "Content-Type: application/json" \
  -d '{"message": "Cách làm bò sốt vang?"}'
```

## 5. Tích hợp food-ai-service

Trong `.env`:

```env
USE_MODAL_LLM=True
MODAL_CHAT_URL=https://xxx--nckh-cooking-chat-chat.modal.run
```

Chi tiết: `docs/QWEN3_FINETUNE.md`
