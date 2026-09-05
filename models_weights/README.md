# HƯỚNG DẪN TẢI VÀ ĐẶT TÊN TRỌNG SỐ MÔ HÌNH (MODELS WEIGHTS)

Thư mục này chứa các file checkpoint / trọng số mô hình trí tuệ nhân tạo (AI Models) dùng để nhận dạng nguyên liệu món ăn Việt Nam trong `food-ai-service`.

---

## 1. Danh Sách Link Tải Google Drive & Tên File Chuẩn

| STT | Mô Hình (Detector) | Link Tải Google Drive | Tên File Bắt Buộc Khi Đặt Vào Thư Mục Này | Dung Lượng Dự Kiến |
|:---:|:---|:---|:---|:---:|
| **1** | **YOLO26** *(Khuyến nghị - 38 classes)* | [Tải YOLO26 từ Google Drive](https://drive.google.com/file/d/1egYsvfNYOJaz0O1IBF4nw21KUwEixn-_/view?usp=sharing) | `yolo_best.pt`<br>*(hoặc `yolo26_best.pt`)* | ~6.2 MB |
| **2** | **RF-DETR** *(32 classes)* | [Tải RF-DETR từ Google Drive](https://drive.google.com/file/d/10o-uwRlbCaeazujrxlcVm4fpXQCWj3Mv/view?usp=sharing) | `rfdetr_best.pth` | ~134 MB |
| **3** | **RT-DETR** *(32 classes)* | [Tải RT-DETR từ Google Drive](https://drive.google.com/file/d/1m43b2Y-1_cE3AWBDVdia-hsOC_ZU5GGC/view?usp=sharing) | `rtdetr_best.pt` | ~66 MB |

---

## 2. Vị Trí Lưu File
Tất cả các file tải về phải được đặt trực tiếp vào thư mục:
```text
food-ai-service/models_weights/
├── yolo_best.pt          (Mô hình YOLO26 chính)
├── rfdetr_best.pth       (Mô hình RF-DETR)
├── rtdetr_best.pt        (Mô hình RT-DETR)
└── README.md             (File hướng dẫn này)
```

---

## 3. Cách Cấu Hình Trong `.env` Tương Ứng

Tùy theo mô hình bạn muốn kích hoạt sử dụng trong `food-ai-service/.env`:

### Dùng YOLO26 (Mặc định - Nhanh & Nhẹ Nhất):
```env
INGREDIENT_MODEL=yolo26
YOLO26_MODEL_PATH=models_weights/yolo_best.pt
DETECTION_CONFIDENCE=0.25
```

### Dùng RF-DETR:
```env
INGREDIENT_MODEL=rfdetr
RFDETR_MODEL_PATH=models_weights/rfdetr_best.pth
DETECTION_CONFIDENCE=0.25
```

### Dùng RT-DETR:
```env
INGREDIENT_MODEL=rtdetr
RTDETR_MODEL_PATH=models_weights/rtdetr_best.pt
DETECTION_CONFIDENCE=0.25
```

---

## 4. Hướng Dẫn Tải Nhanh Bằng Python `gdown` (Tùy chọn)

Nếu môi trường đã cài đặt `gdown` (`pip install gdown`), bạn có thể mở terminal trong thư mục `food-ai-service/models_weights` và chạy:

```bash
# 1. Tải YOLO26
gdown 1egYsvfNYOJaz0O1IBF4nw21KUwEixn-_ -O yolo_best.pt

# 2. Tải RF-DETR
gdown 10o-uwRlbCaeazujrxlcVm4fpXQCWj3Mv -O rfdetr_best.pth

# 3. Tải RT-DETR
gdown 1m43b2Y-1_cE3AWBDVdia-hsOC_ZU5GGC -O rtdetr_best.pt
```
