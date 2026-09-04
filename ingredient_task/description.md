# Mô tả tập dữ liệu công thức món ăn (Recipe Dataset Description)

Tập dữ liệu phục vụ nghiên cứu khoa học chế biến món ăn Việt Nam bao gồm:

1. **data_book.json**: Tập dữ liệu đầy đủ bao gồm trường trích dẫn nguồn (`source`) từ các sách nấu ăn truyền thống.
2. **data_book_no_source.json**: Tập dữ liệu chuẩn hóa rút gọn, loại bỏ trường nguồn sách (`source`), khớp định dạng mẫu.
3. **images/**: Thư mục chứa 96 ảnh thực tế tương ứng với 96 món ăn độc bản. Tên ảnh định dạng JPEG (`.jpg`) được chuẩn hóa không dấu từ tên món ăn (ví dụ: `Thịt bò nấu sốt vang` $\rightarrow$ `thit_bo_nau_sot_vang.jpg`).

---

## 1. Định nghĩa phân loại danh mục nguyên liệu (`category_id`)

Các nguyên liệu trong công thức được gán nhãn nhóm danh mục cụ thể:

- **c1 (Thịt cá)**: Các loại thịt (heo, bò, gà, bầu dục), cá, tôm, hải sản.
- **c2 (Trứng sữa)**: Các loại trứng (trứng gà, trứng vịt), các sản phẩm từ sữa và bơ.
- **c3 (Rau củ)**: Các loại rau, củ, quả, nấm, đậu phụ, hành, tỏi, gừng, sả.
- **c4 (Tinh bột)**: Bột mì, khoai tây, khoai lang, ngô (bắp), các sản phẩm ngũ cốc/tinh bột chính.
- **c5 (Gia vị)**: Nước mắm, muối, đường, hạt tiêu, mì chính, hạt nêm, mắm tôm, dầu ăn, mỡ nước, giấm, rượu vang đỏ, xì dầu, nước hàng.

---

## 2. Cấu trúc định dạng mẫu chuẩn (Format mẫu)

Dưới đây là cấu trúc mẫu của một món ăn trong tập dữ liệu (lưu ý trường `instructions` không chứa trường `image_url` của từng bước hướng dẫn nấu):

```json
[
  {
    "name": "Thịt bò nấu sốt vang",
    "description": "Thịt bò chín mềm, nước sốt màu hồng nâu, thơm mùi tỏi và hành tây.",
    "image_url": "images/thit_bo_nau_sot_vang.jpg",
    "cook_time_minutes": 18,
    "difficulty": "medium",
    "servings": 4,
    "cuisine_type": "Vietnamese",
    "diet_tags": ["Món mặn", "Thịt bò"],
    "ingredients": [
      {
        "name": "Thịt bò",
        "quantity": "1000",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c1"
      },
      {
        "name": "Hành tây",
        "quantity": "150",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c3"
      },
      {
        "name": "Tỏi",
        "quantity": "30",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c3"
      },
      {
        "name": "Cà chua",
        "quantity": "100",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c3"
      },
      {
        "name": "Rượu vang đỏ",
        "quantity": "100",
        "unit": "ml",
        "is_optional": false,
        "category_id": "c5"
      },
      {
        "name": "Mỡ nước",
        "quantity": "100",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c5"
      },
      {
        "name": "Bột mì",
        "quantity": "1",
        "unit": "muỗng xúp",
        "is_optional": false,
        "category_id": "c4"
      },
      {
        "name": "Muối",
        "quantity": "5",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c5"
      },
      {
        "name": "Hạt tiêu",
        "quantity": "2",
        "unit": "gram",
        "is_optional": false,
        "category_id": "c5"
      }
    ],
    "instructions": [
      {
        "step_number": 1,
        "title": "Tẩm ướp",
        "description": "Thịt bò thái vuông 3-4cm, ướp với muối, tiêu và rượu vang trong 60 phút.",
        "tip": "Nên dùng thịt thăn để món ăn mềm ngon."
      },
      {
        "step_number": 2,
        "title": "Xào nấu",
        "description": "Phi thơm hành tỏi với mỡ, cho bò vào xào săn rồi thêm cà chua băm nhỏ vào đun cùng.",
        "tip": null
      },
      {
        "step_number": 3,
        "title": "Hầm chín",
        "description": "Cho nước dùng vào hầm nhỏ lửa đến khi thịt bò chín nhừ, nước xốt sánh lại nhờ bột mỳ.",
        "tip": null
      }
    ],
    "source": "Kỹ thuật chế biến 300 món ăn ngon, trang 103"
  }
]
```

_Lưu ý: Đối với file `data_book_no_source.json`, trường `"source"` ở cuối mỗi món ăn sẽ được loại bỏ hoàn toàn._

\*Link sách:
1.Hướng dẫn nấu ăn 200 món truyền thống: https://drive.google.com/file/d/1ZmHVvNc3GS2Y1x_2mfIbSrPJHRS21sEl/view
2.Kỹ thuật chế biến 300 món ăn ngon
https://drive.google.com/file/d/1E_ClJA4dsnWPgouMSZOzfAQDzK8qp01y/view
3.555 món ăn Việt Nam
https://drive.google.com/file/d/1reCZ4lnyIJf_WqPmwOQACPUxyF_mfHvn/view
4.Nấu ăn gia đình miền Nam
https://drive.google.com/file/d/1gd_Ate-5sZSOAL1peluYNcfPpvu2R2FC/view
