import os
import cv2
import numpy as np
import torch
import uuid
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms

class VisionService:
    def __init__(self):
        print(" [Vision] Đang khởi tạo mô hình YOLOv8 và ResNet...")
        
        # Đường dẫn tuyệt đối tới thư mục chứa models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.debug_images_dir = os.path.join(self.base_dir, "data", "images")
        os.makedirs(self.debug_images_dir, exist_ok=True)
        self.last_annotated_image_path = None
        
        # 1. LOAD YOLOv8
        yolo_path = os.path.join(base_dir, 'models_weights', 'yolo_best.pt') 
        self.yolo_model = YOLO(yolo_path)

        # 2. LOAD RESNET
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Sửa thành ResNet-50 cho khớp với file trọng số của bạn
        self.resnet_model = models.resnet50(pretrained=False) 
        
        num_ftrs = self.resnet_model.fc.in_features
        
        # Sửa số class thành 38 cho khớp với model bạn đã train
        num_classes = 38 
        self.resnet_model.fc = torch.nn.Linear(num_ftrs, num_classes)

        
        resnet_path = os.path.join(base_dir, 'models_weights', 'resnet_best.pth')
        # Load weights (ánh xạ tự động sang CPU nếu không có GPU)
        self.resnet_model.load_state_dict(torch.load(resnet_path, map_location=self.device))
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

        # 3. TRANSFORMS (Chuẩn hóa ảnh cho ResNet)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

       # 4. CLASS MAPPING (Đã chuẩn hóa 38 class của bạn sang Tiếng Việt)
        self.class_names = {
            0: "Táo",               # apple
            1: "Chuối",             # banana
            2: "Thịt bò",           # beef
            3: "Củ dền",            # beetroot
            4: "Ớt chuông",         # bellpepper
            5: "Khổ qua",           # bittergourd
            6: "Bầu",               # bottlegourd
            7: "Súp lơ xanh",       # broccoli
            8: "Bắp cải",           # cabbage
            9: "Cà rốt",            # carrot
            10: "Súp lơ trắng",     # cauliflower
            11: "Su su",            # chayote
            12: "Thịt gà",          # chicken
            13: "Trứng gà",         # chickenegg
            14: "Đùi gà",           # chickenleg
            15: "Cánh gà",          # chickenwin
            16: "Ớt",               # chilli
            17: "Bắp ngô",          # corn
            18: "Dưa leo",          # cucumber
            19: "Trứng vịt",        # duckegg
            20: "Cà tím",           # eggplant
            21: "Tỏi",              # garlic
            22: "Gừng",             # ginger
            23: "Củ đậu",           # jicama (củ sắn)
            24: "Xà lách",          # lettuce
            25: "Đậu bắp",          # okra
            26: "Hành tây",         # onion
            27: "Dứa",              # pineapple (thơm)
            28: "Thịt heo",         # pork
            29: "Khoai tây",        # potato
            30: "Bí đỏ",            # pumpkin
            31: "Củ cải trắng",     # radish
            32: "Hành lá",          # scallion
            33: "Tôm",              # shrimp
            34: "Khoai lang",       # sweetpotato
            35: "Đậu hũ",           # tofu
            36: "Cà chua",          # tomato
            37: "Dưa hấu"           # watermelon
        }
        
    def predict_image(self, image_bytes):
        self.last_annotated_image_path = None

        # Decode ảnh từ bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            return []

        # YOLO nhận diện (Chỉ lấy kết quả độ tự tin > 25%)
        results = self.yolo_model(img_cv2, conf=0.25)
        detected_ingredients = []
        boxes = results[0].boxes
        annotated_img = img_cv2.copy()
        
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img_cv2[y1:y2, x1:x2]
            
            if cropped_img.size == 0:
                continue
                
            # Chuyển hệ màu OpenCV (BGR) -> PIL (RGB)
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cropped_img_rgb)
            
            # Chuẩn bị tensor cho ResNet
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # ResNet phân loại
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
                
                class_id = predicted_class.item()
                conf_score = confidence.item()
                
            # Chỉ chấp nhận phân loại nếu độ tin cậy > 40%
            if conf_score > 0.4: 
                ingredient_name = self.class_names.get(class_id, "unknown")
                detected_ingredients.append({
                    "name": ingredient_name,
                    "confidence": round(conf_score, 2)
                })

                label = f"{ingredient_name} {conf_score:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

        self.last_annotated_image_path = self._save_annotated_image(annotated_img)

        # Xóa trùng lặp (nhiều miếng gà thì chỉ tính là 1 loại nguyên liệu "Gà nguyên con")
        unique_items = {item['name']: item for item in detected_ingredients}.values()
        return list(unique_items)

    def _save_annotated_image(self, annotated_img):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"annotated_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(self.debug_images_dir, file_name)

        if cv2.imwrite(output_path, annotated_img):
            relative_path = os.path.relpath(output_path, self.base_dir)
            return relative_path.replace("\\", "/")

        return None