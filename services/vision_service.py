import os
import cv2
import numpy as np
import torch

# Monkey-patch torch.load cho PyTorch 2.6+ để mặc định weights_only=False,
# giúp load các file weights YOLO & ResNet một cách mượt mà không bị lỗi unpickle.
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms

class VisionService:
    def __init__(self):
        print(" [Vision] Đang khởi tạo mô hình YOLOv8 và ResNet...")
        
        # Đường dẫn tuyệt đối tới thư mục chứa models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)

        # Thử tìm ở D:\NCKH\Code\models trước, sau đó là D:\NCKH\Code\food-ai-service\models_weights
        yolo_candidates = [
            os.path.join(parent_dir, 'models', 'yolo_best.pt'),
            os.path.join(base_dir, 'models_weights', 'yolo_best.pt'),
        ]
        resnet_candidates = [
            os.path.join(parent_dir, 'models', 'resnet_best.pth'),
            os.path.join(base_dir, 'models_weights', 'resnet_best.pth'),
        ]

        yolo_path = next((p for p in yolo_candidates if os.path.exists(p)), yolo_candidates[0])
        resnet_path = next((p for p in resnet_candidates if os.path.exists(p)), resnet_candidates[0])

        print(f" [Vision] Sử dụng YOLO từ: {yolo_path}")
        print(f" [Vision] Sử dụng ResNet từ: {resnet_path}")

        # 1. LOAD YOLOv8
        self.yolo_model = YOLO(yolo_path)

        # 2. LOAD RESNET
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Sửa thành ResNet-50 cho khớp với file trọng số của bạn
        self.resnet_model = models.resnet50(pretrained=False) 
        
        num_ftrs = self.resnet_model.fc.in_features
        
        # Sửa số class thành 38 cho khớp với model bạn đã train
        num_classes = 38 
        self.resnet_model.fc = torch.nn.Linear(num_ftrs, num_classes)

        # Load weights (ánh xạ tự động sang CPU nếu không có GPU)
        self.resnet_model.load_state_dict(torch.load(resnet_path, map_location=self.device, weights_only=False))
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
        # Decode ảnh từ bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            return []

        # YOLO nhận diện (Chỉ lấy kết quả độ tự tin > 25%)
        results = self.yolo_model(img_cv2, conf=0.25)
        detected_ingredients = []
        boxes = results[0].boxes
        
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

        # Xóa trùng lặp (nhiều miếng gà thì chỉ tính là 1 loại nguyên liệu "Gà nguyên con")
        unique_items = {item['name']: item for item in detected_ingredients}.values()
        return list(unique_items)