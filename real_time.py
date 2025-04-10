import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np

# Model Architecture (EfficientNetV2Lite)
class EfficientNetV2Lite(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2Lite, self).__init__()
        self.features = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        ).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Transform defined(as used for inference)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Load Model
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetV2Lite(num_classes=num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

class_map = {0: "O", 1: "R"}

# Start Webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture a photo for classification. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break
    
    cv2.imshow("Live Feed", frame)
    
    key = cv2.waitKey(1)
    # When user presses 'c', take photo for inference
    if key % 256 == ord('c'):
        captured_frame = frame.copy()
        image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = inference_transform(pil_img)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
        pred_class = preds.item()
        result = class_map.get(pred_class, "Unknown")
        print(f"Prediction: {result}")
        
        cv2.putText(captured_frame, f"Prediction: {result}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Captured Frame", captured_frame)
        cv2.waitKey(2000)
    
    # Quit the loop if the user has pressed 'q'
    elif key % 256 == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
