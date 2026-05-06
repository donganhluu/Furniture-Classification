import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# CONFIG

image_path = r"D:/Furniture_classification/Test_img/AXCDNT-003-OAK_Silo_Detail_1.webp"   
model_path = r"D:/Furniture_classification/ResNet18_full/best_resnet18.pth"

class_names = ['cabinet', 'chair', 'sofa', 'table']  # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TRANSFORM (GIỐNG VAL)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# LOAD MODEL

model = models.resnet18(pretrained=False)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# LOAD IMAGE

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)


# INFERENCE

with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)

    confidence, pred = torch.max(probs, 1)

pred_class = class_names[pred.item()]
conf = confidence.item()


# RESULT

print(" Image:", image_path)
print(f" Prediction: {pred_class}")
print(f" Confidence: {conf:.4f}")