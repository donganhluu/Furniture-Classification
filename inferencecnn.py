import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# MODEL 

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, 1).item()

    classes = ["chair", "table", "sofa", "cabinet"]

    print("Predicted class:", classes[pred])
    print("Confidence:", probs[0][pred].item())

    return classes[pred]


if __name__ == "__main__":
    image_path = r"D:\Furniture_classification\Test_img\aqua-sofa-japandi-cover-photo-2.jpg"
    model_path = r"D:\Furniture_classification\CNN\best_cnn.pth"

    predict(image_path, model_path)