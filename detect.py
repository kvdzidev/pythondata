import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import sys

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def load_model(model_path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = "model_v1.pth"
    model = load_model(model_path)
    img = process_image(image_path)
    output = model(img)
    _, predicted = torch.max(output, 1)

    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    print(f"Predicted: {classes[predicted.item()]}")
