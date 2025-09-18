import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

labels = ["Normal", "Benigno", "Maligno"]

# Carregar modelo
def load_model(path="breast_resnet18.pt"):
    # Caminho absoluto relativo ao backend
    path = os.path.join(os.path.dirname(__file__), path)
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Predição
def predict_image(model, img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].numpy().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return labels[pred_idx], {labels[i]: round(p*100,1) for i, p in enumerate(probs)}
