import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import INFO, BreastMNIST

# Número de classes do BREASTMNIST
n_classes = 3  # 0: Normal, 1: Benigno, 2: Maligno

# Configuração do dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5])
])

# Download e carregamento do dataset
train_ds = BreastMNIST(split='train', transform=transform, download=True)
val_ds   = BreastMNIST(split='val', transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Criar modelo ResNet18 adaptado para grayscale e 3 classes
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loop de treino simples
epochs = 5  # número de épocas de teste rápido
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validação rápida
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {correct/total:.3f}")

# Salvar modelo treinado
torch.save(model.state_dict(), "breast_resnet18.pt")
print("Modelo salvo em breast_resnet18.pt")
