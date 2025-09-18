import os
import numpy as np
from medmnist import INFO, BreastMNIST

# Configurações
DATA_DIR = "datasets/medmnist_test"
os.makedirs(DATA_DIR, exist_ok=True)

# Classes
classes = ["Normal", "Benigno", "Maligno"]

for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# Dataset de teste
info = INFO["breastmnist"]
test_ds = BreastMNIST(split='test', download=True)

# Salvar imagens
for idx in range(len(test_ds)):
    img, label = test_ds[idx]
    
    # Extrair valor do label
    if hasattr(label, "item"):  # se for numpy array
        label = label.item()
    
    label_name = classes[int(label)]
    
    # Se já for PIL Image, salva diretamente
    if hasattr(img, "save"):
        img_pil = img
    else:
        from PIL import Image
        import numpy as np
        img_pil = Image.fromarray(np.array(img))
    
    filename = f"{idx}_{label_name}.png"
    img_pil.save(os.path.join(DATA_DIR, label_name, filename))

print("Download e organização concluídos!")
