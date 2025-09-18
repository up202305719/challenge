# Plataforma de Classificação de Tecido Mamário

## Objetivo

Desenvolver uma plataforma simples que permita aos médicos ou usuários enviar imagens de tecido mamário e receber uma classificação imediata em três categorias: **Normal**, **Benigno** ou **Maligno**.

---

## Estrutura da Plataforma

A plataforma é composta por três partes principais:

1. **Backend (FastAPI)**
   - Recebe imagens enviadas pelo frontend.
   - Carrega o modelo pré-treinado `breast_resnet18.pt` treinado com o dataset **BREASTMNIST**.
   - Retorna a predição e as probabilidades de cada classe em formato JSON.

2. **Frontend (HTML/CSS/JS)**
   - Permite ao usuário selecionar uma imagem através de um formulário de upload.
   - Envia a imagem para o backend usando `fetch`.
   - Mostra o resultado de forma visual, com cores diferentes para cada classe:
     - Verde → Normal  
     - Amarelo → Benigno  
     - Vermelho → Maligno  

3. **Dataset de Teste (BREASTMNIST)**
   - Imagens separadas por classe: `Normal`, `Benigno`, `Maligno`.
   - Usadas para testar manualmente a plataforma sem influenciar o modelo.

---

## Funcionamento

1. **Upload de Imagem**  
   O usuário seleciona uma imagem local e clica em “Enviar”.

2. **Predição pelo Backend**  
   - O backend recebe a imagem via endpoint `/predict`.
   - A imagem é pré-processada e passada para o modelo ResNet18 treinado.
   - O modelo retorna:
     - **Classe prevista**  
     - **Probabilidades de cada classe**

3. **Exibição no Frontend**  
   - A interface apresenta a predição em destaque e uma lista de probabilidades.
   - As cores indicam a gravidade da predição para melhor visualização.

---

## Como Testar

1. Rodar o backend:

```bash
cd backend
uvicorn app:app --reload


2. Selecionar uma imagem do dataset de teste ou do computador e enviar.

3. O resultado aparecerá imediatamente na interface.
