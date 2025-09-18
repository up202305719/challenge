from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model import load_model, predict_image

app = FastAPI(title="Classificador de Mama")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("breast_resnet18.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        pred, probs = predict_image(model, img)
        return JSONResponse({"prediction": pred, "probabilities": probs})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
