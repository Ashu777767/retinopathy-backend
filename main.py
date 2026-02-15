import os
import io
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf

# =========================
# LOGGING (Console Only)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Diabetic Retinopathy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL LOADING
# =========================
MODEL_PATH = "retinopathy_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise RuntimeError("Model failed to load")

# =========================
# CONFIG
# =========================
IMG_SIZE = 224

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferative_DR",
    "Severe"
]

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "Backend running 🚀"}

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)

        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        logger.info(f"Prediction: {predicted_class} ({confidence:.4f})")

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
