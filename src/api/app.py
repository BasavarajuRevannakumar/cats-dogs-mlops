
from pathlib import Path

import io

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_PATH = Path("models/cnn-best-candidate_best.keras")
IMG_SIZE = (224, 224)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Cats vs Dogs CNN API",
    description="Image classification API for Cats vs Dogs",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Load model once when the API starts
# -------------------------------------------------------------------

model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Input shape: {model.input_shape}")


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    # Validate content type
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPEG or PNG.",
        )

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        image_array = np.array(image, dtype=np.float32)

        # Model contains Rescaling(1/255), so do NOT normalize here.
        image_array = np.expand_dims(image_array, axis=0)

        probability = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        prediction = int(probability >= 0.5)

        label = "Dog" if prediction == 1 else "Cat"

        return {
            "prediction": prediction,
            "label": label,
            "probability": round(probability, 4),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {exc}",
        )


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Cats vs Dogs CNN API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

