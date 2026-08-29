from pathlib import Path
import io
import logging
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from collections import Counter

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_PATH = Path("models/cnn-best-candidate_best.keras")
IMG_SIZE = (224, 224)


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("cats-dogs-api")


# -------------------------------------------------------------------
# Monitoring metrics
# -------------------------------------------------------------------

metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "prediction_requests": 0,
    "total_prediction_latency_ms": 0.0,
}


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Cats vs Dogs CNN API",
    description="Image classification API for Cats vs Dogs",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Request logging and latency middleware
# -------------------------------------------------------------------

@app.middleware("http")
async def monitoring_middleware(request, call_next):
    start_time = time.perf_counter()

    metrics["total_requests"] += 1

    try:
        response = await call_next(request)

        if response.status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "%s %s | status=%s | latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        return response

    except Exception:
        metrics["failed_requests"] += 1

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "%s %s | status=500 | latency_ms=%.2f",
            request.method,
            request.url.path,
            latency_ms,
        )

        raise


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

    logger.info("Model loaded from: %s", MODEL_PATH)
    logger.info("Input shape: %s", model.input_shape)


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
# Monitoring metrics
# -------------------------------------------------------------------

@app.get("/metrics")
def get_metrics():
    prediction_count = metrics["prediction_requests"]

    if prediction_count > 0:
        average_latency = (
            metrics["total_prediction_latency_ms"]
            / prediction_count
        )
    else:
        average_latency = 0.0

    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "prediction_requests": prediction_count,
        "average_prediction_latency_ms": round(
            average_latency, 2
        ),
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

    start_time = time.perf_counter()

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

        latency_ms = (time.perf_counter() - start_time) * 1000

        metrics["prediction_requests"] += 1
        metrics["total_prediction_latency_ms"] += latency_ms

        # Do not log image contents or uploaded data.
        logger.info(
            "prediction | label=%s | probability=%.4f | latency_ms=%.2f",
            label,
            probability,
            latency_ms,
        )

        return {
            "prediction": prediction,
            "label": label,
            "probability": round(probability, 4),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unable to process prediction request")

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
        "metrics": "/metrics",
    }
