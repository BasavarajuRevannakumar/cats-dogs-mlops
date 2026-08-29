from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app


MODEL_PATH = Path("models/cnn-best-candidate_best.keras")


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def create_test_image():
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Cats vs Dogs CNN API"
    assert data["version"] == "1.0.0"


def test_model_exists():
    assert MODEL_PATH.exists()


def test_predict_valid_image(client):
    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "probability" in data

    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Cat", "Dog"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_file_type(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"This is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "prediction_requests" in data
    assert "average_prediction_latency_ms" in data

    assert data["total_requests"] >= 1
    assert data["successful_requests"] >= 0
    assert data["failed_requests"] >= 0
    assert data["prediction_requests"] >= 0
    assert data["average_prediction_latency_ms"] >= 0
