from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


MODEL_PATH = Path("models/cnn-best-candidate_best.keras")
TEST_IMAGE = next(Path("data/splits/test/Cat").glob("*"))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


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
    assert TEST_IMAGE.exists()

    with TEST_IMAGE.open("rb") as image:
        response = client.post(
            "/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
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
