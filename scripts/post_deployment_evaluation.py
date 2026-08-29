#!/usr/bin/env python3

import csv
import os
import sys
import time
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


BASE_URL = os.getenv(
    "API_URL",
    "http://192.168.49.2:32205",
)

DATASET_DIR = Path("data/processed")
OUTPUT_FILE = Path("reports/post_deployment_performance.csv")

SAMPLES_PER_CLASS = 20


def collect_images():
    images = []

    for label, class_name in [(0, "Cat"), (1, "Dog")]:
        class_dir = DATASET_DIR / class_name

        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        if len(files) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Not enough images for {class_name}"
            )

        for path in files[:SAMPLES_PER_CLASS]:
            images.append((path, label, class_name))

    return images


def main():
    print("=" * 60)
    print("Post-Deployment Model Performance Evaluation")
    print("=" * 60)
    print(f"API: {BASE_URL}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")

    # Verify API
    health = requests.get(
        f"{BASE_URL}/health",
        timeout=30,
    )
    health.raise_for_status()

    health_data = health.json()

    if not health_data.get("model_loaded"):
        raise RuntimeError("Model is not loaded")

    print("API health check: PASS")

    images = collect_images()

    y_true = []
    y_pred = []
    rows = []

    for index, (image_path, true_label, true_class) in enumerate(
        images, start=1
    ):
        start = time.perf_counter()

        with open(image_path, "rb") as image_file:
            response = requests.post(
                f"{BASE_URL}/predict",
                files={
                    "file": (
                        image_path.name,
                        image_file,
                        "image/jpeg",
                    )
                },
                timeout=60,
            )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        response.raise_for_status()

        result = response.json()

        predicted_label = int(result["prediction"])
        predicted_class = result["label"]
        probability = float(result["probability"])

        y_true.append(true_label)
        y_pred.append(predicted_label)

        rows.append(
            {
                "image": str(image_path),
                "true_label": true_class,
                "predicted_label": predicted_class,
                "probability": round(probability, 4),
                "latency_ms": round(latency_ms, 2),
            }
        )

        print(
            f"[{index:02d}/{len(images)}] "
            f"{true_class:3s} -> "
            f"{predicted_class:3s} "
            f"({probability:.4f}) "
            f"{latency_ms:.2f} ms"
        )

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "true_label",
                "predicted_label",
                "probability",
                "latency_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("POST-DEPLOYMENT PERFORMANCE")
    print("=" * 60)
    print(f"Samples       : {len(y_true)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    print()
    print("Confusion Matrix")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(cm)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
