from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("data/splits")
MODEL_PATH = Path("models/cnn-best-candidate_best.keras")
REPORT_DIR = Path("reports")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

def create_test_dataset():
    """Create the test dataset without shuffling."""

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    return test_ds


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, test_ds):
    """Evaluate the trained model on the independent test set."""

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probabilities = model.predict(
            images,
            verbose=0,
        ).ravel()

        y_prob.extend(probabilities)
        y_true.extend(labels.numpy().ravel())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

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

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # Reproducibility
    tf.keras.utils.set_random_seed(SEED)

    REPORT_DIR.mkdir(
        exist_ok=True,
    )

    # Check model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\n==============================")
    print("Final Model Evaluation")
    print("==============================")
    print(f"Model : {MODEL_PATH}")
    print(f"Image : {IMG_SIZE}")
    print(f"Batch : {BATCH_SIZE}")
    print("==============================\n")

    # Load model
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
    )

    print("Model loaded successfully.")

    # Load test data
    test_ds = create_test_dataset()

    # Evaluate
    (
        accuracy,
        precision,
        recall,
        f1,
        cm,
    ) = evaluate_model(
        model,
        test_ds,
    )

    # Print results
    print("\n==============================")
    print("Final Evaluation Results")
    print("==============================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics
    metrics_path = (
        REPORT_DIR / "final_model_evaluation.txt"
    )

    with metrics_path.open("w") as file:

        file.write("Final Model Evaluation\n")
        file.write("======================\n\n")

        file.write(
            f"Model: {MODEL_PATH}\n"
        )

        file.write(
            f"Image Size: {IMG_SIZE}\n"
        )

        file.write(
            f"Batch Size: {BATCH_SIZE}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall: {recall:.4f}\n"
        )

        file.write(
            f"F1 Score: {f1:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix\n"
        )

        file.write(
            "================\n"
        )

        file.write(
            "Rows = Actual [Cat, Dog]\n"
        )

        file.write(
            "Columns = Predicted [Cat, Dog]\n\n"
        )

        file.write(
            str(cm)
        )

        file.write("\n")

    print(
        f"\nEvaluation report saved to: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()
