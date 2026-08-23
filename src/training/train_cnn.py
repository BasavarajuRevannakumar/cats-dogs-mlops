from pathlib import Path
import argparse

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.cnn_model import build_cnn


DATASET_DIR = Path("data/splits")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

IMG_SIZE = (224, 224)

# Baseline defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.5
DEFAULT_RUN_NAME = "cnn-baseline-improved"

SEED = 42


def parse_arguments():
    """Parse command-line hyperparameters."""

    parser = argparse.ArgumentParser(
        description="Train CNN for Cats vs Dogs classification"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"Dropout rate (default: {DEFAULT_DROPOUT})",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=f"MLflow run name (default: {DEFAULT_RUN_NAME})",
    )

    return parser.parse_args()


def create_datasets(batch_size):
    """Create reproducible train, validation and test datasets."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def evaluate_model(model, test_ds):
    """Calculate classification metrics on the test dataset."""

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

    y_pred = (
        np.array(y_prob) >= 0.5
    ).astype(int)

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

    return precision, recall, f1, cm


def main():

    args = parse_arguments()

    # Reproducibility
    tf.keras.utils.set_random_seed(SEED)

    print("\n==============================")
    print("CNN Training Configuration")
    print("==============================")
    print(f"Learning rate : {args.learning_rate}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Dropout       : {args.dropout}")
    print(f"Run name      : {args.run_name}")
    print(f"Seed          : {SEED}")
    print("==============================\n")

    train_ds, val_ds, test_ds = create_datasets(
        args.batch_size
    )

    model = build_cnn(
        input_shape=(*IMG_SIZE, 3),
        dropout_rate=args.dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Directories
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # MLflow
    mlflow.set_experiment("cats-dogs-cnn")

    with mlflow.start_run(
        run_name=args.run_name
    ):

        # Log hyperparameters
        mlflow.log_params(
            {
                "model": "CNN",
                "image_size": "224x224",
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "optimizer": "Adam",
                "seed": SEED,
                "early_stopping_patience": 3,
            }
        )

        # Save best model based on validation loss
        checkpoint_path = (
            MODEL_DIR / f"{RUN_NAME}_best.keras"
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )

        # Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[
                checkpoint,
                early_stopping,
            ],
        )

        # Log training history
        for epoch in range(
            len(history.history["loss"])
        ):

            mlflow.log_metric(
                "train_loss",
                float(
                    history.history["loss"][epoch]
                ),
                step=epoch,
            )

            mlflow.log_metric(
                "train_accuracy",
                float(
                    history.history["accuracy"][epoch]
                ),
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                float(
                    history.history["val_loss"][epoch]
                ),
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                float(
                    history.history["val_accuracy"][epoch]
                ),
                step=epoch,
            )

        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(
            test_ds,
            verbose=1,
        )

        precision, recall, f1, cm = evaluate_model(
            model,
            test_ds,
        )

        # Log test metrics
        mlflow.log_metrics(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "test_precision": float(precision),
                "test_recall": float(recall),
                "test_f1": float(f1),
            }
        )

        # Save confusion matrix
        cm_path = (
            REPORT_DIR / "confusion_matrix.txt"
        )

        with cm_path.open("w") as file:

            file.write("Confusion Matrix\n")
            file.write("================\n")
            file.write(
                "Rows = Actual [Cat, Dog]\n"
                "Columns = Predicted [Cat, Dog]\n\n"
            )

            file.write(str(cm))
            file.write("\n")

        mlflow.log_artifact(
            str(cm_path)
        )

        # Save training history
        history_path = (
            REPORT_DIR / "training_history.txt"
        )

        with history_path.open("w") as file:

            for epoch in range(
                len(history.history["loss"])
            ):

                file.write(
                    f"Epoch {epoch + 1}: "
                    f"loss="
                    f"{history.history['loss'][epoch]:.4f}, "
                    f"accuracy="
                    f"{history.history['accuracy'][epoch]:.4f}, "
                    f"val_loss="
                    f"{history.history['val_loss'][epoch]:.4f}, "
                    f"val_accuracy="
                    f"{history.history['val_accuracy'][epoch]:.4f}\n"
                )

        mlflow.log_artifact(
            str(history_path)
        )

        # Save final best model
        final_model_path = (
            MODEL_DIR / f"{RUN_NAME}_final.keras"
        )

        model.save(final_model_path)

        mlflow.keras.log_model(
            model,
            name="cnn_final",
        )

        # Final results
        print("\n==============================")
        print("Final Test Results")
        print("==============================")
        print(
            f"Test Loss     : {test_loss:.4f}"
        )
        print(
            f"Test Accuracy : {test_accuracy:.4f}"
        )
        print(
            f"Precision     : {precision:.4f}"
        )
        print(
            f"Recall        : {recall:.4f}"
        )
        print(
            f"F1 Score      : {f1:.4f}"
        )

        print("\nConfusion Matrix:")
        print(cm)

        print(
            f"\nBest model saved to: "
            f"{checkpoint_path}"
        )

        print(
            f"Final model saved to: "
            f"{final_model_path}"
        )


if __name__ == "__main__":
    main()