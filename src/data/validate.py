from pathlib import Path
import csv

import tensorflow as tf


RAW_DIR = Path("data/raw/PetImages")
REPORT_DIR = Path("reports")
REPORT_FILE = REPORT_DIR / "data_validation_report.csv"


def validate_image(file_path: Path) -> tuple[bool, str]:
    """Validate image using TensorFlow decoder."""

    try:
        image_bytes = tf.io.read_file(str(file_path))
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        # Force TensorFlow to materialize/decode the image.
        _ = image.numpy()

        return True, ""

    except Exception as exc:
        return False, str(exc)


def validate_images():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for class_name in ["Cat", "Dog"]:

        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {class_dir}"
            )

        for file_path in sorted(class_dir.iterdir()):

            if not file_path.is_file():
                continue

            is_valid, error = validate_image(file_path)

            results.append(
                {
                    "file": str(file_path),
                    "class": class_name,
                    "status": "valid" if is_valid else "invalid",
                    "error": error,
                }
            )

    with REPORT_FILE.open("w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "file",
                "class",
                "status",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    valid_count = sum(
        result["status"] == "valid"
        for result in results
    )

    invalid_count = sum(
        result["status"] == "invalid"
        for result in results
    )

    print("\nData validation completed")
    print("==========================")
    print(f"Total files checked : {len(results)}")
    print(f"Valid files         : {valid_count}")
    print(f"Invalid files       : {invalid_count}")
    print(f"Report              : {REPORT_FILE}")


if __name__ == "__main__":
    validate_images()