from pathlib import Path
from PIL import Image
import csv


RAW_DIR = Path("data/raw/PetImages")
REPORT_DIR = Path("reports")
REPORT_FILE = REPORT_DIR / "data_validation_report.csv"


def validate_images():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for class_name in ["Cat", "Dog"]:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()

                results.append(
                    {
                        "file": str(file_path),
                        "class": class_name,
                        "status": "valid",
                        "error": "",
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "file": str(file_path),
                        "class": class_name,
                        "status": "invalid",
                        "error": str(exc),
                    }
                )

    with REPORT_FILE.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["file", "class", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    valid_count = sum(
        1 for result in results if result["status"] == "valid"
    )

    invalid_count = sum(
        1 for result in results if result["status"] == "invalid"
    )

    print("Data validation completed")
    print(f"Total files checked : {len(results)}")
    print(f"Valid files         : {valid_count}")
    print(f"Invalid files       : {invalid_count}")
    print(f"Report              : {REPORT_FILE}")


if __name__ == "__main__":
    validate_images()
