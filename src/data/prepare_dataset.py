from pathlib import Path
from shutil import copy2

from PIL import Image


RAW_DIR = Path("data/raw/PetImages")
PROCESSED_DIR = Path("data/processed")


def is_valid_image(file_path: Path) -> bool:
    """Return True if the file is a valid image."""
    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def prepare_dataset():
    total = 0
    valid = 0
    invalid = 0

    for class_name in ["Cat", "Dog"]:
        source_dir = RAW_DIR / class_name
        target_dir = PROCESSED_DIR / class_name

        target_dir.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(source_dir.iterdir()):
            if not file_path.is_file():
                continue

            total += 1

            if not is_valid_image(file_path):
                invalid += 1
                print(f"Skipping invalid file: {file_path}")
                continue

            destination = target_dir / file_path.name
            copy2(file_path, destination)
            valid += 1

    print("\nDataset preparation completed")
    print(f"Total files checked : {total}")
    print(f"Valid files copied  : {valid}")
    print(f"Invalid files skipped: {invalid}")
    print(f"Processed directory : {PROCESSED_DIR}")


if __name__ == "__main__":
    prepare_dataset()
