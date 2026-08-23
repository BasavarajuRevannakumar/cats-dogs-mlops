from pathlib import Path
from collections import Counter

from PIL import Image


DATASET_DIR = Path("data/splits")
SPLITS = ["train", "val", "test"]
CLASSES = ["Cat", "Dog"]


def analyze_split(split_name):
    print(f"\n{'=' * 50}")
    print(f"Split: {split_name}")
    print(f"{'=' * 50}")

    all_widths = []
    all_heights = []
    formats = Counter()

    for class_name in CLASSES:
        class_dir = DATASET_DIR / split_name / class_name

        files = [
            path for path in class_dir.iterdir()
            if path.is_file()
        ]

        print(f"{class_name}: {len(files)} images")

        for file_path in files:
            try:
                with Image.open(file_path) as image:
                    width, height = image.size

                    all_widths.append(width)
                    all_heights.append(height)

                    if image.format:
                        formats[image.format] += 1

            except Exception as exc:
                print(f"Could not read {file_path}: {exc}")

    if all_widths:
        print(f"\nWidth:")
        print(f"  Min    : {min(all_widths)}")
        print(f"  Max    : {max(all_widths)}")
        print(f"  Average: {sum(all_widths) / len(all_widths):.2f}")

        print(f"\nHeight:")
        print(f"  Min    : {min(all_heights)}")
        print(f"  Max    : {max(all_heights)}")
        print(f"  Average: {sum(all_heights) / len(all_heights):.2f}")

    print(f"\nImage formats:")
    for image_format, count in formats.items():
        print(f"  {image_format}: {count}")


def main():
    print("Cats vs Dogs - Image EDA")

    for split in SPLITS:
        analyze_split(split)


if __name__ == "__main__":
    main()
