from pathlib import Path
import random
import shutil


SOURCE_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def split_files(files):
    """Shuffle files reproducibly and split into train/val/test."""
    files = files.copy()

    random.Random(RANDOM_SEED).shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_files(files, class_name, split_name):
    """Copy files into the appropriate split directory."""
    destination = SPLIT_DIR / split_name / class_name
    destination.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)


def main():
    print("Creating reproducible dataset split...")
    print(f"Random seed: {RANDOM_SEED}")

    for class_name in ["Cat", "Dog"]:
        source_class_dir = SOURCE_DIR / class_name

        files = [
            file_path
            for file_path in source_class_dir.iterdir()
            if file_path.is_file()
        ]

        train_files, val_files, test_files = split_files(files)

        copy_files(train_files, class_name, "train")
        copy_files(val_files, class_name, "val")
        copy_files(test_files, class_name, "test")

        print(f"\n{class_name}")
        print(f"  Total : {len(files)}")
        print(f"  Train : {len(train_files)}")
        print(f"  Val   : {len(val_files)}")
        print(f"  Test  : {len(test_files)}")


if __name__ == "__main__":
    main()
