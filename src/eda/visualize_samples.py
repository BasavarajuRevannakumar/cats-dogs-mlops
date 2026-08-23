from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATASET_DIR = Path("data/splits")
RANDOM_SEED = 42


def get_samples(class_name, count=5):
    directory = DATASET_DIR / "train" / class_name

    files = [
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file()
    ]

    random.seed(RANDOM_SEED)
    return random.sample(files, count)


def main():
    cat_images = get_samples("Cat")
    dog_images = get_samples("Dog")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for index, file_path in enumerate(cat_images):
        image = Image.open(file_path)

        axes[0, index].imshow(image)
        axes[0, index].set_title("Cat")
        axes[0, index].axis("off")

    for index, file_path in enumerate(dog_images):
        image = Image.open(file_path)

        axes[1, index].imshow(image)
        axes[1, index].set_title("Dog")
        axes[1, index].axis("off")

    plt.tight_layout()

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "sample_images.png"
    plt.savefig(output_file, dpi=150)

    print(f"Saved visualization: {output_file}")


if __name__ == "__main__":
    main()
