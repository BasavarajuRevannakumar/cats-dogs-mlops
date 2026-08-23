import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),

            layers.Dense(1, activation="sigmoid"),
        ],
        name="cats_dogs_cnn",
    )

    return model