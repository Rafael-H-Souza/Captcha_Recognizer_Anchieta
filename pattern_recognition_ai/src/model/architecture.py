from tensorflow.keras.layers import (Bidirectional, Conv2D, Dense, Dropout,
                                     GRU, Input, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model


def build_cnn_gru_model(config):
    """Constrói um modelo CNN + GRU para reconhecimento de CAPTCHA."""
    input_shape = (
        config["image_height"],
        config["image_width"],
        config["image_channels"],
    )
    max_length = config["max_length"]
    num_classes = config["num_classes"]

    inputs = Input(shape=input_shape, name="image_input", dtype="float32")

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)

    outputs = [Dense(num_classes, activation="softmax", name=f"char_{i + 1}")(x) for i in range(max_length)]

    model = Model(inputs=inputs, outputs=outputs, name="captcha_solver")
    return model
