from tensorflow.keras.layers import (
    Bidirectional, Conv2D, Dense, Dropout, GRU, Input,
    MaxPooling2D, Reshape, BatchNormalization, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
import tensorflow as tf

# ======================================================
# ⚙️ CONFIGURAÇÃO OPCIONAL PARA GPU (Mixed Precision)
# ======================================================
def enable_gpu_acceleration():
    """Ativa mixed precision e memória dinâmica da GPU."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            mixed_precision.set_global_policy("mixed_float16")
            print("🚀 GPU acceleration ativada com mixed precision.")
        else:
            print("⚠️ Nenhuma GPU detectada — usando CPU.")
    except Exception as e:
        print(f"⚠️ Falha ao configurar GPU: {e}")


# ======================================================
# 🧠 MODELO CNN + GRU (UNIFICADO PARA CPU/GPU)
# ======================================================
def build_cnn_gru_model(config):
    """
    Modelo CNN + GRU para reconhecimento de CAPTCHA.
    Recebe um dicionário de configuração com:
      - image_height, image_width, image_channels
      - max_length
      - num_classes
    """

    input_shape = (
        config["image_height"],
        config["image_width"],
        config.get("image_channels", 1),
    )
    max_length = config["max_length"]
    num_classes = config["num_classes"]

    # Entrada
    inputs = Input(shape=input_shape, name="image_input", dtype="float32")

    # 🧱 CNN
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # 🔄 Flatten e reshape para GRU
    x = Flatten()(x)
    x = Reshape((max_length, -1))(x)

    # 🔁 GRU bidirecional
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)

    # 🔤 Saída por caractere
    outputs = [
        Dense(num_classes, activation="softmax", name=f"char_{i+1}")(x[:, i, :])
        for i in range(max_length)
    ]

    model = Model(inputs=inputs, outputs=outputs, name="cnn_gru_captcha")
    return model
