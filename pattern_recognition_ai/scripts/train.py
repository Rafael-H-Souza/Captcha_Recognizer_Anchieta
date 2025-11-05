import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime


class Trainer:
    def __init__(self, dataset_dir, model_name):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.type_name = "five_char" if "five_char" in model_name else "single_char"

    def load_data(self):
        """Carrega e valida os arquivos .npy necessários para o treinamento."""
        paths = {
            "x_train": os.path.join(self.dataset_dir, "x_train.npy"),
            "y_train": os.path.join(self.dataset_dir, "y_train.npy"),
            "x_val": os.path.join(self.dataset_dir, "x_val.npy"),
            "y_val": os.path.join(self.dataset_dir, "y_val.npy"),
        }

        # ⚠️ Verifica se todos os arquivos existem
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        # 🧩 Carrega os arrays
        X_train = np.load(paths["x_train"]).astype("float32") / 255.0
        X_val = np.load(paths["x_val"]).astype("float32") / 255.0
        y_train = np.load(paths["y_train"], allow_pickle=True)
        y_val = np.load(paths["y_val"], allow_pickle=True)

        # 🔧 Ajusta dimensões para (40, 100, 1)
        if X_train.ndim == 3:
            X_train = np.expand_dims(X_train, -1)
            X_val = np.expand_dims(X_val, -1)

        # 🧠 Normaliza labels
        y_train = self._normalize_labels(y_train)
        y_val = self._normalize_labels(y_val)

        return X_train, y_train, X_val, y_val

    def _normalize_labels(self, y_data):
        """Garante que as labels sejam strings legíveis."""
        normalized = []
        for y in y_data:
            if isinstance(y, (list, np.ndarray)):
                if all(isinstance(i, (int, np.integer)) for i in y):
                    text = ''.join(chr(i) for i in y if i > 0)
                else:
                    text = ''.join(str(i) for i in y)
                normalized.append(text)
            else:
                normalized.append(str(y))
        return normalized

    def build_model(self, num_classes):
        """Define o modelo CNN para single_char ou five_char."""
        inputs = layers.Input(shape=(50, 200, 1))

        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)

        if self.type_name == "single_char":
            outputs = layers.Dense(num_classes, activation="softmax", name="out")(x)
            model = models.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            outputs = [layers.Dense(num_classes, activation="softmax", name=f"out_{i}")(x) for i in range(5)]
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer="adam",
                loss=["categorical_crossentropy"] * 5,
                metrics=["accuracy"] * 5,
            )
        return model

    def run(self):
        """Executa o processo completo de treinamento."""
        X_train, y_train, X_val, y_val = self.load_data()

        # 🔤 Extrai caracteres únicos
        all_chars = sorted(list(set(''.join(y_train))))
        num_classes = len(all_chars)
        char_to_index = {c: i for i, c in enumerate(all_chars)}

        model = self.build_model(num_classes)

        if self.type_name == "single_char":
            # Codifica as labels
            y_train_enc = tf.keras.utils.to_categorical(
                [char_to_index.get(c, 0) for c in y_train], num_classes
            )
            y_val_enc = tf.keras.utils.to_categorical(
                [char_to_index.get(c, 0) for c in y_val], num_classes
            )

            model.fit(
                X_train,
                y_train_enc,
                validation_data=(X_val, y_val_enc),
                epochs=20,
                batch_size=32,
                verbose=1,
            )
        else:
            # Divide por posição
            y_train_split = np.array([[char_to_index.get(c, 0) for c in text.ljust(5)[:5]] for text in y_train])
            y_val_split = np.array([[char_to_index.get(c, 0) for c in text.ljust(5)[:5]] for text in y_val])

            y_train_onehot = [tf.keras.utils.to_categorical(y_train_split[:, i], num_classes) for i in range(5)]
            y_val_onehot = [tf.keras.utils.to_categorical(y_val_split[:, i], num_classes) for i in range(5)]

            model.fit(
                X_train,
                y_train_onehot,
                validation_data=(X_val, y_val_onehot),
                epochs=20,
                batch_size=32,
                verbose=1,
            )

        # 💾 Salva o modelo com timestamp
        os.makedirs("models/apurados", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = f"models/apurados/{self.model_name.replace('.keras', f'_{timestamp}.keras')}"
        model.save(save_path)
        print(f"✅ Modelo salvo em: {save_path}")


if __name__ == "__main__":
    base_dir = "data/processed"

    for type_name in ["single_char", "five_char"]:
        type_path = os.path.join(base_dir, type_name)
        if not os.path.exists(type_path):
            print(f"⚠️ Diretório ausente: {type_path}")
            continue

        for subtype in os.listdir(type_path):
            dataset_dir = os.path.join(type_path, subtype)
            if not os.path.isdir(dataset_dir):
                continue

            model_name = f"{type_name}_{subtype}_model.keras"
            print(f"\n🚀 Treinando modelo: {model_name} (dataset: {dataset_dir})")
            try:
                trainer = Trainer(dataset_dir, model_name)
                trainer.run()
            except FileNotFoundError as e:
                print(f"⚠️ Dataset incompleto — pulando: {e}")
            except Exception as e:
                print(f"💥 Erro ao treinar modelo {model_name}: {e}")
