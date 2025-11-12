import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import re
import glob


class Trainer:
    def __init__(self, dataset_dir, model_name):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.type_name = "five_char" if "five_char" in model_name else "single_char"

        # Pastas
        self.apurados_dir = "models/apurados"
        self.exported_dir = "models/exported"
        os.makedirs(self.apurados_dir, exist_ok=True)
        os.makedirs(self.exported_dir, exist_ok=True)

    def load_data(self):
        paths = {
            "x_train": os.path.join(self.dataset_dir, "x_train.npy"),
            "y_train": os.path.join(self.dataset_dir, "y_train.npy"),
            "x_val": os.path.join(self.dataset_dir, "x_val.npy"),
            "y_val": os.path.join(self.dataset_dir, "y_val.npy"),
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        X_train = np.load(paths["x_train"]).astype("float32") / 255.0
        X_val = np.load(paths["x_val"]).astype("float32") / 255.0
        y_train = np.load(paths["y_train"], allow_pickle=True)
        y_val = np.load(paths["y_val"], allow_pickle=True)

        if X_train.ndim == 3:
            X_train = np.expand_dims(X_train, -1)
            X_val = np.expand_dims(X_val, -1)

        y_train = self._normalize_labels(y_train)
        y_val = self._normalize_labels(y_val)
        return X_train, y_train, X_val, y_val

    def _normalize_labels(self, y_data):
        normalized = []
        for y in y_data:
            if isinstance(y, (list, np.ndarray)):
                normalized.append(''.join(str(i) for i in y))
            else:
                normalized.append(str(y))
        return normalized

    def build_model(self, num_classes):
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
            model.compile(optimizer="adam", loss=["categorical_crossentropy"]*5, metrics=["accuracy"]*5)
        return model

    def find_previous_model(self):
        pattern = re.compile(re.escape(self.model_name.replace(".keras", "")) + r"\.keras$")
        for file in os.listdir(self.apurados_dir):
            if pattern.match(file):
                return os.path.join(self.apurados_dir, file)
        return None

    def encode_labels(self, y_data, char_to_index):
        num_classes = len(char_to_index)
        if self.type_name == "single_char":
            return tf.keras.utils.to_categorical([char_to_index.get(c, 0) for c in y_data], num_classes)
        else:
            y_split = np.array([[char_to_index.get(c, char_to_index[' ']) for c in text.ljust(5)[:5]] for text in y_data])
            return [tf.keras.utils.to_categorical(y_split[:, i], num_classes) for i in range(5)]

    def run(self):
        X_train, y_train, X_val, y_val = self.load_data()

        all_chars = sorted(list(set(''.join(y_train) + ' ')))
        char_to_index = {c: i for i, c in enumerate(all_chars)}
        num_classes = len(all_chars)

        model = self.build_model(num_classes)

        # Reaproveita pesos do modelo anterior se existir
        prev_model_path = self.find_previous_model()
        if prev_model_path:
            print(f"♻️ Carregando pesos do modelo anterior: {prev_model_path}")
            model.load_weights(prev_model_path)

        y_train_enc = self.encode_labels(y_train, char_to_index)
        y_val_enc = self.encode_labels(y_val, char_to_index)

        model.fit(X_train, y_train_enc, validation_data=(X_val, y_val_enc),
                  epochs=20, batch_size=32, verbose=1)

        # Salva modelo individual
        save_path_apurados = os.path.join(self.apurados_dir, f"{self.model_name.replace('.keras','')}.keras")
        model.save(save_path_apurados)
        print(f"✅ Modelo salvo: {save_path_apurados}")

        return model  # retorna o modelo treinado


def salvar_modelo_unificado(modelos, saida_path):
    """
    Junta todos os modelos treinados em um único arquivo via Merge.
    O modelo unificado não treina, apenas consolida pesos.
    """
    print("\n🔗 Unificando modelos em:", saida_path)
    if not modelos:
        print("⚠️ Nenhum modelo foi passado para unificação.")
        return

    # Cria um modelo simbólico base igual ao primeiro
    base_model = modelos[0]
    input_shape = base_model.input_shape[1:]
    input_layer = layers.Input(shape=input_shape)

    # Concatena saídas de todos os modelos
    outputs = [m(input_layer) for m in modelos]
    merged_output = layers.Concatenate(axis=-1)(outputs)

    unified_model = models.Model(inputs=input_layer, outputs=merged_output)
    unified_model.save(saida_path)
    print(f"✅ Modelo unificado salvo em: {saida_path}")


if __name__ == "__main__":
    base_dir = "data/processed"
    modelos_treinados = []

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
                model = trainer.run()
                modelos_treinados.append(model)
            except FileNotFoundError as e:
                print(f"⚠️ Dataset incompleto — pulando: {e}")
            except Exception as e:
                print(f"💥 Erro ao treinar modelo {model_name}: {e}")

    # 🔥 Após todos os treinos, salva o modelo unificado
    if modelos_treinados:
        all_unified_path = os.path.join("models/apurados", "all_models_unified.keras")
        salvar_modelo_unificado(modelos_treinados, all_unified_path)
