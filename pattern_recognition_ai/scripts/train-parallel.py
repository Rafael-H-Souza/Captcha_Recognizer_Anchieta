#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treinamento robusto e unificado de modelos CAPTCHA
Versão 7.0 — CPU Turbo Mode + tf.data + shape adaptativo + treinamento paralelo compatível
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ===========================================================
# 🚀 MODO TURBO — Aceleração pesada em CPU
# ===========================================================

# Mixed precision (mesmo em CPU reduz custo de memória)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# XLA JIT – compila o modelo para acelerar execução
tf.config.optimizer.set_jit(True)

# Número automático de threads de pipeline
AUTOTUNE = tf.data.AUTOTUNE

# ===========================================================
# 🔹 Configurações globais
# ===========================================================
IMG_HEIGHT = 50
IMG_WIDTH_SINGLE = 40      # <-- width correto para single_char
IMG_WIDTH_FIVE = 200       # <-- width correto para five_char
IMG_CHANNELS = 1

PAD_CHAR = "_"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ===========================================================
# 🔹 Funções auxiliares
# ===========================================================

def clean_label(y):
    """Converte arrays de ord() em string removendo padding."""
    if isinstance(y, str):
        return y
    if isinstance(y, (list, np.ndarray)):
        return "".join(chr(int(v)) for v in y if int(v) != 0)
    return str(y)


def pad_5chars(text: str) -> str:
    text = (text or "").strip()
    if len(text) < 5:
        return text.ljust(5, PAD_CHAR)
    return text[:5]


def normalize_single_char_label(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace(PAD_CHAR, "")
    return t[0] if t else ""


def sane_char_set(y_train, y_val, type_name, subtype):
    joined = "".join(y_train + y_val)
    chars = {c for c in joined if c and not c.isspace()}

    if type_name == "five_char":
        chars.add(PAD_CHAR)

    if len(chars) < 2:
        raise ValueError(
            f"🚨 Charset insuficiente ({len(chars)}) para {type_name}/{subtype}: {sorted(chars)}"
        )

    char_list = sorted(chars)
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    return char_list, char_to_idx


# ===========================================================
# 🔹 Classe Trainer
# ===========================================================

class Trainer:
    def __init__(self, dataset_dir: str, model_name: str):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.type_name = "five_char" if "five_char" in model_name else "single_char"

        self.subtype = Path(dataset_dir).parts[-1]
        self.apurados_dir = Path("models/apurados")
        self.apurados_dir.mkdir(parents=True, exist_ok=True)

        self.char_set = None
        self.char_to_idx = None
        self.num_classes = None

    # =====================================================
    #  Carregar dataset
    # =====================================================
    def load_data(self):
        req = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy"]
        for f in req:
            if not (Path(self.dataset_dir) / f).exists():
                raise FileNotFoundError(f"Arquivo ausente: {f}")

        X_train = np.load(Path(self.dataset_dir, "x_train.npy")).astype("float32") / 255.0
        X_val = np.load(Path(self.dataset_dir, "x_val.npy")).astype("float32") / 255.0
        y_train_raw = np.load(Path(self.dataset_dir, "y_train.npy"), allow_pickle=True)
        y_val_raw = np.load(Path(self.dataset_dir, "y_val.npy"), allow_pickle=True)

        if X_train.ndim == 3:
            X_train = np.expand_dims(X_train, -1)
            X_val = np.expand_dims(X_val, -1)

        y_train = [clean_label(y) for y in y_train_raw]
        y_val = [clean_label(y) for y in y_val_raw]

        # ======== SINGLE CHAR (50x40) ========
        if self.type_name == "single_char":
            new_X_train, new_y_train = [], []
            new_X_val, new_y_val = [], []

            for img, lbl in zip(X_train, y_train):
                lbl = normalize_single_char_label(lbl)
                if lbl:
                    new_X_train.append(img)
                    new_y_train.append(lbl)

            for img, lbl in zip(X_val, y_val):
                lbl = normalize_single_char_label(lbl)
                if lbl:
                    new_X_val.append(img)
                    new_y_val.append(lbl)

            X_train = np.array(new_X_train, dtype="float32")
            X_val = np.array(new_X_val, dtype="float32")
            y_train, y_val = new_y_train, new_y_val

        # ======== FIVE CHAR (50x200) ========
        else:
            y_train = [pad_5chars(t) for t in y_train]
            y_val = [pad_5chars(t) for t in y_val]

        self.char_set, self.char_to_idx = sane_char_set(
            y_train, y_val, self.type_name, self.subtype
        )
        self.num_classes = len(self.char_set)

        print(f"🔡 Tipo: {self.type_name} | Subtipo: {self.subtype}")
        print(f"🔡 Charset ({self.num_classes}): {self.char_set}")

        return X_train, y_train, X_val, y_val

    # =====================================================
    #  Criar modelo (input adaptativo)
    # =====================================================
    def build_model(self):
        input_width = IMG_WIDTH_SINGLE if self.type_name == "single_char" else IMG_WIDTH_FIVE
        inputs = layers.Input(shape=(IMG_HEIGHT, input_width, IMG_CHANNELS))

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        opt = tf.keras.optimizers.Adam(1e-4)

        if self.type_name == "single_char":
            out = layers.Dense(self.num_classes, activation="softmax")(x)
            model = models.Model(inputs, out)
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            outputs = [
                layers.Dense(self.num_classes, activation="softmax", name=f"char_{i}")(x)
                for i in range(5)
            ]
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer=opt,
                loss=["categorical_crossentropy"] * 5,
                metrics=["accuracy"] * 5,
                loss_weights=[0.2] * 5,
            )

        return model

    # =====================================================
    #  Encode labels
    # =====================================================
    def encode_labels(self, y):
        if self.type_name == "single_char":
            idx = [self.char_to_idx[c] for c in y]
            return tf.keras.utils.to_categorical(idx, self.num_classes)

        arr = np.zeros((len(y), 5), dtype="int32")
        for i, text in enumerate(y):
            arr[i] = [self.char_to_idx[c] for c in text]

        return [
            tf.keras.utils.to_categorical(arr[:, pos], self.num_classes)
            for pos in range(5)
        ]

    # =====================================================
    #  Treinar com tf.data acelerado
    # =====================================================
    def run(self, epochs, batch_size):

        print(f"\n📌 Dataset: {self.dataset_dir}")
        X_train, y_train, X_val, y_val = self.load_data()

        model = self.build_model()

        y_train_enc = self.encode_labels(y_train)
        y_val_enc = self.encode_labels(y_val)

        # Criar pipeline turbo com tf.data
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_enc))
        train_ds = train_ds.cache().shuffle(8000).batch(batch_size).prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_enc))
        val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

        # Salvamento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base = self.model_name.replace(".keras", "")

        best_path = self.apurados_dir / f"{base}_{timestamp}_best.keras"
        final_path = self.apurados_dir / f"{base}_{timestamp}.keras"
        hist_path = self.apurados_dir / f"{base}_{timestamp}_history.json"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        ]

        print(f"🚀 Treinando modelo: {self.model_name}")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        with open(hist_path, "w") as f:
            json.dump(history.history, f, indent=4)

        model.save(final_path)
        print(f"✅ Modelo salvo em: {final_path}")

        return model


# ===========================================================
# 🔹 Unificação
# ===========================================================

def salvar_unificado(modelos, saida):
    if not modelos:
        print("⚠ Nenhum modelo para unificar.")
        return

    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH_FIVE, IMG_CHANNELS))
    feats = []

    for m in modelos:
        out = m(inp)
        if isinstance(out, list):
            feats.extend([layers.Flatten()(o) for o in out])
        else:
            feats.append(layers.Flatten()(out))

    merged = layers.Concatenate()(feats)
    unified = models.Model(inp, merged)
    unified.save(saida)

    print(f"✅ Modelo unificado salvo: {saida}")


# ===========================================================
# CLI
# ===========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de modelos CAPTCHA")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_dir = Path("data/processed")
    modelos = []

    for tipo in ["single_char", "five_char"]:
        tipo_dir = base_dir / tipo
        if not tipo_dir.exists():
            continue

        for sub in sorted(os.listdir(tipo_dir)):
            ds_dir = tipo_dir / sub
            if not ds_dir.is_dir():
                continue

            model_name = f"{tipo}_{sub}_model.keras"

            try:
                trainer = Trainer(str(ds_dir), model_name)
                modelo = trainer.run(args.epochs, args.batch_size)
                modelos.append(modelo)
            except Exception as e:
                print(f"💥 Falha ao treinar {model_name}: {e}")

    if modelos:
        salvar_unificado(modelos, "models/apurados/all_models_unified.keras")
