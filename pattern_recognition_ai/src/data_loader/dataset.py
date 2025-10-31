import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)
class CaptchaDataLoader:
    """Carrega imagens e rótulos diretamente da estrutura de pastas e garante que as pastas existam."""

    TYPES = ["raw", "single_char", "five_char"]
    SUBTYPES = ["numb", "world", "alphanumeric"]

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_dir = config["dataset_dir"]
        self.charset = config["charset"]
        self.char_to_num = {char: idx for idx, char in enumerate(self.charset)}
        self._create_folders()
    
    def get_dataset(self, batch_size=32):
        x_train, x_val, y_train_dict, y_val_dict = self.load_data()

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_dict))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val_dict))

        train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset



    def _create_folders(self):
        """Cria as pastas padrão se não existirem."""
        for t in self.TYPES:
            for st in self.SUBTYPES:
                path = os.path.join(self.dataset_dir, t, st)
                os.makedirs(path, exist_ok=True)
        print("✅ Pastas do dataset verificadas/criadas.")

    def _encode_label(self, label: str) -> np.ndarray:
        encoded = np.zeros((self.config["max_length"], self.config["num_classes"]), dtype=np.float32)
        for i, char in enumerate(label):
            if i >= self.config["max_length"]:
                break
            if char in self.char_to_num:
                encoded[i, self.char_to_num[char]] = 1.0
        return encoded

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        images = []
        labels = []
        folder_stats = {}

        for subdir, _, files in os.walk(self.dataset_dir):
            img_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not img_files:
                continue

            folder_stats[subdir] = len(img_files)

            for file in img_files:
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (self.config["image_width"], self.config["image_height"]))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)
                images.append(img)

                label = os.path.splitext(file)[0]
                labels.append(self._encode_label(label))

        # Mostrar estatísticas
        print("📂 Estatísticas do dataset:")
        for folder, count in folder_stats.items():
            print(f" - {folder}: {count} imagens")

        images = np.asarray(images)
        labels = np.asarray(labels)
        label_splits = [labels[:, i, :] for i in range(self.config["max_length"])]

        split = train_test_split(
            images,
            *label_splits,
            test_size=self.config["validation_split"],
            random_state=42,
            shuffle=True
        )

        x_train = split[0]
        x_val = split[1]
        y_train_splits = split[2::2]
        y_val_splits = split[3::2]

        y_train_dict = {f"char_{i + 1}": np.asarray(y_train_splits[i]) for i in range(self.config["max_length"])}
        y_val_dict = {f"char_{i + 1}": np.asarray(y_val_splits[i]) for i in range(self.config["max_length"])}

        print(f"\n✅ Total de imagens: {len(images)}")
        print(f" - Treino: {len(x_train)}")
        print(f" - Validação: {len(x_val)}\n")

        return x_train, x_val, y_train_dict, y_val_dict
    
    def load_data_from_path(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Carrega imagens e labels de uma pasta específica, já dividindo em treino e validação.
        """
        images = []
        labels = []

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (self.config["image_width"], self.config["image_height"]))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)

            label = os.path.splitext(file)[0]
            labels.append(self._encode_label(label))

        images = np.asarray(images)
        labels = np.asarray(labels)
        label_splits = [labels[:, i, :] for i in range(self.config["max_length"])]

        split = train_test_split(
            images,
            *label_splits,
            test_size=self.config["validation_split"],
            random_state=42,
            shuffle=True
        )

        x_train = split[0]
        x_val = split[1]
        y_train_splits = split[2::2]
        y_val_splits = split[3::2]

        y_train_dict = {f"char_{i + 1}": np.asarray(y_train_splits[i]) for i in range(self.config["max_length"])}
        y_val_dict = {f"char_{i + 1}": np.asarray(y_val_splits[i]) for i in range(self.config["max_length"])}

        return x_train, x_val, y_train_dict, y_val_dict
